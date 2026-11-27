#!/usr/bin/env python3
"""Fase 4: Entrenamiento, Validación Espacial + Temporal, Explicabilidad y Mapa de Riesgo.

Aplica estrictamente los guardrails aprobados:
  - Excluye capas de cobertura solo periférica (drenaje municipal/colectores fuera de SIAPA).
  - Evaluación 1: Spatial Block Cross-Validation (5 folds en bloques de 4 km x 4 km).
  - Evaluación 2: Validación Temporal estricta (2020-2025 -> 2026) usando SOLO variables pre-2026.
  - Explicabilidad agrupada en los 3 pilares causales (Uso de suelo/Densificación, Inundaciones/Topografía, Presión Drenaje).

Salidas:
  - outputs/risk_map_zmg_250m.tif       (GeoTIFF georreferenciado EPSG:32613 para QGIS)
  - outputs/risk_cells_zmg.geojson      (polígonos de riesgo Alto/Muy Alto + top_drivers para web/QGIS)
  - outputs/model_report.md             (métricas completas y comparativa)
  - outputs/model/shap_summary.png      (importancia de variables y pilares)
  - outputs/model/risk_map_preview.png  (mapa de riesgo con socavones reales superpuestos)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from geo_io import write_geojson

try:
    import lightgbm as lgb

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# Regla #3: excluir variables con sesgo de cobertura SIAPA vs municipios periféricos
EXCLUDED_BIASED_COLS = {
    "colector_segments_n",
    "dist_colector_m",
    "has_drenaje_municipal",
}

NON_FEATURE_COLS = {
    "cell_id",
    "grid_row",
    "grid_col",
    "x_center",
    "y_center",
    "geometry",
    "spatial_block_id",
    "is_urban_domain",
    "sinkhole_count",
    "has_sinkhole",
    "sinkhole_count_2020_2025",
    "has_sinkhole_2020_2025",
    "sinkhole_count_2026",
    "has_sinkhole_2026",
}

LABELS_ES = {
    "cus_intensidad_max": "Intensidad CUS máx (densificación)",
    "has_cus_incremento": "Zona con incremento CUS",
    "has_rtd_densificacion": "Zona receptora RTD (densificación)",
    "densidad_pob_2020_mean": "Densidad poblacional media",
    "densidad_pob_2020_max": "Densidad poblacional máx",
    "dinamica_consolidacion": "Zona consolidada (2010–2020)",
    "dinamica_crecimiento": "Zona en crecimiento (2010–2020)",
    "dinamica_disminucion": "Colonia antigua / despoblamiento",
    "licencias_urb_500m": "Licencias urbanización (500 m)",
    "licencias_urb_2018_2025_500m": "Licencias recientes 2018–2025 (500 m)",
    "espacios_verdes_n": "Espacios verdes en celda",
    "ndvi_2020_mean": "Vegetación NDVI 2020",
    "ndvi_2026_mean": "Vegetación NDVI 2026",
    "delta_ndvi_2020_2026_mean": "Cambio vegetación ΔNDVI (media)",
    "delta_ndvi_2020_2026_p10": "Pérdida extrema vegetación ΔNDVI (p10)",
    "ndbi_2020_mean": "Superficie pavimentada NDBI 2020",
    "ndbi_2026_mean": "Superficie pavimentada NDBI 2026",
    "delta_ndbi_2020_2026_mean": "Cambio pavimento ΔNDBI (media)",
    "delta_ndbi_2020_2026_p90": "Nueva obra/pavimento ΔNDBI (p90)",
    "riesgo_inundacion_mean": "Riesgo inundación IMEPLAN (media)",
    "riesgo_inundacion_max": "Riesgo inundación IMEPLAN (máx)",
    "riesgo_inundacion_high": "Riesgo inundación Alto/Muy alto",
    "mndwi_wet_2026_mean": "Humedad superficial MNDWI 2026",
    "mndwi_wet_2026_p90": "Humedad extrema MNDWI 2026 (p90)",
    "dem_elevation_mean": "Elevación DEM (m)",
    "slope_deg_mean": "Pendiente media (°)",
    "slope_deg_p90": "Pendiente máx p90 (°)",
    "tpi_local_mean": "Depresión topográfica TPI (media)",
    "tpi_local_p10": "Depresión topográfica TPI (p10)",
    "presion_drenaje_mean": "Presión drenaje AMG (media)",
    "presion_drenaje_max": "Presión drenaje AMG (máx)",
    "presion_drenaje_high": "Presión drenaje ALTA",
}


def feature_pillar(col: str) -> str:
    if col.startswith("presion_drenaje"):
        return "Presión sobre el drenaje"
    if col.startswith(("riesgo_inundacion", "mndwi", "dem_", "slope_", "tpi_")):
        return "Inundaciones y topografía"
    return "Uso de suelo y densificación"


def capture_at_k(y_true: np.ndarray, scores: np.ndarray, frac: float) -> float:
    """Fracción de positivos reales capturados dentro del Top-k% de celdas de mayor riesgo."""
    if y_true.sum() == 0:
        return 0.0
    n_top = max(1, int(round(len(scores) * frac)))
    top_idx = np.argsort(scores)[::-1][:n_top]
    return float(y_true[top_idx].sum() / y_true.sum())


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "LogReg_L2": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(C=0.15, class_weight="balanced", max_iter=1000, random_state=random_state)),
            ]
        ),
        "RandomForest_Reg": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=20,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    if HAS_LGBM:
        models["LightGBM_Reg"] = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    lgb.LGBMClassifier(
                        n_estimators=150,
                        learning_rate=0.03,
                        num_leaves=7,
                        max_depth=3,
                        min_child_samples=40,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=1.0,
                        reg_lambda=5.0,
                        is_unbalance=True,
                        verbose=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    else:
        models["HistGB_Reg"] = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=120,
                        learning_rate=0.04,
                        max_leaf_nodes=8,
                        min_samples_leaf=40,
                        l2_regularization=5.0,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return models


def evaluate_spatial_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_splits: int = 5,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    X = df[feature_cols].to_numpy(dtype="float32")
    y = df[target_col].to_numpy(dtype=int)
    groups = df["spatial_block_id"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = build_models()
    metrics: dict[str, dict[str, float]] = {}
    oof_preds: dict[str, np.ndarray] = {}

    for name, pipe in models.items():
        oof = np.zeros(len(df), dtype="float64")
        for tr_idx, va_idx in cv.split(X, y, groups):
            pipe.fit(X[tr_idx], y[tr_idx])
            oof[va_idx] = pipe.predict_proba(X[va_idx])[:, 1]
        oof_preds[name] = oof
        metrics[name] = {
            "roc_auc": float(roc_auc_score(y, oof)),
            "pr_auc": float(average_precision_score(y, oof)),
            "capture_top10": capture_at_k(y, oof, 0.10),
            "capture_top20": capture_at_k(y, oof, 0.20),
            "capture_top30": capture_at_k(y, oof, 0.30),
        }

    # Ensamble suave (promedio de percentiles entre los modelos para mayor estabilidad espacial)
    rank_stack = np.vstack([pd.Series(v).rank(pct=True).to_numpy() for v in oof_preds.values()])
    ens_oof = rank_stack.mean(axis=0)
    oof_preds["Ensemble_Rank"] = ens_oof
    metrics["Ensemble_Rank"] = {
        "roc_auc": float(roc_auc_score(y, ens_oof)),
        "pr_auc": float(average_precision_score(y, ens_oof)),
        "capture_top10": capture_at_k(y, ens_oof, 0.10),
        "capture_top20": capture_at_k(y, ens_oof, 0.20),
        "capture_top30": capture_at_k(y, ens_oof, 0.30),
    }
    return metrics, oof_preds


def evaluate_temporal_holdout(
    df: pd.DataFrame,
    pre2026_cols: list[str],
) -> dict[str, dict[str, float]]:
    """Regla #4: entrena con 2020-2025 usando SOLO variables pre-2026 y evalúa en 2026."""
    X = df[pre2026_cols].to_numpy(dtype="float32")
    y_train = df["has_sinkhole_2020_2025"].to_numpy(dtype=int)
    y_test_2026 = df["has_sinkhole_2026"].to_numpy(dtype=int)

    # También evaluamos SOLO en celdas sin socavón previo (casos 100% nuevos en 2026)
    new_mask = y_train == 0

    models = build_models()
    preds_all: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, float]] = {}

    for name, pipe in models.items():
        pipe.fit(X, y_train)
        p = pipe.predict_proba(X)[:, 1]
        preds_all[name] = p
        results[name] = {
            "roc_auc_2026": float(roc_auc_score(y_test_2026, p)),
            "pr_auc_2026": float(average_precision_score(y_test_2026, p)),
            "capture_top10_2026": capture_at_k(y_test_2026, p, 0.10),
            "capture_top20_2026": capture_at_k(y_test_2026, p, 0.20),
            "roc_auc_2026_new_cells": float(roc_auc_score(y_test_2026[new_mask], p[new_mask])),
            "capture_top20_2026_new_cells": capture_at_k(y_test_2026[new_mask], p[new_mask], 0.20),
        }

    ens = np.vstack([pd.Series(v).rank(pct=True).to_numpy() for v in preds_all.values()]).mean(axis=0)
    results["Ensemble_Rank"] = {
        "roc_auc_2026": float(roc_auc_score(y_test_2026, ens)),
        "pr_auc_2026": float(average_precision_score(y_test_2026, ens)),
        "capture_top10_2026": capture_at_k(y_test_2026, ens, 0.10),
        "capture_top20_2026": capture_at_k(y_test_2026, ens, 0.20),
        "roc_auc_2026_new_cells": float(roc_auc_score(y_test_2026[new_mask], ens[new_mask])),
        "capture_top20_2026_new_cells": capture_at_k(y_test_2026[new_mask], ens[new_mask], 0.20),
    }
    return results


def compute_exact_shap(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula contribuciones SHAP exactas (TreeSHAP en LightGBM + LinearSHAP en LogReg) y score final."""
    X = df[feature_cols].to_numpy(dtype="float32")
    y = df[target_col].to_numpy(dtype=int)

    models = build_models()
    fitted = {}
    probs = {}
    for name, pipe in models.items():
        pipe.fit(X, y)
        fitted[name] = pipe
        probs[name] = pipe.predict_proba(X)[:, 1]

    # Score final: promedio de percentiles del ensamble (escala 0 a 100, muy interpretable)
    ens_prob = np.vstack([probs[k] for k in probs]).mean(axis=0)
    ens_rank_pct = np.vstack([pd.Series(probs[k]).rank(pct=True).to_numpy() for k in probs]).mean(axis=0)

    # SHAP exacto con LightGBM TreeSHAP (pred_contrib=True) o fallback Linear SHAP de LogReg
    if HAS_LGBM and "LightGBM_Reg" in fitted:
        pipe = fitted["LightGBM_Reg"]
        X_imp = pipe.named_steps["imp"].transform(X)
        contrib = pipe.named_steps["clf"].predict(X_imp, pred_contrib=True)
        shap_vals = np.asarray(contrib)[:, :-1]  # última columna es el bias/base value
    else:
        pipe = fitted["LogReg_L2"]
        X_std = pipe.named_steps["sc"].transform(pipe.named_steps["imp"].transform(X))
        coefs = pipe.named_steps["clf"].coef_[0]
        shap_vals = (X_std - X_std.mean(axis=0)) * coefs

    return ens_prob, ens_rank_pct, shap_vals


def write_geotiff_epsg32613(
    arr: np.ndarray,
    out_path: Path,
    xmin: float,
    ymax: float,
    cell_size: float,
) -> None:
    """Escribe un GeoTIFF georreferenciado estándar EPSG:32613 en Python puro (compatible QGIS)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Tags GeoTIFF estándar: ModelPixelScaleTag (33550), ModelTiepointTag (33922), GeoKeyDirectoryTag (34735)
    # EPSG:32613 = WGS 84 / UTM zone 13N
    extratags = [
        (33550, "d", 3, (float(cell_size), float(cell_size), 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, float(xmin), float(ymax), 0.0), False),
        (
            34735,
            "H",
            16,
            (
                1, 1, 0, 3,          # KeyDirectoryVersion, KeyRevision, MinorRevision, NumberOfKeys=3
                1024, 0, 1, 1,       # GTModelTypeGeoKey = ModelTypeProjected (1)
                1025, 0, 1, 1,       # GTRasterTypeGeoKey = RasterPixelIsArea (1)
                3072, 0, 1, 32613,   # ProjectedCSTypeGeoKey = EPSG:32613
            ),
            False,
        ),
    ]
    tifffile.imwrite(
        str(out_path),
        arr.astype("float32"),
        compression="zlib",
        extratags=extratags,
    )


def save_plots(
    full_grid: gpd.GeoDataFrame,
    urban_df: pd.DataFrame,
    feature_cols: list[str],
    shap_vals: np.ndarray,
    risk_raster: np.ndarray,
    out_model_dir: Path,
) -> tuple[pd.Series, pd.Series]:
    out_model_dir.mkdir(parents=True, exist_ok=True)

    # 1) Importancia global (mean |SHAP|) + por pilar
    mean_abs = pd.Series(np.abs(shap_vals).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    pillar_imp = mean_abs.groupby(feature_pillar).sum()
    pillar_pct = 100.0 * pillar_imp / pillar_imp.sum()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), dpi=160, gridspec_kw={"width_ratios": [1.8, 1]})

    top15 = mean_abs.head(15).sort_values()
    labels = [LABELS_ES.get(c, c) for c in top15.index]
    p_colors = {
        "Uso de suelo y densificación": "#e76f51",
        "Inundaciones y topografía": "#1d70b8",
        "Presión sobre el drenaje": "#2a9d8f",
    }
    bar_cols = [p_colors[feature_pillar(c)] for c in top15.index]
    axes[0].barh(labels, top15.values, color=bar_cols, edgecolor="#333", lw=0.5)
    axes[0].set_title("Top 15 variables más influyentes (TreeSHAP |impacto medio|)", fontweight="bold")
    axes[0].set_xlabel("Impacto medio absoluto en el riesgo (SHAP)")
    axes[0].grid(axis="x", linestyle="--", alpha=0.5)

    p_sorted = pillar_pct.sort_values()
    bars = axes[1].barh(p_sorted.index, p_sorted.values, color=[p_colors[k] for k in p_sorted.index], edgecolor="#333", lw=0.5)
    for b, v in zip(bars, p_sorted.values):
        axes[1].text(v + 1.0, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", fontweight="bold")
    axes[1].set_title("Aporte relativo por pilar causal", fontweight="bold")
    axes[1].set_xlabel("% del poder explicativo total")
    axes[1].set_xlim(0, max(p_sorted.values) * 1.25)
    axes[1].grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_model_dir / "shap_summary.png", bbox_inches="tight")
    plt.close(fig)

    # 2) Mapa de riesgo visual con socavones reales encima
    fig, ax = plt.subplots(figsize=(10, 10), dpi=160)
    im = ax.imshow(risk_raster, cmap="YlOrRd", vmin=0, vmax=100)
    pos = urban_df[urban_df["has_sinkhole"] > 0]
    ax.scatter(
        pos["grid_col"],
        pos["grid_row"],
        s=18,
        facecolors="none",
        edgecolors="#0b3954",
        linewidths=1.1,
        label=f"Socavones reales 2020–2026 ({len(pos)} celdas)",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cbar.set_label("Índice de riesgo de socavón (percentil 0–100)")
    ax.set_title("Mapa Predictivo de Riesgo de Socavones — ZMG (250 m)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor="white")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_model_dir / "risk_map_preview.png", bbox_inches="tight")
    plt.close(fig)

    return mean_abs, pillar_pct


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 4: Modelo predictivo, validación espacial/temporal y mapa de riesgo.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/grid_250m_features.parquet"))
    parser.add_argument("--rasters-meta", type=Path, default=Path("data/processed/rasters/rasters_metadata.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    full = gpd.read_parquet(args.dataset)
    meta = json.loads(args.rasters_meta.read_text(encoding="utf-8"))
    urban = full[full["is_urban_domain"] == 1].copy().reset_index(drop=True)

    all_features = [c for c in full.columns if c not in NON_FEATURE_COLS and c not in EXCLUDED_BIASED_COLS]
    pre2026_features = [c for c in all_features if "2026" not in c]

    print(f"Dominio urbano: {len(urban)} celdas | Positivas: {int(urban['has_sinkhole'].sum())}")
    print(f"Features totales (sin sesgo SIAPA): {len(all_features)} | Pre-2026 (test temporal): {len(pre2026_features)}")

    print("\n1/4 Ejecutando Spatial Block Cross-Validation (5 folds, bloques 4 km)...")
    spatial_metrics, _ = evaluate_spatial_cv(urban, all_features, "has_sinkhole")
    for name, m in spatial_metrics.items():
        print(
            f"  {name:<18s} | ROC-AUC={m['roc_auc']:.3f} | PR-AUC={m['pr_auc']:.4f} | "
            f"Top10%={100*m['capture_top10']:.1f}% | Top20%={100*m['capture_top20']:.1f}%"
        )

    print("\n2/4 Ejecutando Validación Temporal estricta (train 2020-2025 -> test 2026, solo features pre-2026)...")
    temporal_metrics = evaluate_temporal_holdout(urban, pre2026_features)
    for name, m in temporal_metrics.items():
        print(
            f"  {name:<18s} | ROC-AUC 2026={m['roc_auc_2026']:.3f} | "
            f"Top10%={100*m['capture_top10_2026']:.1f}% | Top20%={100*m['capture_top20_2026']:.1f}% | "
            f"Celdas 100% nuevas Top20%={100*m['capture_top20_2026_new_cells']:.1f}%"
        )

    print("\n3/4 Entrenando ensamble final y calculando explicabilidad SHAP...")
    ens_prob, ens_rank, shap_vals = compute_exact_shap(urban, all_features, "has_sinkhole")
    urban["risk_score"] = np.round(100.0 * ens_rank, 1)
    urban["risk_prob_raw"] = np.round(ens_prob, 4)

    def assign_tier(score: float) -> str:
        if score >= 95.0:
            return "Muy Alto"
        if score >= 85.0:
            return "Alto"
        if score >= 60.0:
            return "Medio"
        return "Bajo"

    urban["risk_tier"] = [assign_tier(s) for s in urban["risk_score"]]

    # Top-3 factores positivos que empujan el riesgo en cada celda
    top_drivers = []
    for i in range(len(urban)):
        row_shap = shap_vals[i]
        top_idx = np.argsort(row_shap)[::-1][:3]
        drivers = [LABELS_ES.get(all_features[j], all_features[j]) for j in top_idx if row_shap[j] > 0]
        top_drivers.append(" | ".join(drivers))
    urban["top_drivers"] = top_drivers

    # Reconstruir matriz ráster 250 m (EPSG:32613)
    res_x, _, x0, _, res_y, y0 = meta["transform"]
    n_rows = int(full["grid_row"].max()) + 1
    n_cols = int(full["grid_col"].max()) + 1
    risk_raster = np.full((n_rows, n_cols), np.nan, dtype="float32")
    risk_raster[urban["grid_row"].to_numpy(), urban["grid_col"].to_numpy()] = urban["risk_score"].to_numpy(dtype="float32")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tif_path = args.out_dir / "risk_map_zmg_250m.tif"
    write_geotiff_epsg32613(risk_raster, tif_path, xmin=x0, ymax=y0, cell_size=250.0)

    # Exportar GeoJSON con celdas de riesgo Alto y Muy Alto (ligero y rápido para web y QGIS)
    high_risk = urban[urban["risk_tier"].isin(["Muy Alto", "Alto"])][
        [
            "cell_id",
            "risk_score",
            "risk_tier",
            "top_drivers",
            "sinkhole_count",
            "presion_drenaje_max",
            "riesgo_inundacion_max",
            "cus_intensidad_max",
            "delta_ndbi_2020_2026_mean",
            "geometry",
        ]
    ].copy()
    geojson_path = write_geojson(high_risk, args.out_dir / "risk_cells_zmg.geojson")

    # Guardar tabla completa de predicciones en parquet
    urban.to_parquet(args.out_dir / "risk_predictions_250m.parquet", index=False)

    print("\n4/4 Generando gráficas SHAP, mapa de riesgo y reporte...")
    mean_abs, pillar_pct = save_plots(full, urban, all_features, shap_vals, risk_raster, args.out_dir / "model")

    # Escribir reporte markdown
    report_lines = [
        "# Reporte del Modelo Predictivo — SocavonesGDL",
        "",
        f"- **Dominio urbano evaluado:** `{len(urban):,}` celdas de 250 m × 250 m (`{int(urban['has_sinkhole'].sum())}` celdas con socavones reales 2020–2026)",
        f"- **Variables usadas (sin sesgo SIAPA periférico):** `{len(all_features)}` (y `{len(pre2026_features)}` estrictamente pre-2026 en validación temporal)",
        f"- **Baseline aleatorio:** tasa base = `{100*urban['has_sinkhole'].mean():.2f}%` (en un modelo al azar, Top-10% capturaría solo 10% y Top-20% solo 20%)",
        "",
        "## 1. Validación Espacial (5-Fold Spatial Block CV, bloques de 4 km × 4 km)",
        "",
        "| Modelo | ROC-AUC | PR-AUC | Captura Top-10% | Captura Top-20% | Captura Top-30% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in spatial_metrics.items():
        report_lines.append(
            f"| `{name}` | {m['roc_auc']:.3f} | {m['pr_auc']:.4f} | {100*m['capture_top10']:.1f}% | {100*m['capture_top20']:.1f}% | {100*m['capture_top30']:.1f}% |"
        )

    report_lines += [
        "",
        "## 2. Validación Temporal Estricta (Entrenamiento 2020–2025 → Predicción 2026, solo variables pre-2026)",
        "",
        "| Modelo | ROC-AUC 2026 | Captura Top-10% (2026) | Captura Top-20% (2026) | ROC-AUC celdas nuevas 2026 | Captura Top-20% nuevas |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in temporal_metrics.items():
        report_lines.append(
            f"| `{name}` | {m['roc_auc_2026']:.3f} | {100*m['capture_top10_2026']:.1f}% | {100*m['capture_top20_2026']:.1f}% | {m['roc_auc_2026_new_cells']:.3f} | {100*m['capture_top20_2026_new_cells']:.1f}% |"
        )

    report_lines += [
        "",
        "## 3. Aporte Explicativo por Pilar Causal (TreeSHAP)",
        "",
        "| Pilar Causal | % del poder explicativo |",
        "|---|---:|",
    ]
    for pillar, pct in pillar_pct.sort_values(ascending=False).items():
        report_lines.append(f"| {pillar} | **{pct:.1f}%** |")

    report_lines += [
        "",
        "## 4. Top 10 Variables Más Influyentes",
        "",
        "| Variable | Pilar | Impacto medio |SHAP| |",
        "|---|---|---:|",
    ]
    for col, val in mean_abs.head(10).items():
        report_lines.append(f"| {LABELS_ES.get(col, col)} (`{col}`) | {feature_pillar(col)} | {val:.4f} |")

    report_lines += [
        "",
        "## 5. Archivos Generados",
        f"- GeoTIFF para QGIS: `{tif_path}`",
        f"- GeoJSON de zonas de riesgo Alto / Muy Alto ({len(high_risk):,} celdas): `{geojson_path}`",
        f"- Gráfica SHAP: `{args.out_dir / 'model' / 'shap_summary.png'}`",
        f"- Vista previa mapa de riesgo: `{args.out_dir / 'model' / 'risk_map_preview.png'}`",
    ]
    (args.out_dir / "model_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n--- Aporte por Pilar Causal (TreeSHAP) ---")
    for pillar, pct in pillar_pct.sort_values(ascending=False).items():
        print(f"  {pillar:<32s}: {pct:.1f}%")
    print(f"\nSalidas escritas en: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
