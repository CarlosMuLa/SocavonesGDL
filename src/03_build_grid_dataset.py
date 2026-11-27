#!/usr/bin/env python3
"""Fase 3: Construcción de la malla espacial 250 m x 250 m e ingeniería de variables.

Cruza en EPSG:32613:
  1) Los 10 GeoTIFFs satelitales y topográficos de la Fase 2 (data/processed/rasters/)
  2) Las 9 capas oficiales de IMEPLAN (data/imeplan/)
  3) Los 129 socavones validados de la Fase 1 (data/processed/sinkholes_clean.parquet)

Salidas:
  - data/processed/grid_250m_features.parquet
  - data/processed/grid_250m_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import tifffile
from shapely.geometry import box

from geo_io import read_vector

TARGET_CRS = "EPSG:32613"

PRESION_MAP = {"baja": 1.0, "media": 2.0, "alta": 3.0}
INUNDACION_MAP = {
    "muy bajo": 1.0,
    "bajo": 2.0,
    "medio": 3.0,
    "alto": 4.0,
    "muy alto": 5.0,
}


def find_first(base: Path, pattern: str) -> Optional[Path]:
    hits = sorted(base.rglob(pattern))
    return hits[0] if hits else None


def parse_density_range(val: object) -> float:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return np.nan
    s = str(val).strip()
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not nums:
        return np.nan
    return float(np.mean(nums))


def parse_float_safe(val: object) -> float:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return np.nan
    s = str(val).strip().replace(",", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def build_grid(meta: dict, cell_size_m: float, amg_geom) -> tuple[gpd.GeoDataFrame, int, int, float, float]:
    """Construye la malla regular de cell_size_m alineada al origen de los rásters."""
    res_x, _, x0, _, res_y, y0 = meta["transform"]
    height, width = meta["shape"]
    xmin = x0
    ymax = y0
    xmax = x0 + width * res_x
    ymin = y0 + height * res_y  # res_y es negativo (-20.0)

    n_cols = int(math.floor((xmax - xmin) / cell_size_m))
    n_rows = int(math.floor((ymax - ymin) / cell_size_m))

    records = []
    geoms = []
    cid = 0
    for r in range(n_rows):
        cy_max = ymax - r * cell_size_m
        cy_min = cy_max - cell_size_m
        for c in range(n_cols):
            cx_min = xmin + c * cell_size_m
            cx_max = cx_min + cell_size_m
            poly = box(cx_min, cy_min, cx_max, cy_max)
            records.append(
                {
                    "cell_id": cid,
                    "grid_row": r,
                    "grid_col": c,
                    "x_center": (cx_min + cx_max) / 2.0,
                    "y_center": (cy_min + cy_max) / 2.0,
                }
            )
            geoms.append(poly)
            cid += 1

    grid = gpd.GeoDataFrame(records, geometry=geoms, crs=TARGET_CRS)
    if amg_geom is not None:
        inside = grid.geometry.intersects(amg_geom)
        grid = grid[inside].reset_index(drop=True)
    return grid, n_rows, n_cols, xmin, ymax


def extract_raster_zonal_stats(grid: gpd.GeoDataFrame, rasters_dir: Path, meta: dict, cell_size_m: float) -> pd.DataFrame:
    """Extrae estadísticas zonales exactas para cada celda usando mapeo de píxeles rápido."""
    res_x, _, x0, _, res_y, y0 = meta["transform"]
    px_size = abs(res_x)
    height, width = meta["shape"]

    # Precalcular slice de filas y columnas del ráster para cada celda
    row_starts = np.floor((grid["grid_row"].values * cell_size_m) / px_size).astype(int)
    row_ends = np.minimum(height, np.ceil(((grid["grid_row"].values + 1) * cell_size_m) / px_size).astype(int))
    col_starts = np.floor((grid["grid_col"].values * cell_size_m) / px_size).astype(int)
    col_ends = np.minimum(width, np.ceil(((grid["grid_col"].values + 1) * cell_size_m) / px_size).astype(int))

    extra_quantiles = {
        "delta_ndbi_2020_2026": ("p90", 90),
        "delta_ndvi_2020_2026": ("p10", 10),
        "mndwi_wet_2026": ("p90", 90),
        "slope_deg": ("p90", 90),
        "tpi_local": ("p10", 10),
    }

    out = pd.DataFrame(index=grid.index)
    for layer in meta["layers"]:
        tif_path = rasters_dir / f"{layer}.tif"
        arr = tifffile.imread(tif_path).astype("float32")
        means = np.full(len(grid), np.nan, dtype="float32")
        q_spec = extra_quantiles.get(layer)
        q_vals = np.full(len(grid), np.nan, dtype="float32") if q_spec else None

        for i in range(len(grid)):
            patch = arr[row_starts[i] : row_ends[i], col_starts[i] : col_ends[i]]
            valid = patch[np.isfinite(patch)]
            if valid.size > 0:
                means[i] = float(np.mean(valid))
                if q_spec is not None and q_vals is not None:
                    q_vals[i] = float(np.percentile(valid, q_spec[1]))

        out[f"{layer}_mean"] = means
        if q_spec is not None and q_vals is not None:
            out[f"{layer}_{q_spec[0]}"] = q_vals
        print(f"  [Raster] {layer}: OK (media global={np.nanmean(means):.3f})")
    return out


def join_polygon_stats(
    grid: gpd.GeoDataFrame,
    poly_gdf: gpd.GeoDataFrame,
    value_col: str,
    prefix: str,
    high_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Cruza polígonos contra las celdas de la malla y agrega mean/max (y opcional flag alto)."""
    sub = poly_gdf[[value_col, "geometry"]].dropna(subset=[value_col]).copy()
    sub = sub[sub.geometry.notna() & (~sub.geometry.is_empty)]
    joined = gpd.sjoin(grid[["cell_id", "geometry"]], sub, how="left", predicate="intersects")
    grp = joined.groupby("cell_id")[value_col]
    res = pd.DataFrame(index=grid["cell_id"])
    res[f"{prefix}_mean"] = grp.mean()
    res[f"{prefix}_max"] = grp.max()
    if high_threshold is not None:
        res[f"{prefix}_high"] = (res[f"{prefix}_max"] >= high_threshold).astype("float32")
    return res.reset_index(drop=True)


def add_imeplan_features(grid: gpd.GeoDataFrame, imeplan_dir: Path) -> pd.DataFrame:
    """Extrae features de las 9 capas oficiales de IMEPLAN agrupadas por los 3 pilares."""
    feats = pd.DataFrame(index=grid.index)
    grid_bbox = box(*grid.total_bounds)

    # --- PILAR 1: DRENAJE / TUBERÍAS ---
    p_presion = find_first(imeplan_dir, "amg_presion_drenaje_*.shp")
    if p_presion:
        gdf = read_vector(p_presion).to_crs(TARGET_CRS)
        gdf = gdf[gdf.geometry.intersects(grid_bbox)].copy()
        gdf["presion_num"] = gdf["presion"].astype(str).str.strip().str.lower().map(PRESION_MAP)
        st = join_polygon_stats(grid, gdf, "presion_num", "presion_drenaje", high_threshold=3.0)
        for col in st.columns:
            feats[col] = st[col].fillna(0.0).values
        print(f"  [IMEPLAN] Presión drenaje: OK ({len(gdf)} polígonos)")

    p_colectores = find_first(imeplan_dir, "amg_red_colectores_*.shp")
    if p_colectores:
        gdf = read_vector(p_colectores).to_crs(TARGET_CRS)
        gdf = gdf[gdf.geometry.intersects(grid_bbox)].copy()
        joined = gpd.sjoin(grid[["cell_id", "geometry"]], gdf[["geometry"]], how="left", predicate="intersects")
        counts = joined.groupby("cell_id")["index_right"].count().reindex(grid["cell_id"]).fillna(0).values
        feats["colector_segments_n"] = counts.astype("float32")
        # Distancia desde el centroide de la celda al colector municipal más cercano
        if len(gdf) > 0:
            centroids = gpd.GeoDataFrame({"cell_id": grid["cell_id"]}, geometry=grid.geometry.centroid, crs=TARGET_CRS)
            nearest = gpd.sjoin_nearest(centroids, gdf[["geometry"]], how="left", distance_col="dist_colector_m")
            nearest = nearest.groupby("cell_id")["dist_colector_m"].min().reindex(grid["cell_id"]).values
            feats["dist_colector_m"] = nearest.astype("float32")
        print(f"  [IMEPLAN] Red colectores municipales: OK ({len(gdf)} tramos)")

    p_dren_mun = find_first(imeplan_dir, "amg_drenajemunicipal_*.shp")
    if p_dren_mun:
        gdf = read_vector(p_dren_mun).to_crs(TARGET_CRS)
        gdf = gdf[gdf.geometry.intersects(grid_bbox)].copy()
        joined = gpd.sjoin(grid[["cell_id", "geometry"]], gdf[["geometry"]], how="left", predicate="intersects")
        has_dm = joined.groupby("cell_id")["index_right"].count().reindex(grid["cell_id"]).fillna(0) > 0
        feats["has_drenaje_municipal"] = has_dm.astype("float32").values
        print(f"  [IMEPLAN] Cobertura drenaje municipal: OK ({len(gdf)} polígonos)")

    # --- PILAR 2: INUNDACIONES ---
    p_inund = find_first(imeplan_dir, "amg_riesgo_inundaciones_*.shp")
    if p_inund:
        print("  [IMEPLAN] Cargando Riesgo de inundaciones (capa grande, ~15 s)...")
        gdf = read_vector(p_inund).to_crs(TARGET_CRS)
        gdf = gdf[gdf.geometry.intersects(grid_bbox)].copy()
        gdf["riesgo_num"] = gdf["nivel_de_r"].astype(str).str.strip().str.lower().map(INUNDACION_MAP)
        st = join_polygon_stats(grid, gdf, "riesgo_num", "riesgo_inundacion", high_threshold=4.0)
        for col in st.columns:
            feats[col] = st[col].fillna(0.0).values
        print(f"  [IMEPLAN] Riesgo de inundaciones: OK ({len(gdf)} polígonos en área)")

    # --- PILAR 3: CAMBIO DE USO DE SUELO / DENSIFICACIÓN ---
    p_dens = find_first(imeplan_dir, "vwAMG_densidad_2020*.shp")
    if p_dens:
        gdf = read_vector(p_dens).to_crs(TARGET_CRS)
        gdf = gdf[gdf.geometry.intersects(grid_bbox)].copy()
        gdf["dens_num"] = gdf["Densidad_d"].apply(parse_density_range)
        st = join_polygon_stats(grid, gdf, "dens_num", "densidad_pob_2020")
        feats["densidad_pob_2020_mean"] = st["densidad_pob_2020_mean"].fillna(0.0).values
        feats["densidad_pob_2020_max"] = st["densidad_pob_2020_max"].fillna(0.0).values
        print(f"  [IMEPLAN] Densidad población 2020: OK ({len(gdf)} polígonos)")

    p_din = find_first(imeplan_dir, "vw_AMG_Por_Dinamica_Pob_Urb.shp")
    if p_din:
        gdf = read_vector(p_din, encoding="latin1").to_crs(TARGET_CRS)
        gdf = gdf[gdf.geometry.intersects(grid_bbox)].copy()
        cls_low = gdf["Clasificac"].astype(str).str.lower()
        gdf["is_consolidacion"] = cls_low.str.contains("consolid").astype("float32")
        gdf["is_crecimiento"] = cls_low.str.contains("crecim").astype("float32")
        gdf["is_disminucion"] = cls_low.str.contains("dismin").astype("float32")
        # En colonias consolidadas/con despoblamiento (centro/antiguas) suele haber tubería más vieja
        for col in ["is_consolidacion", "is_crecimiento", "is_disminucion"]:
            st = join_polygon_stats(grid, gdf, col, f"dinamica_{col[3:]}")
            feats[f"dinamica_{col[3:]}"] = st[f"dinamica_{col[3:]}_mean"].fillna(0.0).values
        print(f"  [IMEPLAN] Dinámica poblacional 2010-2020: OK ({len(gdf)} polígonos)")

    p_lic = find_first(imeplan_dir, "amg_licencias_urbanizacion_*.shp")
    if p_lic:
        gdf = read_vector(p_lic).to_crs(TARGET_CRS)
        gdf = gdf[gdf.geometry.intersects(grid_bbox.buffer(500.0))].copy()
        gdf["ano_num"] = pd.to_numeric(gdf["ano_de_exp"], errors="coerce")
        # Conteo de licencias dentro de buffer de 500 m alrededor de cada celda
        grid_buf = gpd.GeoDataFrame({"cell_id": grid["cell_id"]}, geometry=grid.geometry.buffer(500.0), crs=TARGET_CRS)
        j_all = gpd.sjoin(grid_buf, gdf[["ano_num", "geometry"]], how="left", predicate="intersects")
        feats["licencias_urb_500m"] = (
            j_all.groupby("cell_id")["index_right"].count().reindex(grid["cell_id"]).fillna(0).astype("float32").values
        )
        recent = gdf[gdf["ano_num"] >= 2018]
        j_rec = gpd.sjoin(grid_buf, recent[["geometry"]], how="left", predicate="intersects")
        feats["licencias_urb_2018_2025_500m"] = (
            j_rec.groupby("cell_id")["index_right"].count().reindex(grid["cell_id"]).fillna(0).astype("float32").values
        )
        print(f"  [IMEPLAN] Licencias de urbanización: OK ({len(gdf)} puntos, {len(recent)} desde 2018)")

    p_icus = find_first(imeplan_dir, "vwGDL_ICUS_ICOS_v3.shp")
    p_cusmax = find_first(imeplan_dir, "vw_Poligonos_CUSMAX_*.shp")
    cus_frames = []
    for p_c in [p_icus, p_cusmax]:
        if p_c:
            g = read_vector(p_c, encoding="latin1").to_crs(TARGET_CRS)
            g = g[g.geometry.intersects(grid_bbox)].copy()
            # Extrae el coeficiente máximo numérico si está disponible
            cand_col = "Coeficien0" if "Coeficien0" in g.columns else None
            g["cus_val"] = g[cand_col].apply(parse_float_safe) if cand_col else 1.0
            cus_frames.append(g[["cus_val", "geometry"]])
    if cus_frames:
        cus_all = pd.concat(cus_frames, ignore_index=True)
        cus_gdf = gpd.GeoDataFrame(cus_all, geometry="geometry", crs=TARGET_CRS)
        st = join_polygon_stats(grid, cus_gdf, "cus_val", "cus_intensidad")
        feats["cus_intensidad_max"] = st["cus_intensidad_max"].fillna(0.0).values
        feats["has_cus_incremento"] = (st["cus_intensidad_max"].notna()).astype("float32").values
        print(f"  [IMEPLAN] Incremento CUS/ICUS: OK ({len(cus_gdf)} polígonos)")

    p_rtd = find_first(imeplan_dir, "vw_AMG_RTD_*.shp")
    if p_rtd:
        gdf = read_vector(p_rtd, encoding="latin1").to_crs(TARGET_CRS)
        gdf = gdf[gdf.geometry.intersects(grid_bbox)].copy()
        joined = gpd.sjoin(grid[["cell_id", "geometry"]], gdf[["geometry"]], how="left", predicate="intersects")
        has_rtd = joined.groupby("cell_id")["index_right"].count().reindex(grid["cell_id"]).fillna(0) > 0
        feats["has_rtd_densificacion"] = has_rtd.astype("float32").values
        print(f"  [IMEPLAN] Zonas receptoras RTD: OK ({len(gdf)} polígonos)")

    p_verde = find_first(imeplan_dir, "vwEspacio_verde_*.shp")
    if p_verde:
        gdf = read_vector(p_verde, encoding="latin1").to_crs(TARGET_CRS)
        gdf = gdf[gdf.geometry.intersects(grid_bbox)].copy()
        joined = gpd.sjoin(grid[["cell_id", "geometry"]], gdf[["geometry"]], how="left", predicate="intersects")
        cnt = joined.groupby("cell_id")["index_right"].count().reindex(grid["cell_id"]).fillna(0)
        feats["espacios_verdes_n"] = cnt.astype("float32").values
        print(f"  [IMEPLAN] Espacios verdes: OK ({len(gdf)} polígonos)")

    return feats


def attach_sinkhole_targets(grid: gpd.GeoDataFrame, sinkholes_path: Path) -> pd.DataFrame:
    """Asigna conteos y banderas de socavones (total, 2020-2025 train temporal y 2026 test temporal)."""
    pts = gpd.read_parquet(sinkholes_path).to_crs(TARGET_CRS)
    joined = gpd.sjoin(grid[["cell_id", "geometry"]], pts[["anio", "causa_normalizada", "geometry"]], how="left", predicate="contains")

    out = pd.DataFrame(index=grid["cell_id"])
    out["sinkhole_count"] = joined.groupby("cell_id")["index_right"].count().reindex(grid["cell_id"]).fillna(0).astype(int)
    out["has_sinkhole"] = (out["sinkhole_count"] > 0).astype(int)

    j_pre = joined[joined["anio"] <= 2025]
    out["sinkhole_count_2020_2025"] = j_pre.groupby("cell_id")["index_right"].count().reindex(grid["cell_id"]).fillna(0).astype(int)
    out["has_sinkhole_2020_2025"] = (out["sinkhole_count_2020_2025"] > 0).astype(int)

    j_26 = joined[joined["anio"] == 2026]
    out["sinkhole_count_2026"] = j_26.groupby("cell_id")["index_right"].count().reindex(grid["cell_id"]).fillna(0).astype(int)
    out["has_sinkhole_2026"] = (out["sinkhole_count_2026"] > 0).astype(int)

    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 3: Malla espacial 250 m + features IMEPLAN y satelitales.")
    parser.add_argument("--rasters-dir", type=Path, default=Path("data/processed/rasters"))
    parser.add_argument("--imeplan-dir", type=Path, default=Path("data/imeplan"))
    parser.add_argument("--sinkholes", type=Path, default=Path("data/processed/sinkholes_clean.parquet"))
    parser.add_argument("--cell-size", type=float, default=250.0)
    parser.add_argument("--block-size", type=float, default=4000.0, help="Tamaño de bloque espacial para CV (metros).")
    parser.add_argument("--out-parquet", type=Path, default=Path("data/processed/grid_250m_features.parquet"))
    parser.add_argument("--out-summary", type=Path, default=Path("data/processed/grid_250m_summary.json"))
    args = parser.parse_args()

    meta = json.loads((args.rasters_dir / "rasters_metadata.json").read_text(encoding="utf-8"))
    limite_path = find_first(args.imeplan_dir, "LimiteAMG.shp")
    amg_geom = read_vector(limite_path).to_crs(TARGET_CRS).geometry.union_all() if limite_path else None

    print(f"1/4 Construyendo malla de {args.cell_size:.0f} m x {args.cell_size:.0f} m...")
    grid, n_rows, n_cols, xmin, ymax = build_grid(meta, args.cell_size, amg_geom)
    print(f"    Celdas dentro de LimiteAMG: {len(grid)} (rejilla base {n_rows} x {n_cols})")

    # Bloques espaciales de 4 km x 4 km para Spatial Block CV sin fuga
    bx = np.floor((grid["x_center"].values - xmin) / args.block_size).astype(int)
    by = np.floor((ymax - grid["y_center"].values) / args.block_size).astype(int)
    grid["spatial_block_id"] = [f"B_{x}_{y}" for x, y in zip(bx, by)]

    print("\n2/4 Extrayendo estadísticas zonales de los 10 GeoTIFFs satelitales/DEM...")
    raster_feats = extract_raster_zonal_stats(grid, args.rasters_dir, meta, args.cell_size)

    print("\n3/4 Integrando las capas oficiales de IMEPLAN...")
    imeplan_feats = add_imeplan_features(grid, args.imeplan_dir)

    print("\n4/4 Cruzando con los 129 socavones validados...")
    targets = attach_sinkhole_targets(grid, args.sinkholes)

    full = pd.concat([grid, raster_feats, imeplan_feats, targets], axis=1)
    full = gpd.GeoDataFrame(full, geometry="geometry", crs=TARGET_CRS)

    # Dominio urbano real (evita entrenar sobre miles de celdas de bosque vacío sin calles/drenaje)
    full["is_urban_domain"] = (
        (full.get("densidad_pob_2020_max", 0) > 0)
        | (full.get("presion_drenaje_max", 0) > 0)
        | (full.get("has_drenaje_municipal", 0) > 0)
        | (full.get("has_cus_incremento", 0) > 0)
        | (full.get("ndbi_2026_mean", -1) > -0.10)
        | (full["has_sinkhole"] > 0)
    ).astype(int)

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(args.out_parquet, index=False)

    urban = full[full["is_urban_domain"] == 1]
    feature_cols = [c for c in list(raster_feats.columns) + list(imeplan_feats.columns) if c in full.columns]
    corrs = (
        urban[feature_cols]
        .corrwith(urban["has_sinkhole"])
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    summary = {
        "total_cells": int(len(full)),
        "urban_domain_cells": int(len(urban)),
        "total_sinkholes_matched": int(full["sinkhole_count"].sum()),
        "positive_cells_total": int(full["has_sinkhole"].sum()),
        "positive_cells_2020_2025": int(full["has_sinkhole_2020_2025"].sum()),
        "positive_cells_2026": int(full["has_sinkhole_2026"].sum()),
        "n_spatial_blocks_urban": int(urban["spatial_block_id"].nunique()),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "top_10_correlations_urban": {k: round(float(v), 4) for k, v in corrs.head(10).items()},
    }
    args.out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n--- Resumen Fase 3 ---")
    print(f"Celdas totales en malla 250 m: {summary['total_cells']}")
    print(f"Celdas en dominio urbano real: {summary['urban_domain_cells']}")
    print(f"Socavones mapeados a la malla: {summary['total_sinkholes_matched']} en {summary['positive_cells_total']} celdas positivas")
    print(f"  - Celdas positivas 2020-2025 (train temporal): {summary['positive_cells_2020_2025']}")
    print(f"  - Celdas positivas 2026 (test temporal):       {summary['positive_cells_2026']}")
    print(f"Variables construidas: {summary['n_features']} | Bloques espaciales (4 km): {summary['n_spatial_blocks_urban']}")
    print("\nTop 10 correlaciones simples con has_sinkhole (dominio urbano):")
    for k, v in summary["top_10_correlations_urban"].items():
        print(f"  {k:<32s}: {v:+.4f}")
    print(f"\nDataset guardado en: {args.out_parquet.resolve()}")


if __name__ == "__main__":
    main()
