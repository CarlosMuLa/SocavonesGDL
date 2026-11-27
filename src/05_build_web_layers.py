#!/usr/bin/env python3
"""Genera overlays PNG transparentes en Web Mercator (EPSG:3857) para el visor Leaflet.

Reproyecta exactamente desde EPSG:32613 a EPSG:3857 para que cada capa caiga
al milímetro sobre OpenStreetMap y Satélite Esri en Leaflet (L.imageOverlay).

Capas generadas en web/public/layers/:
  - delta_ndvi.png             (Pérdida de vegetación 2020-2026)
  - delta_ndbi.png             (Nuevo pavimento / urbanización 2020-2026)
  - mndwi_wet.png              (Humedad y agua superficial lluvias 2026)
  - tpi_depressions.png        (Depresiones y arroyos topográficos TPI)
  - imeplan_presion_drenaje.png (Presión sobre el drenaje AMG - IMEPLAN)
  - imeplan_inundaciones.png    (Riesgo de inundaciones AMG - IMEPLAN)
  - imeplan_cus.png             (Incremento CUS / densificación - IMEPLAN)
  - layers_meta.json            (bounds exactos WGS84 [[S,W],[N,E]] para Leaflet)
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import tifffile
from PIL import Image
from pyproj import Transformer

RASTERS_DIR = Path("data/processed/rasters")
GRID_PARQUET = Path("data/processed/grid_250m_features.parquet")
OUT_DIR = Path("web/public/layers")


def build_mercator_sampler(meta: dict, out_w: int = 1100):
    """Precalcula índices (row, col) en el ráster UTM 13N para cada píxel Web Mercator."""
    res_x, _, x0, _, res_y, y0 = meta["transform"]
    h, w = meta["shape"]
    xmin, ymax = x0, y0
    xmax, ymin = x0 + w * res_x, y0 + h * res_y

    to_wgs = Transformer.from_crs("EPSG:32613", "EPSG:4326", always_xy=True)
    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    merc_to_utm = Transformer.from_crs("EPSG:3857", "EPSG:32613", always_xy=True)

    # Bounding box WGS84 conservador de las 4 esquinas UTM
    corners_x = [xmin, xmax, xmin, xmax]
    corners_y = [ymin, ymin, ymax, ymax]
    lons, lats = to_wgs.transform(corners_x, corners_y)
    west, east = float(min(lons)), float(max(lons))
    south, north = float(min(lats)), float(max(lats))

    mx0, my0 = to_merc.transform(west, south)
    mx1, my1 = to_merc.transform(east, north)
    aspect = (my1 - my0) / (mx1 - mx0)
    out_h = int(round(out_w * aspect))

    xs = np.linspace(mx0, mx1, out_w, dtype="float64")
    ys = np.linspace(my1, my0, out_h, dtype="float64")
    gx, gy = np.meshgrid(xs, ys)
    ux, uy = merc_to_utm.transform(gx, gy)

    col_idx = np.floor((ux - x0) / res_x).astype("int32")
    row_idx = np.floor((y0 - uy) / abs(res_y)).astype("int32")
    in_bounds = (row_idx >= 0) & (row_idx < h) & (col_idx >= 0) & (col_idx < w)
    row_safe = np.clip(row_idx, 0, h - 1)
    col_safe = np.clip(col_idx, 0, w - 1)

    bounds_leaflet = [[round(south, 6), round(west, 6)], [round(north, 6), round(east, 6)]]
    return {
        "row_safe": row_safe,
        "col_safe": col_safe,
        "in_bounds": in_bounds,
        "out_h": out_h,
        "out_w": out_w,
        "ux": ux,
        "uy": uy,
        "x0": x0,
        "y0": y0,
        "bounds_leaflet": bounds_leaflet,
    }


def sample_raster(tif_path: Path, sampler: dict) -> np.ndarray:
    arr = tifffile.imread(tif_path).astype("float32")
    out = np.where(sampler["in_bounds"], arr[sampler["row_safe"], sampler["col_safe"]], np.nan)
    return out


def save_rgba_png(rgba: np.ndarray, out_path: Path) -> None:
    img = Image.fromarray(rgba, mode="RGBA")
    img.save(out_path, format="PNG", optimize=True)
    print(f"  {out_path.name}: {out_path.stat().st_size / 1024:.1f} KB")


def colorize_threshold(
    arr: np.ndarray,
    stops: list[tuple[float, tuple[int, int, int, int]]],
    invert: bool = False,
) -> np.ndarray:
    """Convierte un arreglo float32 a RGBA transparente aplicando umbrales ordenados."""
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype="uint8")
    valid = np.isfinite(arr)
    for thresh, color in stops:
        mask = valid & (arr <= thresh if invert else arr >= thresh)
        rgba[mask] = color
    return rgba


def sample_grid_250m(grid_df: gpd.GeoDataFrame, col: str, sampler: dict, cell_size: float = 250.0) -> np.ndarray:
    n_rows = int(grid_df["grid_row"].max()) + 1
    n_cols = int(grid_df["grid_col"].max()) + 1
    mat = np.full((n_rows, n_cols), np.nan, dtype="float32")
    mat[grid_df["grid_row"].to_numpy(), grid_df["grid_col"].to_numpy()] = grid_df[col].to_numpy(dtype="float32")

    g_col = np.floor((sampler["ux"] - sampler["x0"]) / cell_size).astype("int32")
    g_row = np.floor((sampler["y0"] - sampler["uy"]) / cell_size).astype("int32")
    ok = (g_row >= 0) & (g_row < n_rows) & (g_col >= 0) & (g_col < n_cols)
    r_s = np.clip(g_row, 0, n_rows - 1)
    c_s = np.clip(g_col, 0, n_cols - 1)
    return np.where(ok, mat[r_s, c_s], np.nan)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((RASTERS_DIR / "rasters_metadata.json").read_text(encoding="utf-8"))
    grid = gpd.read_parquet(GRID_PARQUET)

    print("Construyendo proyección Web Mercator (EPSG:3857)...")
    sampler = build_mercator_sampler(meta, out_w=1100)

    print("1/2 Generando overlays satelitales y topográficos (20 m)...")
    # 1) ΔNDVI: pérdida de vegetación (valores negativos fuertes)
    dndvi = sample_raster(RASTERS_DIR / "delta_ndvi_2020_2026.tif", sampler)
    rgba = colorize_threshold(
        dndvi,
        [
            (-0.08, (253, 186, 116, 145)),  # pérdida moderada (naranja claro)
            (-0.15, (234, 88, 12, 185)),    # pérdida fuerte (naranja intenso)
            (-0.25, (153, 27, 27, 220)),    # pérdida severa (rojo vino)
        ],
        invert=True,
    )
    save_rgba_png(rgba, OUT_DIR / "delta_ndvi.png")

    # 2) ΔNDBI: nueva pavimentación / construcción (valores positivos)
    dndbi = sample_raster(RASTERS_DIR / "delta_ndbi_2020_2026.tif", sampler)
    rgba = colorize_threshold(
        dndbi,
        [
            (0.03, (251, 191, 36, 135)),    # incremento leve (ámbar)
            (0.08, (220, 38, 38, 185)),     # nueva obra / pavimento (rojo)
            (0.15, (126, 34, 206, 215)),    # impermeabilización fuerte (morado)
        ],
        invert=False,
    )
    save_rgba_png(rgba, OUT_DIR / "delta_ndbi.png")

    # 3) MNDWI lluvias 2026: humedad y cuerpos de agua superficiales
    mndwi = sample_raster(RASTERS_DIR / "mndwi_wet_2026.tif", sampler)
    rgba = colorize_threshold(
        mndwi,
        [
            (-0.08, (56, 189, 248, 140)),   # superficie húmeda
            (0.02, (2, 132, 199, 190)),     # acumulación / humedad alta
            (0.15, (30, 58, 138, 225)),     # cuerpo de agua
        ],
        invert=False,
    )
    save_rgba_png(rgba, OUT_DIR / "mndwi_wet.png")

    # 4) TPI depresiones topográficas (arroyos/bajos donde escurre agua)
    tpi = sample_raster(RASTERS_DIR / "tpi_local.tif", sampler)
    rgba = colorize_threshold(
        tpi,
        [
            (-2.0, (45, 212, 191, 140)),    # depresión leve
            (-5.0, (13, 148, 136, 185)),    # cauce / bajo marcado
            (-10.0, (17, 94, 89, 220)),     # barranca / depresión profunda
        ],
        invert=True,
    )
    save_rgba_png(rgba, OUT_DIR / "tpi_depressions.png")

    print("2/2 Generando overlays oficiales IMEPLAN (malla 250 m)...")
    # 5) Presión sobre el drenaje AMG (1=Baja, 2=Media, 3=Alta)
    pres = sample_grid_250m(grid, "presion_drenaje_max", sampler)
    rgba = colorize_threshold(
        pres,
        [
            (1.0, (254, 240, 138, 120)),    # Baja
            (2.0, (249, 115, 22, 175)),     # Media
            (3.0, (153, 27, 27, 215)),      # Alta
        ],
        invert=False,
    )
    save_rgba_png(rgba, OUT_DIR / "imeplan_presion_drenaje.png")

    # 6) Riesgo de inundaciones AMG (3=Medio, 4=Alto, 5=Muy alto)
    inund = sample_grid_250m(grid, "riesgo_inundacion_max", sampler)
    rgba = colorize_threshold(
        inund,
        [
            (3.0, (125, 211, 252, 135)),    # Medio
            (4.0, (37, 99, 235, 185)),      # Alto
            (5.0, (49, 46, 129, 220)),      # Muy alto
        ],
        invert=False,
    )
    save_rgba_png(rgba, OUT_DIR / "imeplan_inundaciones.png")

    # 7) Densificación / Incremento CUS (IMEPLAN)
    cus = sample_grid_250m(grid, "cus_intensidad_max", sampler)
    rgba = colorize_threshold(
        cus,
        [
            (0.5, (232, 121, 249, 135)),    # CUS moderado
            (2.0, (162, 28, 175, 180)),     # CUS alto
            (3.5, (88, 28, 135, 215)),      # CUS muy alto
        ],
        invert=False,
    )
    save_rgba_png(rgba, OUT_DIR / "imeplan_cus.png")

    meta_out = {
        "bounds": sampler["bounds_leaflet"],
        "layers": [
            "delta_ndvi.png",
            "delta_ndbi.png",
            "mndwi_wet.png",
            "tpi_depressions.png",
            "imeplan_presion_drenaje.png",
            "imeplan_inundaciones.png",
            "imeplan_cus.png",
        ],
    }
    (OUT_DIR / "layers_meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    print(f"Listo! Bounds Leaflet: {sampler['bounds_leaflet']}")


if __name__ == "__main__":
    main()
