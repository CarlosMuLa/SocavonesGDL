#!/usr/bin/env python3
"""Fase 2: Stack satelital multi-temporal (2020 vs 2026) y DEM para la ZMG.

Diseñado para ejecutarse en la EC2 (us-west-2, junto a los COGs de Element 84)
y subir los GeoTIFFs ligeros a S3 para luego descargarlos a tu PC local.

Genera 10 GeoTIFFs alineados pixel-a-pixel en EPSG:32613 (20 m/píxel):
  Cambio de uso de suelo (misma temporada jun-oct para evitar sesgo estacional):
    - ndvi_2020.tif, ndvi_2026.tif, delta_ndvi_2020_2026.tif
    - ndbi_2020.tif, ndbi_2026.tif, delta_ndbi_2020_2026.tif
  Humedad e hidro-topografía:
    - mndwi_wet_2026.tif
    - dem_elevation.tif, slope_deg.tif, tpi_local.tif
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import pystac_client
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

STAC_URL = "https://earth-search.aws.element84.com/v1"
TARGET_CRS = CRS.from_epsg(32613)

# Mancha urbana completa de la ZMG (cubre 100% de los 129 socavones + margen de ~3-5 km)
DEFAULT_BBOX_WGS84 = [-103.52, 20.48, -103.20, 20.82]
DEFAULT_RESOLUTION_M = 20.0

# Clases SCL limpias de Sentinel-2 L2A (excluye nodata, saturado, sombras, nubes y cirrus)
VALID_SCL_CLASSES = np.array([2, 4, 5, 6, 7, 11], dtype="uint8")

# Fecha de entrada de Sentinel-2 Processing Baseline 04.00 (offset radiométrico -1000)
BASELINE_04_DATE = datetime(2022, 1, 25, tzinfo=timezone.utc)


def build_target_grid(bbox: list[float], dst_crs: CRS, resolution: float):
    """Construye la rejilla canónica UTM 13N ajustada a múltiplos exactos de resolución."""
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", dst_crs, *bbox)
    minx = math.floor(minx / resolution) * resolution
    miny = math.floor(miny / resolution) * resolution
    maxx = math.ceil(maxx / resolution) * resolution
    maxy = math.ceil(maxy / resolution) * resolution

    width = int(round((maxx - minx) / resolution))
    height = int(round((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)
    return height, width, transform


def get_scale_offset(item, asset) -> tuple[float, float]:
    """Obtiene scale y offset radiométrico manejando Sentinel-2 2020 (0.0) vs 2022+ (-0.1)."""
    raster_bands = asset.extra_fields.get("raster:bands")
    if raster_bands and isinstance(raster_bands, list):
        info = raster_bands[0]
        scale = float(info.get("scale", 0.0001))
        if "offset" in info:
            return scale, float(info["offset"])
    # Fallback por fecha de Baseline 04.00 si el item STAC omite offset explícito
    dt = item.datetime
    if dt is not None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        offset = -0.1 if dt >= BASELINE_04_DATE else 0.0
    else:
        offset = 0.0
    return 0.0001, offset


def read_band_to_grid(
    href: str,
    height: int,
    width: int,
    dst_transform,
    dst_crs: CRS,
    resampling: Resampling,
    dst_nodata: float = np.nan,
) -> np.ndarray:
    """Transmite y reproyecta un COG remoto directo a la rejilla destino."""
    dst = np.full((height, width), dst_nodata, dtype="float32")
    with rasterio.open(href) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            dst_nodata=dst_nodata,
        )
    return dst


def select_sentinel_items(
    catalog: pystac_client.Client,
    bbox: list[float],
    date_range: str,
    max_cloud: float,
    max_per_tile: int,
) -> tuple[list, dict[str, list]]:
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    all_items = list(search.items())
    if not all_items:
        raise RuntimeError(f"Sin escenas Sentinel-2 para {date_range} (nubes < {max_cloud}%).")

    by_tile = defaultdict(list)
    for item in all_items:
        tile = item.properties.get("grid:code") or item.id.split("_")[1]
        by_tile[tile].append(item)

    selected = []
    for tile, items in sorted(by_tile.items()):
        items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
        chosen = items[:max_per_tile]
        selected.extend(chosen)
        clouds = [round(c.properties["eo:cloud_cover"], 1) for c in chosen]
        print(f"  [{date_range}] Tile {tile}: {len(chosen)}/{len(items)} escenas (nubes={clouds}%)")
    return selected, dict(by_tile)


def build_sentinel_composite_low_ram(
    items: list,
    band_names: list[str],
    height: int,
    width: int,
    dst_transform,
    dst_crs: CRS,
) -> dict[str, np.ndarray]:
    """Calcula medianas por banda una a la vez para mantener pico de RAM < 300 MB en t3.small."""
    # Paso 1: pre-calcular máscaras SCL (uint8, ~3 MB por escena)
    clear_masks: list[tuple[object, np.ndarray]] = []
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for item in items:
            scl = read_band_to_grid(
                item.assets["scl"].href,
                height,
                width,
                dst_transform,
                dst_crs,
                Resampling.nearest,
                dst_nodata=0.0,
            )
            mask = np.isin(scl.astype("uint8"), VALID_SCL_CLASSES)
            if np.any(mask):
                clear_masks.append((item, mask))

    if not clear_masks:
        raise RuntimeError("Ninguna escena tuvo píxeles despejados tras aplicar máscara SCL.")

    # Paso 2: una banda a la vez para minimizar uso de memoria
    composites: dict[str, np.ndarray] = {}
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for bname in band_names:
            stack = np.full((len(clear_masks), height, width), np.nan, dtype="float32")
            resamp = Resampling.average if bname in {"red", "green", "nir"} else Resampling.bilinear
            for i, (item, mask) in enumerate(clear_masks):
                asset = item.assets[bname]
                scale, offset = get_scale_offset(item, asset)
                raw = read_band_to_grid(
                    asset.href,
                    height,
                    width,
                    dst_transform,
                    dst_crs,
                    resamp,
                    dst_nodata=0.0,
                )
                valid = mask & np.isfinite(raw) & (raw > 0)
                if np.any(valid):
                    refl = np.clip(raw[valid] * scale + offset, 0.0, 1.0)
                    stack[i][valid] = refl
            composites[bname] = np.nanmedian(stack, axis=0).astype("float32")
            del stack
            gc.collect()
            print(f"    Banda '{bname}' lista (cobertura válida: {100.0 * np.isfinite(composites[bname]).mean():.1f}%)")

    return composites


def build_dem_derivatives(
    catalog: pystac_client.Client,
    bbox: list[float],
    height: int,
    width: int,
    dst_transform,
    dst_crs: CRS,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construye DEM Copernicus GLO-30, pendiente (grados) y TPI local (depresiones)."""
    search = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)
    dem_items = list(search.items())
    if not dem_items:
        raise RuntimeError("No se encontraron tiles de cop-dem-glo-30 para el bbox.")
    print(f"  [DEM] Tiles Copernicus GLO-30 encontrados: {len(dem_items)}")

    stack = np.full((len(dem_items), height, width), np.nan, dtype="float32")
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for i, item in enumerate(dem_items):
            asset_key = "data" if "data" in item.assets else next(iter(item.assets))
            stack[i] = read_band_to_grid(
                item.assets[asset_key].href,
                height,
                width,
                dst_transform,
                dst_crs,
                Resampling.bilinear,
                dst_nodata=np.nan,
            )
    dem = np.nanmean(stack, axis=0).astype("float32")
    del stack
    gc.collect()

    # Pendiente en grados
    gy, gx = np.gradient(dem, resolution_m)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")

    # TPI local (diferencia contra media móvil de ventana ~500 m -> negativo = depresión/bajo)
    win = max(3, int(round(500.0 / resolution_m)) | 1)
    pad = win // 2
    dem_filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    padded = np.pad(dem_filled, pad, mode="edge")
    # Integral image (summed-area table) para media móvil O(N) sin scipy
    csum = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    local_mean = (
        csum[win:, win:]
        - csum[:-win, win:]
        - csum[win:, :-win]
        + csum[:-win, :-win]
    ) / float(win * win)
    tpi_local = np.where(np.isfinite(dem), dem - local_mean, np.nan).astype("float32")

    print(
        f"    DEM listo: elevación {np.nanmin(dem):.0f}–{np.nanmax(dem):.0f} m, "
        f"pendiente media {np.nanmean(slope_deg):.2f}°"
    )
    return dem, slope_deg, tpi_local


def safe_index(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where((den == 0) | (~np.isfinite(den)) | (~np.isfinite(num)), np.nan, num / den)
    return np.clip(out, -1.0, 1.0).astype("float32")


def write_geotiff(
    arr: np.ndarray,
    path: Path,
    transform,
    crs: CRS,
    description: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "height": arr.shape[0],
        "width": arr.shape[1],
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype("float32"), 1)
        dst.set_band_description(1, description)
    return path


def upload_to_s3(local_path: Path, bucket: str, s3_key: str, region: str) -> str:
    s3 = boto3.client("s3", region_name=region)
    content_type = "image/tiff" if local_path.suffix == ".tif" else (
        "image/png" if local_path.suffix == ".png" else "application/json"
    )
    s3.upload_file(str(local_path), bucket, s3_key, ExtraArgs={"ContentType": content_type})
    return f"s3://{bucket}/{s3_key}"


def save_preview_panel(
    layers: dict[str, np.ndarray],
    out_png: Path,
    bucket: Optional[str],
    s3_prefix: str,
    region: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [
        ("delta_ndvi_2020_2026", "ΔNDVI (2026 - 2020)\nRojo = pérdida vegetación", "RdYlGn", -0.35, 0.35),
        ("delta_ndbi_2020_2026", "ΔNDBI (2026 - 2020)\nRojo = más pavimento/obra", "RdBu_r", -0.25, 0.25),
        ("mndwi_wet_2026", "MNDWI Lluvias 2026\nAzul = agua/humedad", "RdYlBu", -0.5, 0.3),
        ("tpi_local", "TPI Topografía (500 m)\nAzul = depresiones/bajos", "BrBG_r", -15.0, 15.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=130)
    for ax, (key, title, cmap, vmin, vmax) in zip(axes.flat, specs):
        im = ax.imshow(layers[key], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Stack Satelital y Topográfico ZMG (EPSG:32613, 20 m)", fontsize=13)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    if bucket:
        key = f"{s3_prefix.rstrip('/')}/stack_preview_zmg.png"
        uri = upload_to_s3(out_png, bucket, key, region)
        s3 = boto3.client("s3", region_name=region)
        url = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600)
        print(f"Preview guardado en: {uri}")
        print(f"URL temporal (1 hora) para ver los 4 paneles:\n{url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fase 2: Stack satelital y DEM ZMG (EC2 -> S3).")
    parser.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX_WGS84, metavar=("W", "S", "E", "N"))
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION_M)
    parser.add_argument("--base-dates", default="2020-06-01/2020-10-31")
    parser.add_argument("--recent-dates", default="2026-06-01/2026-10-31")
    parser.add_argument("--max-cloud", type=float, default=20.0)
    parser.add_argument("--max-per-tile", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/socavones_rasters"))
    parser.add_argument("--s3-bucket", default=os.environ.get("NDVI_S3_BUCKET", ""))
    parser.add_argument("--s3-prefix", default="rasters")
    parser.add_argument("--aws-region", default="us-west-2")
    parser.add_argument("--dry-run", action="store_true", help="Solo consulta catálogo STAC sin bajar píxeles.")
    parser.add_argument("--save-preview", action="store_true", help="Genera PNG de 4 paneles y lo sube a S3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    height, width, transform = build_target_grid(args.bbox, TARGET_CRS, args.resolution)
    print(f"Rejilla ZMG: {height} x {width} px ({args.resolution:.0f} m/px, EPSG:32613)")
    print(f"BBox WGS84: {args.bbox}")

    catalog = pystac_client.Client.open(STAC_URL)
    items_2020, tiles_2020 = select_sentinel_items(catalog, args.bbox, args.base_dates, args.max_cloud, args.max_per_tile)
    items_2026, tiles_2026 = select_sentinel_items(catalog, args.bbox, args.recent_dates, args.max_cloud, args.max_per_tile)

    if args.dry_run:
        dem_items = list(catalog.search(collections=["cop-dem-glo-30"], bbox=args.bbox).items())
        print(f"  [DEM] Tiles Copernicus GLO-30 encontrados: {len(dem_items)}")
        print("Dry-run completado sin descargar píxeles.")
        return

    print("\n1/3 Procesando Sentinel-2 2020 (red, nir, swir16)...")
    c20 = build_sentinel_composite_low_ram(items_2020, ["red", "nir", "swir16"], height, width, transform, TARGET_CRS)
    ndvi_2020 = safe_index(c20["nir"] - c20["red"], c20["nir"] + c20["red"])
    ndbi_2020 = safe_index(c20["swir16"] - c20["nir"], c20["swir16"] + c20["nir"])
    del c20
    gc.collect()

    print("\n2/3 Procesando Sentinel-2 2026 (red, green, nir, swir16)...")
    c26 = build_sentinel_composite_low_ram(
        items_2026, ["red", "green", "nir", "swir16"], height, width, transform, TARGET_CRS
    )
    ndvi_2026 = safe_index(c26["nir"] - c26["red"], c26["nir"] + c26["red"])
    ndbi_2026 = safe_index(c26["swir16"] - c26["nir"], c26["swir16"] + c26["nir"])
    mndwi_2026 = safe_index(c26["green"] - c26["swir16"], c26["green"] + c26["swir16"])
    del c26
    gc.collect()

    delta_ndvi = (ndvi_2026 - ndvi_2020).astype("float32")
    delta_ndbi = (ndbi_2026 - ndbi_2020).astype("float32")

    print("\n3/3 Procesando Copernicus GLO-30 DEM (elevación, pendiente, TPI local)...")
    dem, slope_deg, tpi_local = build_dem_derivatives(
        catalog, args.bbox, height, width, transform, TARGET_CRS, args.resolution
    )

    outputs = {
        "ndvi_2020": ndvi_2020,
        "ndvi_2026": ndvi_2026,
        "delta_ndvi_2020_2026": delta_ndvi,
        "ndbi_2020": ndbi_2020,
        "ndbi_2026": ndbi_2026,
        "delta_ndbi_2020_2026": delta_ndbi,
        "mndwi_wet_2026": mndwi_2026,
        "dem_elevation": dem,
        "slope_deg": slope_deg,
        "tpi_local": tpi_local,
    }

    written_files = []
    for name, arr in outputs.items():
        local_tif = write_geotiff(arr, args.out_dir / f"{name}.tif", transform, TARGET_CRS, name)
        written_files.append(local_tif)
        msg = f"  Escrito {local_tif.name} ({local_tif.stat().st_size / 1e6:.1f} MB) | media={np.nanmean(arr):.3f}"
        if args.s3_bucket:
            uri = upload_to_s3(local_tif, args.s3_bucket, f"{args.s3_prefix.rstrip('/')}/{local_tif.name}", args.aws_region)
            msg += f" -> {uri}"
        print(msg)

    meta = {
        "crs": str(TARGET_CRS),
        "resolution_m": args.resolution,
        "shape": [height, width],
        "transform": list(transform)[:6],
        "bbox_wgs84": args.bbox,
        "base_dates": args.base_dates,
        "recent_dates": args.recent_dates,
        "tiles_2020": sorted(tiles_2020.keys()),
        "tiles_2026": sorted(tiles_2026.keys()),
        "layers": list(outputs.keys()),
    }
    meta_path = args.out_dir / "rasters_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.s3_bucket:
        upload_to_s3(meta_path, args.s3_bucket, f"{args.s3_prefix.rstrip('/')}/rasters_metadata.json", args.aws_region)

    if args.save_preview:
        save_preview_panel(outputs, args.out_dir / "stack_preview_zmg.png", args.s3_bucket, "previews", args.aws_region)

    print("\nFase 2 completada correctamente.")


if __name__ == "__main__":
    main()
