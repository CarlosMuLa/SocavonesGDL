"""Portable geospatial I/O helpers with pure-Python fallback (pyshp / GeoJSON).

Avoids failures on Windows 11 machines where Smart App Control blocks unsigned
GDAL/pyogrio native DLLs, while using pyogrio natively on Linux/EC2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import shapefile
from pyproj import CRS
from shapely.geometry import mapping, shape


def read_vector(path: Path, encoding: Optional[str] = None) -> gpd.GeoDataFrame:
    """Read a Shapefile or GeoJSON into a GeoDataFrame, falling back to pyshp."""
    path = Path(path)
    try:
        kwargs = {"encoding": encoding} if encoding else {}
        return gpd.read_file(path, **kwargs)
    except Exception:
        if path.suffix.lower() != ".shp":
            raise

    # Pure-Python fallback via pyshp
    enc = encoding
    if enc is None:
        for sidecar in (path.with_suffix(".cpg"), path.with_suffix(".cst")):
            if sidecar.exists():
                enc = sidecar.read_text(encoding="ascii", errors="ignore").strip() or "utf-8"
                break
        if not enc:
            enc = "latin1"

    crs = None
    prj = path.with_suffix(".prj")
    if prj.exists():
        crs = CRS.from_wkt(prj.read_text(encoding="utf-8", errors="ignore"))

    reader = shapefile.Reader(str(path), encoding=enc, encodingErrors="replace")
    field_names = [f[0] for f in reader.fields[1:]]
    rows = []
    geoms = []
    for sr in reader.iterShapeRecords():
        rows.append(dict(zip(field_names, sr.record)))
        geoms.append(shape(sr.shape.__geo_interface__) if sr.shape.shapeType != 0 else None)

    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)


def write_geojson(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    """Write a GeoDataFrame to RFC 7946 GeoJSON (EPSG:4326) using pure Python."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf_wgs = gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf

    features = []
    prop_cols = [c for c in gdf_wgs.columns if c != gdf_wgs.geometry.name]
    for _, row in gdf_wgs.iterrows():
        geom = row.geometry
        props = {}
        for col in prop_cols:
            val = row[col]
            if pd.isna(val):
                props[col] = None
            elif isinstance(val, pd.Timestamp):
                props[col] = val.strftime("%Y-%m-%d")
            elif hasattr(val, "item"):
                props[col] = val.item()
            else:
                props[col] = val
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom) if geom is not None and not geom.is_empty else None,
                "properties": props,
            }
        )

    fc = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(fc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
