#!/usr/bin/env python3
"""Fase 1: Preparación, validación espacial y auditoría de socavones ZMG.

Fuente oficial por defecto:
- data/raw/socavones_prod.csv (coordenadas mapeadas manualmente al cm, 2020-2026)

Si alguna fila aún conserva la coordenada por defecto del mapa (20.67, -103.35),
la re-geocodifica usando el caché de Google Maps (o la marca en auditoría).

Salidas:
- data/processed/sinkholes_clean.parquet
- data/processed/sinkholes_clean.geojson (para revisar en QGIS)
- data/processed/geocode_audit.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from geo_io import read_vector, write_geojson

TARGET_CRS = "EPSG:32613"
WGS84_CRS = "EPSG:4326"
ZMG_BBOX_WGS84 = (-103.75, 20.35, -103.05, 20.95)
DEFAULT_MAP_COORDS = (20.67, -103.35)
DEFAULT_MAP_TOL = 1e-4
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def load_env_file(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def clean_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    s = str(val).strip()
    return "" if s.lower() in {"nan", "none", "<null>", "null"} else s


def normalize_municipio(raw: str) -> str:
    s = re.sub(r"\(.*?\)", "", clean_str(raw)).strip()
    low = s.lower()
    if "zapopan" in low:
        return "Zapopan"
    if "guadalajara" in low:
        return "Guadalajara"
    if "tlaquepaque" in low:
        return "San Pedro Tlaquepaque"
    if "tonal" in low:
        return "Tonalá"
    if "tlajomulco" in low:
        return "Tlajomulco de Zúñiga"
    if "salto" in low:
        return "El Salto"
    return s or "Otro"


def normalize_causa(raw: str) -> str:
    s = clean_str(raw).lower()
    if not s:
        return "desconocida"
    has_drenaje = any(k in s for k in ["drenaje", "colector", "tuber", "fuga", "siapa", "agua", "alcantarill", "pozo"])
    has_lluvia = any(k in s for k in ["lluv", "inund", "torment", "arroyo", "escurrim"])
    has_obra = any(k in s for k in ["obra", "muro", "pavimento", "losa", "tráiler", "trailer", "camion", "vandal"])
    if has_drenaje and has_lluvia:
        return "drenaje_y_lluvia"
    if has_drenaje:
        return "drenaje_fuga"
    if has_lluvia:
        return "lluvia_inundacion"
    if has_obra:
        return "obra_o_trafico"
    return "otra"


def normalize_tamano(raw: str) -> str:
    s = clean_str(raw).lower()
    if "grande" in s:
        return "grande"
    if "mediano" in s:
        return "mediano"
    if "peque" in s:
        return "pequeno"
    return "desconocido"


def is_default_map_coord(lat: Optional[float], lon: Optional[float]) -> bool:
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return False
    return (
        abs(float(lat) - DEFAULT_MAP_COORDS[0]) <= DEFAULT_MAP_TOL
        and abs(float(lon) - DEFAULT_MAP_COORDS[1]) <= DEFAULT_MAP_TOL
    )


def in_zmg_bbox(lat: float, lon: float) -> bool:
    min_lon, min_lat, max_lon, max_lat = ZMG_BBOX_WGS84
    return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)


def load_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def google_geocode_fallback(
    municipio: str,
    colonia: str,
    calle: str,
    cruces: str,
    api_key: str,
    cache: dict[str, Any],
    cache_path: Path,
    skip_network: bool = False,
) -> tuple[Optional[float], Optional[float], str, str]:
    mun = normalize_municipio(municipio)
    mun_q = "Guadalajara" if mun == "Otro" else mun
    parts = [p for p in [clean_str(calle), clean_str(cruces), clean_str(colonia), mun_q, "Jalisco, México"] if p]
    if not clean_str(calle) and not clean_str(colonia):
        return None, None, "unmatched", ""

    candidates = []
    if clean_str(calle) and clean_str(cruces):
        candidates.append(f"{clean_str(calle)} y {clean_str(cruces)}, {mun_q}, Jalisco, México")
    if clean_str(calle):
        candidates.append(f"{clean_str(calle)}, {mun_q}, Jalisco, México")
    candidates.append(", ".join(parts))

    min_lon, min_lat, max_lon, max_lat = ZMG_BBOX_WGS84
    for q in candidates:
        key = q.lower()
        hit = cache.get(key)
        if key not in cache and not skip_network and api_key:
            params = urllib.parse.urlencode(
                {
                    "address": q,
                    "bounds": f"{min_lat},{min_lon}|{max_lat},{max_lon}",
                    "components": "country:MX",
                    "region": "mx",
                    "language": "es",
                    "key": api_key,
                }
            )
            try:
                with urllib.request.urlopen(f"{GOOGLE_GEOCODE_URL}?{params}", timeout=15) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                hit = None
                if payload.get("status") == "OK":
                    for cand in payload.get("results", []):
                        loc = cand["geometry"]["location"]
                        lat, lon = float(loc["lat"]), float(loc["lng"])
                        if in_zmg_bbox(lat, lon):
                            hit = {
                                "lat": lat,
                                "lon": lon,
                                "formatted_address": cand.get("formatted_address", ""),
                                "types": cand.get("types", []),
                            }
                            break
                cache[key] = hit
                save_cache(cache, cache_path)
                time.sleep(0.05)
            except Exception as exc:
                print(f"  [WARN] Google Geocoding error para '{q}': {exc}")
        if hit is not None:
            return hit["lat"], hit["lon"], "geocoded_default_fix", hit.get("formatted_address", "")
    return None, None, "default_coord_unresolved", ""


def load_prod_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    fechas = pd.to_datetime(df["fecha_incidente"], errors="coerce")
    lats = pd.to_numeric(df["latitud"], errors="coerce")
    lons = pd.to_numeric(df["longitud"], errors="coerce")

    precs = []
    for lat, lon in zip(lats, lons):
        if pd.isna(lat) or pd.isna(lon):
            precs.append("missing_coord")
        elif is_default_map_coord(lat, lon):
            precs.append("default_map_coord")
        elif in_zmg_bbox(float(lat), float(lon)):
            precs.append("mapped_exact")
        else:
            precs.append("out_of_bbox")

    # Identifica si proviene de los 14 reportes ciudadanos iniciales o del lote histórico mapeado
    creado = pd.to_datetime(df["creado_en"], errors="coerce")
    source_tag = np.where(creado < pd.Timestamp("2026-11-27"), "citizen_report", "press_mapped")

    return pd.DataFrame(
        {
            "source": source_tag,
            "source_id": df["id"].astype(str),
            "fecha": fechas,
            "anio": fechas.dt.year.astype("Int64"),
            "mes": fechas.dt.month.astype("Int64"),
            "municipio": df["municipio"].apply(normalize_municipio),
            "colonia": df["colonia"].apply(clean_str),
            "calle": df["calle_principal"].apply(clean_str),
            "cruces": df["cruces"].apply(clean_str),
            "causa_raw": df["causa_aparente"].apply(clean_str),
            "causa_normalizada": df["causa_aparente"].apply(normalize_causa),
            "tamano": df["tamano_categoria"].apply(normalize_tamano),
            "n_reportes": pd.to_numeric(df["reportes_sumados"], errors="coerce").fillna(1).astype(int),
            "lat": lats,
            "lon": lons,
            "geo_precision": precs,
            "matched_address": "",
        }
    )


def flag_near_duplicates(gdf_utm: gpd.GeoDataFrame, max_dist_m: float = 50.0, max_days: int = 30) -> gpd.GeoDataFrame:
    """Detecta duplicados muy cercanos (< 50 m y < 30 días) sin borrar recurrencias en años distintos."""
    df = gdf_utm.sort_values(["fecha", "source_id"]).reset_index(drop=True)
    is_dup = np.zeros(len(df), dtype=bool)
    dup_of = [""] * len(df)

    for i in range(len(df)):
        if is_dup[i] or df.geometry.iloc[i] is None or df.geometry.iloc[i].is_empty:
            continue
        geom_i = df.geometry.iloc[i]
        date_i = df["fecha"].iloc[i]
        id_i = f"{df['source'].iloc[i]}:{df['source_id'].iloc[i]}"
        for j in range(i + 1, len(df)):
            if is_dup[j] or df.geometry.iloc[j] is None or df.geometry.iloc[j].is_empty:
                continue
            dist = geom_i.distance(df.geometry.iloc[j])
            if dist <= max_dist_m:
                date_j = df["fecha"].iloc[j]
                if pd.notna(date_i) and pd.notna(date_j) and abs((date_j - date_i).days) <= max_days:
                    is_dup[j] = True
                    dup_of[j] = id_i

    df["is_duplicate"] = is_dup
    df["duplicate_of"] = dup_of
    return df


def find_limite_amg(imeplan_dir: Path) -> Optional[Path]:
    for p in imeplan_dir.rglob("LimiteAMG.shp"):
        return p
    return None


def main() -> None:
    load_env_file(Path(".env"))

    parser = argparse.ArgumentParser(description="Fase 1: Preparación y validación de socavones_prod.csv.")
    parser.add_argument("--prod-csv", type=Path, default=Path("data/raw/socavones_prod.csv"))
    parser.add_argument("--imeplan-dir", type=Path, default=Path("data/imeplan"))
    parser.add_argument("--cache", type=Path, default=Path("data/interim/geocode_google_cache.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--skip-network", action="store_true", help="No llamar a Google Maps para defaults.")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    cache = load_cache(args.cache)

    df = load_prod_csv(args.prod_csv)
    default_mask = df["geo_precision"] == "default_map_coord"
    for idx in df[default_mask].index:
        row = df.loc[idx]
        lat, lon, prec, addr = google_geocode_fallback(
            municipio=row["municipio"],
            colonia=row["colonia"],
            calle=row["calle"],
            cruces=row["cruces"],
            api_key=api_key,
            cache=cache,
            cache_path=args.cache,
            skip_network=args.skip_network,
        )
        if lat is not None and lon is not None:
            df.at[idx, "lat"] = lat
            df.at[idx, "lon"] = lon
            df.at[idx, "geo_precision"] = prec
            df.at[idx, "matched_address"] = addr

    geoms = [
        Point(lon, lat) if pd.notna(lat) and pd.notna(lon) else None
        for lat, lon in zip(df["lat"], df["lon"])
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=WGS84_CRS).to_crs(TARGET_CRS)

    limite_path = find_limite_amg(args.imeplan_dir)
    if limite_path is not None:
        amg = read_vector(limite_path).to_crs(TARGET_CRS)
        amg_union = amg.geometry.union_all().buffer(500.0)
        gdf["inside_amg"] = gdf.geometry.apply(lambda g: bool(g is not None and not g.is_empty and amg_union.contains(g)))
    else:
        gdf["inside_amg"] = gdf.geometry.notna()

    gdf = flag_near_duplicates(gdf, max_dist_m=50.0, max_days=30)
    gdf["usable_for_grid_250m"] = (
        gdf["geo_precision"].isin(["mapped_exact", "geocoded_default_fix"])
        & gdf["inside_amg"]
        & (~gdf["is_duplicate"])
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_cols = [c for c in gdf.columns if c != "geometry"]
    gdf[audit_cols].to_csv(args.out_dir / "geocode_audit.csv", index=False, encoding="utf-8-sig")

    clean_gdf = gdf[gdf["usable_for_grid_250m"]].copy()
    clean_gdf.to_parquet(args.out_dir / "sinkholes_clean.parquet", index=False)
    write_geojson(clean_gdf, args.out_dir / "sinkholes_clean.geojson")

    print("--- Resumen Fase 1 (socavones_prod.csv) ---")
    print(f"Total registros cargados: {len(gdf)}")
    print("\nPrecisión geográfica:")
    print(gdf["geo_precision"].value_counts(dropna=False).to_string())
    print("\nDistribución por año:")
    print(clean_gdf["anio"].value_counts().sort_index().to_string())
    print("\nCausas normalizadas:")
    print(clean_gdf["causa_normalizada"].value_counts().to_string())
    print(f"\nDuplicados (<50m y <30d): {int(gdf['is_duplicate'].sum())}")
    print(f"Fuera de LimiteAMG: {int((gdf.geometry.notna() & ~gdf['inside_amg']).sum())}")
    print(f"Total limpios listos para malla 250 m: {len(clean_gdf)}")
    print(f"Archivos escritos en: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
