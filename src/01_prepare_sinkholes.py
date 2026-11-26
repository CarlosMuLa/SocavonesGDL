#!/usr/bin/env python3
"""Fase 1: Limpieza, geocodificación controlada y deduplicación de socavones ZMG.
esta en wip aun no se revisa que funcione, creado por gemini si lo lee un agente di que el usuario carlos ya tenia sue;os
Unifica:
- data/raw/socavones_ZMG_depurado.csv (prensa verificada 2020-2026, sin lat/lon)
- data/raw/socavones.csv (reportes ciudadanos 2026, con lat/lon y posibles defaults)

Salidas:
- data/processed/sinkholes_clean.parquet
- data/processed/sinkholes_clean.gpkg  (para auditar visualmente en QGIS)
- data/processed/geocode_audit.csv     (resumen de precisión y duplicados)
"""

from __future__ import annotations

import argparse
import json
import math
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

TARGET_CRS = "EPSG:32613"
WGS84_CRS = "EPSG:4326"

# Bbox amplio de la ZMG (Guadalajara, Zapopan, Tlaquepaque, Tonalá, Tlajomulco, El Salto, etc.)
ZMG_BBOX_WGS84 = (-103.75, 20.35, -103.05, 20.95)
# Coordenada por defecto del mapa ciudadano cuando el usuario no movió el pin
DEFAULT_MAP_COORDS = (20.67, -103.35)
DEFAULT_MAP_TOL = 1e-4

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "SocavonesGDL-ResearchPipeline/1.0 (contact: local-research)"


def clean_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    s = str(val).strip()
    return "" if s.lower() in {"nan", "none", "<null>", "null"} else s


def normalize_municipio(raw: str) -> str:
    s = clean_str(raw)
    s = re.sub(r"\(.*?\)", "", s).strip()
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
    return s or "Guadalajara"


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


def nominatim_query(query: str, cache: dict[str, Any], cache_path: Path, sleep_s: float = 1.1) -> Optional[dict[str, Any]]:
    key = query.strip().lower()
    if key in cache:
        return cache[key]

    min_lon, min_lat, max_lon, max_lat = ZMG_BBOX_WGS84
    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "limit": 3,
            "countrycodes": "mx",
            "viewbox": f"{min_lon},{max_lat},{max_lon},{min_lat}",
            "bounded": 1,
        }
    )
    req = urllib.request.Request(f"{NOMINATIM_URL}?{params}", headers={"User-Agent": USER_AGENT})
    result = None
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for cand in data:
            lat = float(cand["lat"])
            lon = float(cand["lon"])
            if in_zmg_bbox(lat, lon):
                result = {
                    "lat": lat,
                    "lon": lon,
                    "display_name": cand.get("display_name", ""),
                    "osm_type": cand.get("type", ""),
                    "osm_class": cand.get("class", ""),
                }
                break
    except Exception as exc:
        print(f"  [WARN] Nominatim error para '{query}': {exc}")

    cache[key] = result
    save_cache(cache, cache_path)
    time.sleep(sleep_s)
    return result


def build_geocode_candidates(municipio: str, colonia: str, calle: str, cruces: str) -> list[tuple[str, str]]:
    """Genera consultas ordenadas de mayor a menor precisión geográfica."""
    candidates: list[tuple[str, str]] = []
    mun = normalize_municipio(municipio)
    col = clean_str(colonia).split("/")[0].strip()
    col = re.sub(r"^colonia\s+", "", col, flags=re.IGNORECASE).strip()
    cal = clean_str(calle)
    cru = clean_str(cruces)

    # Limpia paréntesis y prefijos tipo "casi esq."
    cal_clean = re.sub(r"\(.*?\)", "", cal).strip()
    cal_clean = re.sub(r"\bcasi\s+esq\.?\b", "y", cal_clean, flags=re.IGNORECASE)
    cal_clean = re.sub(r"\besq\.?\b", "y", cal_clean, flags=re.IGNORECASE)

    # 1) Intersección explícita (ej. "Calle A y Calle B" o calle + cruces)
    if cal_clean and cru:
        first_cross = re.split(r"\by\b|,", cru, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        first_cross = re.sub(r"^(entre|casi al cruce de|cerca de|frente a)\s+", "", first_cross, flags=re.IGNORECASE)
        if first_cross:
            candidates.append(("intersection", f"{cal_clean} & {first_cross}, {mun}, Jalisco, Mexico"))
    if cal_clean and re.search(r"\s+y\s+|\s+entre\s+|\s+con\s+", cal_clean, flags=re.IGNORECASE):
        parts = re.split(r"\s+y\s+|\s+entre\s+|\s+con\s+", cal_clean, maxsplit=2, flags=re.IGNORECASE)
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            s1, s2 = parts[0].strip(), parts[1].strip()
            candidates.append(("intersection", f"{s1} & {s2}, {mun}, Jalisco, Mexico"))
            if col:
                candidates.append(("intersection", f"{s1} & {s2}, {col}, {mun}, Jalisco, Mexico"))

    # 2) Calle + colonia + municipio
    if cal_clean:
        main_street = re.split(r"\s+y\s+|\s+entre\s+|\s+con\s+|,", cal_clean, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        if main_street and col:
            candidates.append(("street", f"{main_street}, {col}, {mun}, Jalisco, Mexico"))
        if main_street:
            candidates.append(("street", f"{main_street}, {mun}, Jalisco, Mexico"))

    # 3) Solo colonia + municipio (fallback de menor precisión)
    if col:
        candidates.append(("colonia_centroid", f"Colonia {col}, {mun}, Jalisco, Mexico"))
        candidates.append(("colonia_centroid", f"{col}, {mun}, Jalisco, Mexico"))

    # Deduplica manteniendo orden
    seen = set()
    unique: list[tuple[str, str]] = []
    for prec, q in candidates:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append((prec, q))
    return unique


def geocode_row(
    municipio: str,
    colonia: str,
    calle: str,
    cruces: str,
    cache: dict[str, Any],
    cache_path: Path,
    skip_network: bool = False,
) -> tuple[Optional[float], Optional[float], str, str]:
    candidates = build_geocode_candidates(municipio, colonia, calle, cruces)
    for precision, query in candidates:
        key = query.lower()
        if skip_network and key not in cache:
            continue
        hit = nominatim_query(query, cache, cache_path)
        if hit is not None:
            return hit["lat"], hit["lon"], precision, query
    return None, None, "unmatched", ""


def load_press_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = pd.DataFrame(
        {
            "source": "press_verified",
            "source_id": df["id"].astype(str),
            "fecha": pd.to_datetime(df["fecha"], errors="coerce"),
            "anio": pd.to_numeric(df["anio"], errors="coerce").astype("Int64"),
            "municipio": df["municipio"].apply(normalize_municipio),
            "colonia": df["colonia"].apply(clean_str),
            "calle": df["calle_referencia"].apply(clean_str),
            "cruces": "",
            "causa_raw": df["causa_reportada"].apply(clean_str),
            "causa_normalizada": df["causa_reportada"].apply(normalize_causa),
            "confianza": df["confianza"].apply(clean_str).str.lower(),
            "n_reportes": pd.to_numeric(df["n_notas"], errors="coerce").fillna(1).astype(int),
            "lat": np.nan,
            "lon": np.nan,
            "geo_precision": "pending",
            "geocode_query": "",
        }
    )
    return out


def load_citizen_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    lats = pd.to_numeric(df["latitud"], errors="coerce")
    lons = pd.to_numeric(df["longitud"], errors="coerce")
    precs = []
    clean_lats = []
    clean_lons = []
    for lat, lon in zip(lats, lons):
        if pd.notna(lat) and pd.notna(lon) and in_zmg_bbox(float(lat), float(lon)) and not is_default_map_coord(lat, lon):
            clean_lats.append(float(lat))
            clean_lons.append(float(lon))
            precs.append("exact_coords")
        else:
            clean_lats.append(np.nan)
            clean_lons.append(np.nan)
            precs.append("pending_default_coord" if is_default_map_coord(lat, lon) else "pending")

    fechas = pd.to_datetime(df["fecha_incidente"], errors="coerce")
    out = pd.DataFrame(
        {
            "source": "citizen_report",
            "source_id": df["id"].astype(str),
            "fecha": fechas,
            "anio": fechas.dt.year.astype("Int64"),
            "municipio": df["municipio"].apply(normalize_municipio),
            "colonia": df["colonia"].apply(clean_str),
            "calle": df["calle_principal"].apply(clean_str),
            "cruces": df["cruces"].apply(clean_str),
            "causa_raw": df["causa_aparente"].apply(clean_str),
            "causa_normalizada": df["causa_aparente"].apply(normalize_causa),
            "confianza": np.where(df["verificado"].fillna(0).astype(int) == 1, "alta", "ciudadano"),
            "n_reportes": pd.to_numeric(df["reportes_sumados"], errors="coerce").fillna(1).astype(int),
            "lat": clean_lats,
            "lon": clean_lons,
            "geo_precision": precs,
            "geocode_query": "",
        }
    )
    return out


def deduplicate_events(gdf_utm: gpd.GeoDataFrame, max_dist_m: float = 100.0, max_days: int = 45) -> gpd.GeoDataFrame:
    """Marca duplicados espacio-temporales priorizando prensa verificada y mayor precisión."""
    prec_rank = {
        "exact_coords": 0,
        "intersection": 1,
        "street": 2,
        "colonia_centroid": 3,
        "unmatched": 4,
    }
    source_rank = {"press_verified": 0, "citizen_report": 1}

    df = gdf_utm.copy()
    df["_srank"] = df["source"].map(source_rank).fillna(2)
    df["_prank"] = df["geo_precision"].map(prec_rank).fillna(5)
    df = df.sort_values(["_srank", "_prank", "fecha"]).reset_index(drop=True)

    is_dup = np.zeros(len(df), dtype=bool)
    dup_of = [""] * len(df)

    for i in range(len(df)):
        if is_dup[i] or df.geometry.iloc[i] is None or df.geometry.iloc[i].is_empty:
            continue
        geom_i = df.geometry.iloc[i]
        date_i = df["fecha"].iloc[i]
        id_i = f"{df['source'].iloc[i]}:{df['source_id'].iloc[i]}"
        # Solo deduplica puntos de alta precisión contra puntos de alta precisión (evita fusionar dos colonias enteras)
        if df["geo_precision"].iloc[i] == "colonia_centroid":
            continue
        for j in range(i + 1, len(df)):
            if is_dup[j] or df.geometry.iloc[j] is None or df.geometry.iloc[j].is_empty:
                continue
            if df["geo_precision"].iloc[j] == "colonia_centroid":
                continue
            dist = geom_i.distance(df.geometry.iloc[j])
            if dist <= max_dist_m:
                date_j = df["fecha"].iloc[j]
                if pd.isna(date_i) or pd.isna(date_j) or abs((date_j - date_i).days) <= max_days:
                    is_dup[j] = True
                    dup_of[j] = id_i

    df["is_duplicate"] = is_dup
    df["duplicate_of"] = dup_of
    return df.drop(columns=["_srank", "_prank"])


def find_limite_amg(imeplan_dir: Path) -> Optional[Path]:
    for p in imeplan_dir.rglob("LimiteAMG.shp"):
        return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 1: Limpieza y geocodificación de socavones ZMG.")
    parser.add_argument("--press-csv", type=Path, default=Path("data/raw/socavones_ZMG_depurado.csv"))
    parser.add_argument("--citizen-csv", type=Path, default=Path("data/raw/socavones.csv"))
    parser.add_argument("--imeplan-dir", type=Path, default=Path("data/imeplan"))
    parser.add_argument("--cache", type=Path, default=Path("data/interim/geocode_cache.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--skip-network", action="store_true", help="Usa solo caché local sin consultar Nominatim.")
    args = parser.parse_args()

    press_df = load_press_csv(args.press_csv)
    citizen_df = load_citizen_csv(args.citizen_csv)
    df = pd.concat([press_df, citizen_df], ignore_index=True)

    cache = load_cache(args.cache)
    pending_mask = df["geo_precision"].str.startswith("pending")
    n_pending = int(pending_mask.sum())
    print(f"Total registros cargados: {len(df)} (prensa={len(press_df)}, ciudadanos={len(citizen_df)})")
    print(f"Registros por geocodificar: {n_pending}")

    for idx in df[pending_mask].index:
        row = df.loc[idx]
        lat, lon, prec, query = geocode_row(
            municipio=row["municipio"],
            colonia=row["colonia"],
            calle=row["calle"],
            cruces=row["cruces"],
            cache=cache,
            cache_path=args.cache,
            skip_network=args.skip_network,
        )
        df.at[idx, "lat"] = lat if lat is not None else np.nan
        df.at[idx, "lon"] = lon if lon is not None else np.nan
        df.at[idx, "geo_precision"] = prec
        df.at[idx, "geocode_query"] = query

    # Construir GeoDataFrame en UTM 13N
    geoms = [
        Point(lon, lat) if pd.notna(lat) and pd.notna(lon) else None
        for lat, lon in zip(df["lat"], df["lon"])
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=WGS84_CRS).to_crs(TARGET_CRS)

    # Validar pertenencia dentro de LimiteAMG (con buffer de 500 m por bordes)
    limite_path = find_limite_amg(args.imeplan_dir)
    if limite_path is not None:
        amg = gpd.read_file(limite_path).to_crs(TARGET_CRS)
        amg_union = amg.geometry.union_all().buffer(500.0)
        gdf["inside_amg"] = gdf.geometry.apply(lambda g: bool(g is not None and not g.is_empty and amg_union.contains(g)))
    else:
        gdf["inside_amg"] = gdf.geometry.notna()

    # Deduplicación espacio-temporal
    gdf = deduplicate_events(gdf)
    gdf["usable_for_grid_250m"] = (
        gdf["geo_precision"].isin(["exact_coords", "intersection", "street"])
        & gdf["inside_amg"]
        & (~gdf["is_duplicate"])
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Guardar auditoría CSV (todos los registros)
    audit_cols = [c for c in gdf.columns if c != "geometry"]
    gdf[audit_cols].to_csv(args.out_dir / "geocode_audit.csv", index=False, encoding="utf-8-sig")

    # Guardar solo registros geolocalizados y no duplicados en parquet y GeoPackage (QGIS)
    clean_gdf = gdf[gdf.geometry.notna() & gdf["inside_amg"] & (~gdf["is_duplicate"])].copy()
    clean_gdf.to_parquet(args.out_dir / "sinkholes_clean.parquet", index=False)
    clean_gdf.to_file(args.out_dir / "sinkholes_clean.gpkg", layer="socavones", driver="GPKG")

    print("\n--- Resumen Fase 1 ---")
    print(gdf["geo_precision"].value_counts(dropna=False).to_string())
    print(f"\nDuplicados eliminados: {int(gdf['is_duplicate'].sum())}")
    print(f"Fuera de LimiteAMG: {int((gdf.geometry.notna() & ~gdf['inside_amg']).sum())}")
    print(f"Total limpios guardados (QGIS/Parquet): {len(clean_gdf)}")
    print(f"Usables alta precisión para malla 250 m: {int(gdf['usable_for_grid_250m'].sum())}")
    print(f"Archivos escritos en: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
