#!/usr/bin/env python3
# sample_tiles_from_inferred_notfield_fixed.py

import argparse, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Regex to parse from image_path
DATE_RE  = re.compile(r"&date(\d{4}-\d{2})")
LAT_RE   = re.compile(r"&GSVLat(-?\d+(?:\.\d+)?)")
LON_RE   = re.compile(r"&GSVLon(-?\d+(?:\.\d+)?)")
HEAD_RE  = re.compile(r"&head(-?\d+(?:\.\d+)?)")

def parse_image_path(s: str):
    if not isinstance(s, str):
        return None, np.nan, np.nan, np.nan
    dm = DATE_RE.search(s)
    lm = LAT_RE.search(s)
    gm = LON_RE.search(s)
    hm = HEAD_RE.search(s)
    date_ym = dm.group(1) if dm else None
    lat = float(lm.group(1)) if lm else np.nan
    lon = float(gm.group(1)) if gm else np.nan
    head = float(hm.group(1)) if hm else np.nan
    return date_ym, lat, lon, head

EARTH_RADIUS_M = 6371000.0
def destination_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_m: float):
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    dR = distance_m / EARTH_RADIUS_M
    sin_lat1, cos_lat1 = math.sin(lat1), math.cos(lat1)
    sin_dR, cos_dR = math.sin(dR), math.cos(dR)
    sin_lat2 = sin_lat1 * cos_dR + cos_lat1 * sin_dR * math.cos(brng)
    lat2 = math.asin(sin_lat2)
    y = math.sin(brng) * sin_dR * cos_lat1
    x = cos_dR - sin_lat1 * sin_lat2
    lon2 = lon1 + math.atan2(y, x)
    lon2 = (lon2 + math.pi) % (2 * math.pi) - math.pi
    return math.degrees(lat2), math.degrees(lon2)

def run(input_csv, output_csv, n_per_tile, tile_deg, min_lat, max_lat, min_lon, max_lon, seed=0):
    tqdm.write(f"Reading: {input_csv}")
    df_raw = pd.read_csv(input_csv, dtype=str, low_memory=False, on_bad_lines="skip")

    # Only keep the two columns we actually need from the merged CSV
    keep = [c for c in ["image_path", "label"] if c in df_raw.columns]
    if set(keep) != {"image_path", "label"}:
        raise ValueError("Input must contain columns: image_path,label")
    df = df_raw[keep].copy()

    # Drop empty image_path early to avoid ghost rows
    df = df[df["image_path"].notna() & (df["image_path"].str.strip() != "")]
    tqdm.write(f"Rows after dropping empty image_path: {len(df):,}")

    # Filter label == not-field (case/whitespace tolerant)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"] == "not-field"].copy()
    tqdm.write(f"Rows after label filter: {len(df):,}")

    # Parse from image_path into distinct, non-colliding columns
    tqdm.write("Parsing image_path → gsv_date_ym, gsv_lat, gsv_lon, gsv_head …")
    tqdm.pandas(desc="parse image_path")
    parsed = df["image_path"].progress_apply(parse_image_path)
    dfp = pd.DataFrame(parsed.tolist(), columns=["gsv_date_ym", "gsv_lat", "gsv_lon", "gsv_head"])
    df = pd.concat([df.reset_index(drop=True), dfp], axis=1)

    # Coerce numeric & drop rows missing parsed coords or date
    for c in ["gsv_lat", "gsv_lon", "gsv_head"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["gsv_lat", "gsv_lon", "gsv_date_ym"]).copy()
    tqdm.write(f"Rows with valid parsed coords+date: {len(df):,}")

    # Date window: 2023-05..2023-12
    df["gsv_date"] = pd.to_datetime(df["gsv_date_ym"] + "-01", errors="coerce")
    mask = (df["gsv_date"] >= pd.Timestamp("2023-05-01")) & (df["gsv_date"] <= pd.Timestamp("2023-12-31"))
    df = df[mask].copy()
    tqdm.write(f"Rows after date filter (2023-05..2023-12): {len(df):,}")

    # BBox crop using parsed gsv coords
    within = (
        (df["gsv_lat"] >= min_lat) & (df["gsv_lat"] <= max_lat) &
        (df["gsv_lon"] >= min_lon) & (df["gsv_lon"] <= max_lon)
    )
    df = df[within].copy()
    tqdm.write(f"Rows within bbox: {len(df):,}")

    # Tile indices from parsed coords
    df["tile_x"] = ((df["gsv_lon"] - min_lon) / tile_deg).apply(np.floor).astype(int)
    df["tile_y"] = ((df["gsv_lat"] - min_lat) / tile_deg).apply(np.floor).astype(int)

    max_tx = int(math.floor((max_lon - min_lon) / tile_deg)) - 1
    max_ty = int(math.floor((max_lat - min_lat) / tile_deg)) - 1
    df = df[(df["tile_x"] >= 0) & (df["tile_y"] >= 0) &
            (df["tile_x"] <= max_tx) & (df["tile_y"] <= max_ty)].copy()

    # Random sample N per tile
    rng = np.random.default_rng(seed)
    tqdm.write(f"Sampling up to {n_per_tile} per mini-tile …")
    sampled = []
    tile_groups = df.groupby(["tile_x", "tile_y"], sort=False)
    for _, g in tqdm(tile_groups, total=tile_groups.ngroups, desc="tiles"):
        if len(g) <= n_per_tile:
            sampled.append(g)
        else:
            idx = rng.choice(g.index.values, size=n_per_tile, replace=False)
            sampled.append(g.loc[idx])
    if not sampled:
        tqdm.write("No tiles had points; nothing to write.")
        sys.exit(0)
    out = pd.concat(sampled, ignore_index=True)

    # Compute 20 m forward from parsed coords & head
    tqdm.write("Computing lat_20m/lon_20m along gsv_head …")
    def _dest(row):
        if pd.isna(row["gsv_head"]):
            return np.nan, np.nan
        return destination_point(float(row["gsv_lat"]), float(row["gsv_lon"]), float(row["gsv_head"]), 20.0)
    tqdm.pandas(desc="forward 20m")
    dests = out.progress_apply(_dest, axis=1, result_type="expand")
    dests.columns = ["lat_20m", "lon_20m"]
    out = pd.concat([out, dests], axis=1)

    # Final columns (explicitly use parsed gsv_* coords)
    final_cols = [
        "image_path", "label", "gsv_date_ym",
        "gsv_lat", "gsv_lon", "gsv_head",
        "lat_20m", "lon_20m",
        "tile_x", "tile_y"
    ]
    out = out[final_cols].copy()

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    tqdm.write(f"Wrote {len(out):,} rows → {output_csv}")

def main():
    ap = argparse.ArgumentParser(description="Filter not-field by date, tile, sample N/tile, compute 20m forward (using parsed GSV coords).")
    ap.add_argument("--input", default="/home/laguarta_jordi/sean7391/inferred_final_merged.csv")
    ap.add_argument("--output", default='home/laguarta_jordi/sean7391/sample_tiles_from_inferred_notfield_101.csv')
    ap.add_argument("--n-per-tile", type=int, default=100)
    ap.add_argument("--tile-deg", type=float, default=0.25)
    ap.add_argument("--min-lat", type=float, default=6.0)
    ap.add_argument("--max-lat", type=float, default=37.5)
    ap.add_argument("--min-lon", type=float, default=68.0)
    ap.add_argument("--max-lon", type=float, default=97.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run(
        input_csv=args.input, output_csv=args.output,
        n_per_tile=args.n_per_tile, tile_deg=args.tile_deg,
        min_lat=args.min_lat, max_lat=args.max_lat,
        min_lon=args.min_lon, max_lon=args.max_lon,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
