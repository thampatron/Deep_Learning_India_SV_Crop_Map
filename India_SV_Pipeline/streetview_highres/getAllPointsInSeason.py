import os
import re
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import json

def load_inferred_fallow(inferred_folder):
    lat_pattern  = re.compile(r'GSVLat(-?\d+\.\d+)')
    lon_pattern  = re.compile(r'GSVLon(-?\d+\.\d+)')
    date_pattern = re.compile(r'date(\d{4}-\d{2})')

    csv_files = glob.glob(os.path.join(inferred_folder, "*.csv"))
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    if "fallow_label" in df.columns:
        df = df[df["fallow_label"] == "growing"].copy()

    def parse_from_image_path(path):
        lat = lat_pattern.search(path)
        lon = lon_pattern.search(path)
        date = date_pattern.search(path)
        return (
            float(lat.group(1)) if lat else None,
            float(lon.group(1)) if lon else None,
            date.group(1) if date else "1900-01",
        )

    if "latitude" not in df.columns or "longitude" not in df.columns:
        df[["latitude", "longitude", "date_str"]] = df["image_path"].apply(
            lambda x: pd.Series(parse_from_image_path(x))
        )
    else:
        df["date_str"] = df["image_path"].apply(
            lambda x: parse_from_image_path(x)[2]
        )

    df["date_inferred"] = pd.to_datetime(df["date_str"], format="%Y-%m", errors="coerce")
    df["year_inferred"]  = df["date_inferred"].dt.year
    df["month_inferred"] = df["date_inferred"].dt.month

    def classify_season(row):
        m = row["month_inferred"]
        y = row["year_inferred"]
        if m in [6, 7, 8, 9, 10, 11]:
            return "Kharif", y
        if m == 12:
            return "Rabi", y + 1
        if m in [1, 2, 3, 4, 5]:
            return "Rabi", y
        return "Other", y

    df[["season_inferred", "season_year"]] = df.apply(
        lambda r: pd.Series(classify_season(r)), axis=1
    )

    df.dropna(subset=["latitude", "longitude"], inplace=True)
    return df

def extract_kharif_2023_and_save(inferred_folder, out_csv):
    df = load_inferred_fallow(inferred_folder)
    df_kharif_2023 = df[
    (df["year_inferred"] == 2023) & (df["season_inferred"].isin(["Kharif", "Rabi"]))
    ].copy()
    print(f"Filtered to Kharif 2023: {len(df_kharif_2023)} rows.")
    df_kharif_2023.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")

# ========================
# Example usage
# ========================
if __name__ == "__main__":
    inferred_folder = "/home/laguarta_jordi/sean7391/inferred_fallow"
    out_csv = "/home/laguarta_jordi/sean7391/streetview_highres/kharif_rabi_2023_allpoints.csv"
    extract_kharif_2023_and_save(inferred_folder, out_csv)
