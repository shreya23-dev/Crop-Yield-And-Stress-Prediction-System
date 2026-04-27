"""
merge_features.py — Merge all processed features into the final multimodal dataset.

This script combines:
  1. Cleaned yield data (maharashtra_kharif_yield_clean.csv)
  2. NDVI time series (ndvi_timeseries.csv) — monthly NDVI for Jun-Nov
  3. Weather time series (weather_timeseries.csv) — 22 weekly weather features
  4. Soil pH (soil_ph.csv) — static per-district feature
  5. Satellite image availability check (satellite_images/*.npy)

Output:
  data/processed/final_dataset.csv — One row per (district, year, crop) with all features merged.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- paths ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SATELLITE_DIR = PROCESSED_DIR / "satellite_images"

YIELD_CSV       = PROJECT_ROOT / "maharashtra_kharif_yield_clean.csv"
NDVI_CSV        = PROCESSED_DIR / "ndvi_timeseries.csv"
WEATHER_CSV     = PROCESSED_DIR / "weather_timeseries.csv"
SOIL_CSV        = PROCESSED_DIR / "soil_ph.csv"
OUTPUT_CSV      = PROCESSED_DIR / "final_dataset.csv"


def load_yield_data() -> pd.DataFrame:
    """Load cleaned yield data and keep relevant columns."""
    print("[1/5] Loading yield data ...")
    df = pd.read_csv(YIELD_CSV)
    
    # Keep only key columns needed for the model
    keep_cols = [
        "district", "latitude", "longitude", "year", "crop", "season",
        "area_hectare", "production", "yield_value", "yield_unit"
    ]
    df = df[keep_cols].copy()
    
    print(f"       Yield records : {len(df):,}")
    print(f"       Districts    : {df['district'].nunique()}")
    print(f"       Crops        : {df['crop'].unique().tolist()}")
    print(f"       Year range   : {df['year'].min()} - {df['year'].max()}")
    return df


def load_ndvi_data() -> pd.DataFrame:
    """Load monthly NDVI time series (Jun–Nov)."""
    print("[2/5] Loading NDVI time series ...")
    df = pd.read_csv(NDVI_CSV)
    
    # Ensure column names are clean
    df.columns = df.columns.str.strip()
    
    print(f"       NDVI records  : {len(df):,}")
    print(f"       Columns       : {list(df.columns)}")
    return df


def load_weather_data() -> pd.DataFrame:
    """Load weekly weather time series (22 weeks × 3 vars)."""
    print("[3/5] Loading weather time series ...")
    df = pd.read_csv(WEATHER_CSV)
    
    # Ensure column names are clean
    df.columns = df.columns.str.strip()
    
    # Drop lat/lon from weather (already in yield data)
    if "latitude" in df.columns:
        df = df.drop(columns=["latitude", "longitude"], errors="ignore")
    
    print(f"       Weather records : {len(df):,}")
    print(f"       Weather features: {len(df.columns) - 2} weekly vars")
    return df


def load_soil_data() -> pd.DataFrame:
    """Load static soil pH data (per district)."""
    print("[4/5] Loading soil pH data ...")
    df = pd.read_csv(SOIL_CSV)
    df.columns = df.columns.str.strip()
    
    print(f"       Soil records : {len(df):,}")
    print(f"       pH range     : {df['soil_ph'].min():.1f} - {df['soil_ph'].max():.1f}")
    return df


def check_satellite_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check satellite image availability for each (district, year) pair.
    Images are expected as {district}_{year}_{month:02d}.npy for months 06–11.
    Adds a boolean column 'has_satellite_images' and an integer 'satellite_image_count'.
    """
    print("[5/5] Checking satellite image availability ...")
    
    # Get unique (district, year) pairs from the merged data
    pairs = df[["district", "year"]].drop_duplicates()
    
    image_counts = []
    for _, row in pairs.iterrows():
        district = row["district"]
        year = row["year"]
        count = 0
        for month in range(6, 12):  # Jun(06) through Nov(11)
            fname = f"{district}_{year}_{month:02d}.npy"
            if (SATELLITE_DIR / fname).exists():
                count += 1
        image_counts.append({
            "district": district,
            "year": year,
            "satellite_image_count": count,
            "has_satellite_images": count == 6  # all 6 months available
        })
    
    img_df = pd.DataFrame(image_counts)
    
    full_count = img_df["has_satellite_images"].sum()
    partial_count = ((img_df["satellite_image_count"] > 0) & ~img_df["has_satellite_images"]).sum()
    none_count = (img_df["satellite_image_count"] == 0).sum()
    
    print(f"       Complete (6/6) : {full_count}")
    print(f"       Partial        : {partial_count}")
    print(f"       No images      : {none_count}")
    
    return img_df


def create_crop_and_district_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer IDs for crop and district (for embedding layers)."""
    # Crop ID mapping (sorted alphabetically)
    crops_sorted = sorted(df["crop"].unique())
    crop_to_id = {c: i for i, c in enumerate(crops_sorted)}
    df["crop_id"] = df["crop"].map(crop_to_id)
    
    # District ID mapping (sorted alphabetically)
    districts_sorted = sorted(df["district"].unique())
    district_to_id = {d: i for i, d in enumerate(districts_sorted)}
    df["district_id"] = df["district"].map(district_to_id)
    
    print(f"\n  Crop ID mapping     : {crop_to_id}")
    print(f"  District ID mapping : {len(district_to_id)} districts (0–{len(district_to_id)-1})")
    
    return df


def main():
    print("=" * 70)
    print("  MERGE FEATURES -> final_dataset.csv")
    print("=" * 70)
    
    # ---- 1. Load all data sources ----------------------------------------
    yield_df   = load_yield_data()
    ndvi_df    = load_ndvi_data()
    weather_df = load_weather_data()
    soil_df    = load_soil_data()
    
    # ---- 2. Merge yield ← NDVI (on district + year) ----------------------
    print("\n--- Merging yield + NDVI ---")
    merged = yield_df.merge(ndvi_df, on=["district", "year"], how="left")
    
    ndvi_cols = [c for c in ndvi_df.columns if c.startswith("ndvi_")]
    ndvi_missing = merged[ndvi_cols[0]].isna().sum()
    print(f"  After NDVI merge   : {len(merged):,} rows  ({ndvi_missing} rows without NDVI)")
    
    # ---- 3. Merge ← Weather (on district + year) -------------------------
    print("\n--- Merging + Weather ---")
    merged = merged.merge(weather_df, on=["district", "year"], how="left")
    
    weather_cols = [c for c in weather_df.columns if c.startswith("week_")]
    weather_missing = merged[weather_cols[0]].isna().sum() if weather_cols else 0
    print(f"  After weather merge: {len(merged):,} rows  ({weather_missing} rows without weather)")
    
    # ---- 4. Merge ← Soil pH (on district) --------------------------------
    print("\n--- Merging + Soil pH ---")
    merged = merged.merge(soil_df, on="district", how="left")
    
    soil_missing = merged["soil_ph"].isna().sum()
    print(f"  After soil merge   : {len(merged):,} rows  ({soil_missing} rows without soil pH)")

    # ---- 5. Check satellite image availability ----------------------------
    print("\n--- Checking satellite images ---")
    img_df = check_satellite_images(merged)
    merged = merged.merge(img_df, on=["district", "year"], how="left")
    
    # ---- 6. Add crop & district IDs for embeddings -----------------------
    print("\n--- Creating embedding IDs ---")
    merged = create_crop_and_district_ids(merged)
    
    # ---- 7. Add normalized year feature ----------------------------------
    merged["year_normalized"] = (merged["year"] - 1997) / 25.0
    
    # ---- 8. Compute seasonal NDVI summary stats (for baselines) ----------
    print("\n--- Computing NDVI summary statistics ---")
    ndvi_data_cols = [c for c in ndvi_cols if c in merged.columns]
    if ndvi_data_cols:
        ndvi_matrix = merged[ndvi_data_cols].to_numpy(dtype=float)
        valid_rows = ~np.isnan(ndvi_matrix).all(axis=1)

        ndvi_mean = np.full(len(merged), np.nan)
        ndvi_max = np.full(len(merged), np.nan)
        ndvi_min = np.full(len(merged), np.nan)
        ndvi_std = np.full(len(merged), np.nan)
        if valid_rows.any():
            ndvi_mean[valid_rows] = np.nanmean(ndvi_matrix[valid_rows], axis=1)
            ndvi_max[valid_rows] = np.nanmax(ndvi_matrix[valid_rows], axis=1)
            ndvi_min[valid_rows] = np.nanmin(ndvi_matrix[valid_rows], axis=1)
            ndvi_std[valid_rows] = np.nanstd(ndvi_matrix[valid_rows], axis=1)

        merged["ndvi_mean"] = ndvi_mean
        merged["ndvi_max"] = ndvi_max
        merged["ndvi_min"] = ndvi_min
        merged["ndvi_range"] = merged["ndvi_max"] - merged["ndvi_min"]
        merged["ndvi_std"] = ndvi_std
        
        # Month of peak NDVI (0-indexed June=0)
        peak_month = np.full(len(merged), np.nan)
        if valid_rows.any():
            peak_month[valid_rows] = np.nanargmax(ndvi_matrix[valid_rows], axis=1) + 6
        merged["ndvi_peak_month"] = peak_month  # actual month number (Jun=6 ... Nov=11)
        
        # Slope (linear trend over growing season)
        months = np.arange(len(ndvi_data_cols))
        slopes = []
        for i in range(len(merged)):
            row_vals = ndvi_matrix[i]
            valid = ~np.isnan(row_vals)
            if valid.sum() >= 2:
                slope = np.polyfit(months[valid], row_vals[valid], 1)[0]
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        merged["ndvi_slope"] = slopes
    
    # ---- 9. Compute weather summary stats (for baselines) ----------------
    print("--- Computing weather summary statistics ---")
    temp_mean_cols = [c for c in merged.columns if c.endswith("_temp_mean")]
    temp_max_cols  = [c for c in merged.columns if c.endswith("_temp_max")]
    rain_cols      = [c for c in merged.columns if c.endswith("_rain")]
    
    if temp_mean_cols:
        merged["weather_temp_mean_avg"]  = merged[temp_mean_cols].mean(axis=1)
        merged["weather_temp_max_avg"]   = merged[temp_max_cols].mean(axis=1)
        merged["weather_total_rainfall"] = merged[rain_cols].sum(axis=1)
        merged["weather_rain_variance"]  = merged[rain_cols].var(axis=1)
        
        # Dry weeks count (rainfall < 5mm)
        rain_matrix = merged[rain_cols].values
        merged["weather_dry_weeks"] = (rain_matrix < 5).sum(axis=1)
        
        # Max consecutive dry weeks
        max_consec = []
        for i in range(len(merged)):
            weekly_rain = rain_matrix[i]
            dry = weekly_rain < 5
            max_run = 0
            current_run = 0
            for d in dry:
                if d:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0
            max_consec.append(max_run)
        merged["weather_max_consec_dry_weeks"] = max_consec
    
    # ---- 10. Drop rows with too many missing critical features -----------
    print("\n--- Final cleanup ---")
    before = len(merged)
    
    # Only keep rows that have NDVI data (critical for the model)
    if ndvi_data_cols:
        merged = merged.dropna(subset=ndvi_data_cols, how="all")
    
    after = len(merged)
    print(f"  Dropped {before - after} rows with no NDVI data")
    
    # ---- 11. Sort and save -----------------------------------------------
    merged = merged.sort_values(["district", "year", "crop"]).reset_index(drop=True)
    
    # Save
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\n{'=' * 70}")
    print(f"  SAVED: {OUTPUT_CSV}")
    print(f"     Final rows     : {len(merged):,}")
    print(f"     Final columns  : {len(merged.columns):,}")
    print(f"     Districts      : {merged['district'].nunique()}")
    print(f"     Crops          : {merged['crop'].nunique()}")
    print(f"     Years          : {merged['year'].min()} - {merged['year'].max()}")
    print(f"     With sat. imgs : {merged['has_satellite_images'].sum():,}")
    print(f"{'=' * 70}")
    
    # ---- 12. Print column summary ----------------------------------------
    print("\n  COLUMN GROUPS:")
    groups = {
        "Identity"  : ["district", "year", "crop", "season", "crop_id", "district_id"],
        "Location"  : ["latitude", "longitude"],
        "Yield"     : ["area_hectare", "production", "yield_value", "yield_unit"],
        "NDVI (raw)": [c for c in merged.columns if c.startswith("ndvi_") and c not in [
            "ndvi_mean","ndvi_max","ndvi_min","ndvi_range","ndvi_std","ndvi_peak_month","ndvi_slope"]],
        "NDVI (summary)": ["ndvi_mean","ndvi_max","ndvi_min","ndvi_range","ndvi_std","ndvi_peak_month","ndvi_slope"],
        "Weather (raw)": [c for c in merged.columns if c.startswith("week_")],
        "Weather (summary)": [c for c in merged.columns if c.startswith("weather_")],
        "Soil"      : ["soil_ph"],
        "Satellite" : ["satellite_image_count", "has_satellite_images"],
        "Engineered": ["year_normalized"],
    }
    for gname, cols in groups.items():
        existing = [c for c in cols if c in merged.columns]
        print(f"    {gname:20s} : {len(existing):3d} columns")
    
    # ---- 13. Null summary ------------------------------------------------
    print("\n  NULL SUMMARY (top-10 by null count):")
    null_counts = merged.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False).head(10)
    if len(null_counts) == 0:
        print("    No nulls!")
    else:
        for col, cnt in null_counts.items():
            pct = cnt / len(merged) * 100
            print(f"    {col:40s} : {cnt:5d} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
