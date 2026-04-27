"""
Step 2: Satellite Image Patches + NDVI Time Series Extraction from Google Earth Engine

Extracts for each (district, year) pair (year >= 2000 only):
  1. 64x64 satellite image patches (4 bands: Red, NIR, NDVI, EVI) from MODIS MOD13Q1
     → saved as .npy files in data/processed/satellite_images/
  2. District-mean monthly NDVI (Jun-Nov)
     → saved as rows in data/processed/ndvi_timeseries.csv

Usage:
    python src/data/fetch_ndvi.py

Prerequisites:
    pip install earthengine-api numpy pandas
    earthengine authenticate
"""

import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import ee

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# MODIS MOD13Q1: 250m, 16-day NDVI/EVI + surface reflectance bands
MODIS_COLLECTION = "MODIS/061/MOD13Q1"

# Bands to extract (4 channels)
# sur_refl_b01 = Red (620-670nm), sur_refl_b02 = NIR (841-876nm)
# NDVI and EVI are vegetation indices
BANDS = ["sur_refl_b01", "sur_refl_b02", "NDVI", "EVI"]
BAND_NAMES = ["red", "nir", "ndvi_band", "evi"]
NUM_BANDS = len(BANDS)

# Scale factors for MODIS MOD13Q1
# Reflectance bands: scale = 0.0001, valid range 0-10000
# NDVI/EVI: scale = 0.0001, valid range -2000 to 10000
REFLECTANCE_SCALE = 0.0001
VI_SCALE = 0.0001

# Spatial parameters
BUFFER_RADIUS_M = 8000  # 8km radius → ~16km x 16km region
IMAGE_SIZE = 64         # 64x64 pixels at 250m resolution

# Kharif season months (June=6 through November=11)
KHARIF_MONTHS = [6, 7, 8, 9, 10, 11]
MONTH_NAMES = ["jun", "jul", "aug", "sep", "oct", "nov"]

# Rate limiting
SLEEP_BETWEEN_REQUESTS = 0.3   # seconds between GEE API calls
SLEEP_BETWEEN_DISTRICTS = 1.0  # seconds between districts

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOOKUP_CSV = os.path.join(PROJECT_ROOT, "district_year_lookup.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "satellite_images")
NDVI_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "ndvi_timeseries.csv")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "_progress.json")

# Minimum year for MODIS data
MIN_MODIS_YEAR = 2000

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize()
        log.info("GEE initialized successfully (existing credentials).")
    except Exception:
        log.info("Attempting GEE authentication...")
        ee.Authenticate()
        ee.Initialize(project='famous-charge-479508-f6')
        log.info("GEE initialized after authentication.")


def get_month_date_range(year: int, month: int) -> tuple[str, str]:
    """Get start and end date strings for a given year and month."""
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    start = f"{year}-{month:02d}-01"
    end = f"{year}-{month:02d}-{last_day}"
    return start, end


def extract_image_patch(lat: float, lon: float, year: int, month: int) -> np.ndarray | None:
    """
    Extract a 64x64 image patch from MODIS MOD13Q1 for a given location, year, month.

    Returns:
        numpy array of shape (4, 64, 64) — channels-first (Red, NIR, NDVI, EVI)
        or None if extraction fails
    """
    try:
        # Create point and buffer geometry
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(BUFFER_RADIUS_M).bounds()

        # Date range for this month
        start_date, end_date = get_month_date_range(year, month)

        # Filter MODIS collection
        collection = (
            ee.ImageCollection(MODIS_COLLECTION)
            .filterDate(start_date, end_date)
            .filterBounds(point)
            .select(BANDS)
        )

        # Check if any images exist
        count = collection.size().getInfo()
        if count == 0:
            log.warning(f"  No MODIS images for {year}-{month:02d} at ({lat}, {lon})")
            return None

        # Median composite (removes cloud artifacts)
        composite = collection.median()

        # Apply scale factors
        red_nir = composite.select(["sur_refl_b01", "sur_refl_b02"]).multiply(REFLECTANCE_SCALE)
        ndvi_evi = composite.select(["NDVI", "EVI"]).multiply(VI_SCALE)
        scaled = red_nir.addBands(ndvi_evi)

        # Clip to region and get as numpy array
        # Use getRegion for small areas (faster than Export)
        # Sample the image at native resolution within the region
        patch = scaled.clipToBoundsAndScale(
            geometry=region,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
        )

        # Convert to numpy via sampleRectangle
        band_arrays = []
        for band_name in ["sur_refl_b01", "sur_refl_b02", "NDVI", "EVI"]:
            arr = patch.select(band_name).sampleRectangle(
                region=region,
                defaultValue=0
            ).get(band_name).getInfo()
            band_arrays.append(np.array(arr, dtype=np.float32))

        # Stack into (4, H, W) array
        image = np.stack(band_arrays, axis=0)

        # Resize to exactly 64x64 if needed (sampleRectangle may return slightly different sizes)
        if image.shape[1] != IMAGE_SIZE or image.shape[2] != IMAGE_SIZE:
            from scipy.ndimage import zoom
            factors = (1, IMAGE_SIZE / image.shape[1], IMAGE_SIZE / image.shape[2])
            image = zoom(image, factors, order=1)

        # Clamp values to valid range
        image = np.clip(image, 0, 1)

        return image.astype(np.float32)

    except Exception as e:
        log.error(f"  Error extracting patch for {year}-{month:02d} at ({lat}, {lon}): {e}")
        return None


def extract_mean_ndvi(lat: float, lon: float, year: int, month: int) -> float | None:
    """
    Extract district-mean NDVI for a given location, year, month.

    Returns:
        Mean NDVI value (float) or None if extraction fails
    """
    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(BUFFER_RADIUS_M)

        start_date, end_date = get_month_date_range(year, month)

        collection = (
            ee.ImageCollection(MODIS_COLLECTION)
            .filterDate(start_date, end_date)
            .filterBounds(point)
            .select(["NDVI"])
        )

        count = collection.size().getInfo()
        if count == 0:
            return None

        composite = collection.median()
        scaled = composite.multiply(VI_SCALE)

        # Compute mean NDVI over the buffer region
        mean_val = scaled.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=250,
            maxPixels=1e6,
        ).get("NDVI").getInfo()

        if mean_val is not None:
            return round(float(mean_val), 6)
        return None

    except Exception as e:
        log.error(f"  Error extracting mean NDVI for {year}-{month:02d}: {e}")
        return None


def load_progress() -> set:
    """Load set of completed district-year keys from progress file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_progress(completed: set):
    """Save set of completed district-year keys to progress file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(sorted(completed), f)


def append_ndvi_row(row_data: dict):
    """Append a single row to the NDVI CSV file."""
    df_row = pd.DataFrame([row_data])
    if os.path.exists(NDVI_CSV):
        df_row.to_csv(NDVI_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(NDVI_CSV, mode="w", header=True, index=False)


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def run_extraction():
    """Main function: extract satellite images + NDVI for all district-year pairs."""

    # Initialize GEE
    initialize_gee()

    # Load lookup table
    if not os.path.exists(LOOKUP_CSV):
        log.error(f"Lookup CSV not found: {LOOKUP_CSV}")
        sys.exit(1)

    lookup = pd.read_csv(LOOKUP_CSV)
    log.info(f"Loaded {len(lookup)} district-year pairs from lookup CSV.")

    # Filter to years >= 2000 (MODIS availability)
    lookup = lookup[lookup["year"] >= MIN_MODIS_YEAR].reset_index(drop=True)
    log.info(f"After filtering year >= {MIN_MODIS_YEAR}: {len(lookup)} pairs remaining.")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load progress (for resume capability)
    completed = load_progress()
    if completed:
        log.info(f"Resuming: {len(completed)} district-year pairs already completed.")

    # Group by district for logging
    districts = lookup["district"].unique()
    total_districts = len(districts)
    total_pairs = len(lookup)
    pairs_done = 0
    pairs_skipped = 0

    log.info(f"Starting extraction: {total_pairs} pairs across {total_districts} districts")
    log.info(f"Output images: {OUTPUT_DIR}")
    log.info(f"Output NDVI CSV: {NDVI_CSV}")
    log.info("=" * 70)

    for d_idx, district in enumerate(districts):
        district_rows = lookup[lookup["district"] == district]
        lat = district_rows.iloc[0]["latitude"]
        lon = district_rows.iloc[0]["longitude"]

        log.info(f"[{d_idx + 1}/{total_districts}] {district} ({lat}, {lon}) — "
                 f"{len(district_rows)} years")

        for _, row in district_rows.iterrows():
            year = int(row["year"])
            key = f"{district}_{year}"

            # Skip if already done
            if key in completed:
                pairs_skipped += 1
                continue

            # Extract NDVI time series (mean values) for this district-year
            ndvi_values = {}
            for month, month_name in zip(KHARIF_MONTHS, MONTH_NAMES):
                mean_ndvi = extract_mean_ndvi(lat, lon, year, month)
                ndvi_values[f"ndvi_{month_name}"] = mean_ndvi
                time.sleep(SLEEP_BETWEEN_REQUESTS)

            # Save NDVI row
            ndvi_row = {"district": district, "year": year, **ndvi_values}
            append_ndvi_row(ndvi_row)

            # Extract satellite image patches for each month
            images_saved = 0
            for month, month_name in zip(KHARIF_MONTHS, MONTH_NAMES):
                filename = f"{district}_{year}_{month:02d}.npy"
                filepath = os.path.join(OUTPUT_DIR, filename)

                patch = extract_image_patch(lat, lon, year, month)
                if patch is not None:
                    np.save(filepath, patch)
                    images_saved += 1
                else:
                    # Save zero array as placeholder for missing months
                    zero_patch = np.zeros((NUM_BANDS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
                    np.save(filepath, zero_patch)

                time.sleep(SLEEP_BETWEEN_REQUESTS)

            # Mark as completed
            completed.add(key)
            pairs_done += 1

            log.info(f"  {district} {year}: {images_saved}/6 images saved, "
                     f"NDVI=[{', '.join(f'{v:.3f}' if v else 'None' for v in ndvi_values.values())}]")

        # Save progress after each district
        save_progress(completed)
        log.info(f"  ✓ District complete. Progress saved. "
                 f"({pairs_done + pairs_skipped}/{total_pairs} total)")
        time.sleep(SLEEP_BETWEEN_DISTRICTS)

    log.info("=" * 70)
    log.info(f"EXTRACTION COMPLETE!")
    log.info(f"  Total pairs processed: {pairs_done}")
    log.info(f"  Pairs skipped (already done): {pairs_skipped}")
    log.info(f"  Image files: {OUTPUT_DIR}")
    log.info(f"  NDVI CSV: {NDVI_CSV}")

    # Final validation
    validate_outputs(total_pairs)


def validate_outputs(expected_pairs: int):
    """Validate the extraction outputs."""
    log.info("\n--- Validation ---")

    # Check NDVI CSV
    if os.path.exists(NDVI_CSV):
        df = pd.read_csv(NDVI_CSV)
        log.info(f"NDVI CSV: {len(df)} rows (expected ~{expected_pairs})")
        log.info(f"  Columns: {list(df.columns)}")
        ndvi_cols = [c for c in df.columns if c.startswith("ndvi_")]
        non_null = df[ndvi_cols].notna().sum()
        log.info(f"  Non-null counts:\n{non_null.to_string()}")

        # Check value ranges
        for col in ndvi_cols:
            valid = df[col].dropna()
            if len(valid) > 0:
                log.info(f"  {col}: min={valid.min():.4f}, max={valid.max():.4f}, "
                         f"mean={valid.mean():.4f}")
    else:
        log.warning(f"NDVI CSV not found: {NDVI_CSV}")

    # Check image files
    import glob
    npy_files = glob.glob(os.path.join(OUTPUT_DIR, "*.npy"))
    log.info(f"\nImage files: {len(npy_files)} .npy files found")

    if npy_files:
        # Check a sample
        sample = np.load(npy_files[0])
        log.info(f"  Sample shape: {sample.shape} (expected (4, 64, 64))")
        log.info(f"  Sample dtype: {sample.dtype}")
        log.info(f"  Sample value range: [{sample.min():.4f}, {sample.max():.4f}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=" * 70)
    log.info("STEP 2: Satellite Image + NDVI Extraction from Google Earth Engine")
    log.info("  MODIS MOD13Q1 | 250m | 4 bands (Red, NIR, NDVI, EVI)")
    log.info("  Kharif season: June–November | Year range: 2000–2022")
    log.info("=" * 70)

    run_extraction()
