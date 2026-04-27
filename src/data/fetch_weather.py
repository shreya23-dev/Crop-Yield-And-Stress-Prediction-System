"""
Step 3: Weekly Weather Data Extraction from Open-Meteo Historical API

Extracts for each (district, year) pair in district_year_lookup.csv:
  - Daily temperature_mean, temperature_max, precipitation_sum
  - Aggregated into weekly values for Kharif season (June 1 – November 30)
  - Padded/trimmed to exactly 22 weeks

Output: data/processed/weather_timeseries.csv
  Columns: district, year, latitude, longitude,
           week_1_temp_mean, week_1_temp_max, week_1_rain,
           week_2_temp_mean, week_2_temp_max, week_2_rain,
           ...,
           week_22_temp_mean, week_22_temp_max, week_22_rain

Usage:
    python src/data/fetch_weather.py

Prerequisites:
    pip install requests pandas numpy
    No API key needed — Open-Meteo is free.
"""

import os
import sys
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Open-Meteo Historical Weather Archive API
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Daily variables to fetch
DAILY_VARS = "temperature_2m_mean,temperature_2m_max,precipitation_sum"

# Kharif season window
SEASON_START_MONTH = 6   # June
SEASON_START_DAY = 1
SEASON_END_MONTH = 11    # November
SEASON_END_DAY = 30

# Number of weeks to output (June 1 – Nov 30 ≈ 26 weeks, trimmed to 22 per spec)
NUM_WEEKS = 22

# Rate limiting
SLEEP_BETWEEN_REQUESTS = 0.5  # seconds — respect Open-Meteo rate limits
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOOKUP_CSV = os.path.join(PROJECT_ROOT, "district_year_lookup.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "weather_timeseries.csv")
PROGRESS_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "_weather_progress.json")

# ---------------------------------------------------------------------------
# Logging
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


def fetch_daily_weather(lat: float, lon: float, year: int) -> pd.DataFrame | None:
    """
    Fetch daily weather data for one district-year from Open-Meteo.

    Returns DataFrame with columns: date, temperature_2m_mean, temperature_2m_max, precipitation_sum
    or None on failure.
    """
    start_date = f"{year}-{SEASON_START_MONTH:02d}-{SEASON_START_DAY:02d}"
    end_date = f"{year}-{SEASON_END_MONTH:02d}-{SEASON_END_DAY:02d}"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": DAILY_VARS,
        "timezone": "Asia/Kolkata",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(API_URL, params=params, timeout=30)

            if resp.status_code == 429:
                # Rate limited — wait and retry
                wait = RETRY_DELAY * attempt
                log.warning(f"  Rate limited. Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if "daily" not in data:
                log.warning(f"  No daily data returned for {year} at ({lat}, {lon})")
                return None

            daily = data["daily"]
            df = pd.DataFrame({
                "date": pd.to_datetime(daily["time"]),
                "temp_mean": daily["temperature_2m_mean"],
                "temp_max": daily["temperature_2m_max"],
                "rain": daily["precipitation_sum"],
            })

            return df

        except requests.exceptions.Timeout:
            log.warning(f"  Timeout (attempt {attempt}/{MAX_RETRIES}) for {year} at ({lat}, {lon})")
            time.sleep(RETRY_DELAY)
        except requests.exceptions.RequestException as e:
            log.error(f"  Request error (attempt {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY)
        except (KeyError, ValueError) as e:
            log.error(f"  Data parsing error for {year} at ({lat}, {lon}): {e}")
            return None

    log.error(f"  Failed after {MAX_RETRIES} retries for {year} at ({lat}, {lon})")
    return None


def aggregate_daily_to_weekly(daily_df: pd.DataFrame) -> dict:
    """
    Aggregate daily weather data into exactly NUM_WEEKS (22) weekly values.

    Weekly aggregation:
      - temp_mean: mean of daily means
      - temp_max: mean of daily maxes (representative weekly max temperature)
      - rain: sum of daily precipitation

    Returns dict with keys like week_1_temp_mean, week_1_temp_max, week_1_rain, etc.
    """
    # Assign week number (0-indexed) based on days since June 1
    daily_df = daily_df.copy()
    day_of_season = (daily_df["date"] - daily_df["date"].iloc[0]).dt.days
    daily_df["week"] = day_of_season // 7

    weekly_data = {}

    for w in range(NUM_WEEKS):
        week_rows = daily_df[daily_df["week"] == w]

        if len(week_rows) > 0:
            weekly_data[f"week_{w+1}_temp_mean"] = round(week_rows["temp_mean"].mean(), 2)
            weekly_data[f"week_{w+1}_temp_max"] = round(week_rows["temp_max"].mean(), 2)
            weekly_data[f"week_{w+1}_rain"] = round(week_rows["rain"].sum(), 2)
        else:
            # Pad with NaN if week has no data (shouldn't happen for Jun-Nov)
            weekly_data[f"week_{w+1}_temp_mean"] = np.nan
            weekly_data[f"week_{w+1}_temp_max"] = np.nan
            weekly_data[f"week_{w+1}_rain"] = np.nan

    return weekly_data


def load_progress() -> set:
    """Load completed district-year keys."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_progress(completed: set):
    """Save completed district-year keys."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(sorted(completed), f)


def append_row(row_data: dict):
    """Append a single row to the output CSV."""
    df_row = pd.DataFrame([row_data])
    if os.path.exists(OUTPUT_CSV):
        df_row.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(OUTPUT_CSV, mode="w", header=True, index=False)


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def run_extraction():
    """Fetch weather data for all district-year pairs."""

    # Load lookup table
    if not os.path.exists(LOOKUP_CSV):
        log.error(f"Lookup CSV not found: {LOOKUP_CSV}")
        sys.exit(1)

    lookup = pd.read_csv(LOOKUP_CSV)
    total_pairs = len(lookup)
    log.info(f"Loaded {total_pairs} district-year pairs.")

    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Resume support
    completed = load_progress()
    if completed:
        log.info(f"Resuming: {len(completed)} pairs already completed.")

    districts = lookup["district"].unique()
    total_districts = len(districts)
    pairs_done = 0
    pairs_skipped = 0
    errors = 0

    log.info(f"Fetching weather for {total_pairs} pairs across {total_districts} districts")
    log.info(f"Output: {OUTPUT_CSV}")
    log.info("=" * 70)

    for d_idx, district in enumerate(districts):
        district_rows = lookup[lookup["district"] == district]
        lat = district_rows.iloc[0]["latitude"]
        lon = district_rows.iloc[0]["longitude"]
        years = sorted(district_rows["year"].values)

        log.info(f"[{d_idx + 1}/{total_districts}] {district} ({lat}, {lon}) — "
                 f"{len(years)} years ({years[0]}-{years[-1]})")

        for year in years:
            key = f"{district}_{year}"

            if key in completed:
                pairs_skipped += 1
                continue

            # Fetch daily data from Open-Meteo
            daily_df = fetch_daily_weather(lat, lon, int(year))

            if daily_df is not None and len(daily_df) > 0:
                # Aggregate to weekly
                weekly = aggregate_daily_to_weekly(daily_df)

                # Build row
                row = {
                    "district": district,
                    "year": int(year),
                    "latitude": lat,
                    "longitude": lon,
                    **weekly,
                }
                append_row(row)
                pairs_done += 1

                # Summary for this pair
                total_rain = sum(v for k, v in weekly.items()
                                 if k.endswith("_rain") and not np.isnan(v))
                avg_temp = np.nanmean([v for k, v in weekly.items()
                                       if k.endswith("_temp_mean")])
                log.info(f"  {year}: avg_temp={avg_temp:.1f}°C, "
                         f"total_rain={total_rain:.0f}mm")
            else:
                errors += 1
                log.warning(f"  {year}: FAILED — no data returned")

            # Mark done and rate-limit
            completed.add(key)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        # Save progress after each district
        save_progress(completed)
        log.info(f"  ✓ District complete. ({pairs_done + pairs_skipped}/{total_pairs})")

    log.info("=" * 70)
    log.info("EXTRACTION COMPLETE!")
    log.info(f"  Rows fetched:  {pairs_done}")
    log.info(f"  Rows skipped:  {pairs_skipped}")
    log.info(f"  Errors:        {errors}")
    log.info(f"  Output:        {OUTPUT_CSV}")

    # Validation
    validate_output(total_pairs)


def validate_output(expected: int):
    """Validate the weather CSV."""
    log.info("\n--- Validation ---")

    if not os.path.exists(OUTPUT_CSV):
        log.warning("Output CSV not found!")
        return

    df = pd.read_csv(OUTPUT_CSV)
    log.info(f"Rows: {len(df)} (expected ~{expected})")
    log.info(f"Columns: {len(df.columns)} (expected {4 + NUM_WEEKS * 3} = {4 + NUM_WEEKS * 3})")

    # Check for NaN
    weather_cols = [c for c in df.columns if c.startswith("week_")]
    nan_count = df[weather_cols].isna().sum().sum()
    log.info(f"Total NaN cells in weather data: {nan_count}")

    # Temperature sanity check (Maharashtra climate)
    temp_cols = [c for c in weather_cols if "temp_mean" in c]
    all_temps = df[temp_cols].values.flatten()
    all_temps = all_temps[~np.isnan(all_temps)]
    if len(all_temps) > 0:
        log.info(f"Temperature range: {all_temps.min():.1f}°C — {all_temps.max():.1f}°C "
                 f"(expected ~20-35°C for Maharashtra Kharif)")

    # Rainfall sanity check
    rain_cols = [c for c in weather_cols if "rain" in c]
    season_rain = df[rain_cols].sum(axis=1)
    log.info(f"Season total rainfall: min={season_rain.min():.0f}mm, "
             f"max={season_rain.max():.0f}mm, mean={season_rain.mean():.0f}mm")

    # Districts check
    log.info(f"Districts: {df['district'].nunique()} unique")
    log.info(f"Year range: {df['year'].min()} — {df['year'].max()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=" * 70)
    log.info("STEP 3: Weekly Weather Data from Open-Meteo Historical API")
    log.info("  Variables: temp_mean, temp_max, precipitation")
    log.info("  Season: June 1 – November 30 (Kharif)")
    log.info(f"  Output: {NUM_WEEKS} weeks × 3 variables per district-year")
    log.info("=" * 70)

    run_extraction()
