import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DS_PATH = DATA_DIR / "final_dataset.csv"

_ndvi_cache = {}

def fetch_ndvi_timeseries(district: str, year: int) -> list:
    """
    Since we don't have GEE credentials in the production backend,
    we simulate fetching it by reading from the curated dataset.
    Returns: list of 6 floats (Jun-Nov) or fallback if not found.
    """
    if not _ndvi_cache and DS_PATH.exists():
        df = pd.read_csv(DS_PATH, usecols=["district", "year", 
            "ndvi_jun", "ndvi_jul", "ndvi_aug", "ndvi_sep", "ndvi_oct", "ndvi_nov"])
        for _, row in df.iterrows():
            d = row["district"].strip()
            y = int(row["year"])
            _ndvi_cache[(d, y)] = [
                row["ndvi_jun"], row["ndvi_jul"], row["ndvi_aug"],
                row["ndvi_sep"], row["ndvi_oct"], row["ndvi_nov"]
            ]
            
    key = (district.strip(), year)
    if key in _ndvi_cache:
        return _ndvi_cache[key]
        
    # Fallback to general district average, else static defaults
    district_averages = [v for k, v in _ndvi_cache.items() if k[0] == district]
    if district_averages:
        avg = np.mean(district_averages, axis=0)
        return [float(x) for x in avg]
        
    return [0.25, 0.42, 0.58, 0.61, 0.45, 0.30]

def get_peak_month(ndvi_vals: list) -> str:
    month_names = ["June", "July", "August", "September", "October", "November"]
    peak_idx = np.argmax(ndvi_vals)
    return month_names[peak_idx]

def get_health_status(ndvi_vals: list) -> str:
    avg_val = np.mean(ndvi_vals)
    if avg_val > 0.6: return "Excellent"
    if avg_val > 0.5: return "Good"
    if avg_val > 0.35: return "Fair"
    return "Poor"
