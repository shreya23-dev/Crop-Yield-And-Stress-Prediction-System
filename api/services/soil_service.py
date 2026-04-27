import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DS_PATH = DATA_DIR / "final_dataset.csv"

_soil_cache = {}

def get_soil_ph(district: str) -> float:
    """
    Looks up soil pH. If not found, returns a safe default (6.5).
    """
    if not _soil_cache and DS_PATH.exists():
        df = pd.read_csv(DS_PATH, usecols=["district", "soil_ph"])
        df = df.drop_duplicates(subset=["district"])
        for _, row in df.iterrows():
            d = row["district"].strip()
            _soil_cache[d] = float(row["soil_ph"])
            
    return _soil_cache.get(district.strip(), 6.5)
