"""
GET /api/validate?district=Ahilyanagar&crop=Rice&years=5
Compares model predictions on historical data against known actual values.
Returns per-row actual vs predicted, plus aggregate metrics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query

from api.config import DISTRICT_COORDINATES, DISTRICT_CROPS, VALID_CROPS
from api.services.ndvi_service    import fetch_ndvi_timeseries
from api.services.weather_service import summarize_weather
from api.services.soil_service    import get_soil_ph
from api.services.satellite_service import fetch_satellite_images, extract_cnn_features
from api.services.prediction_service import get_store, run_prediction

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DS_PATH      = PROJECT_ROOT / "data" / "processed" / "final_dataset.csv"

_df_cache: Optional[pd.DataFrame] = None

def _load_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        if not DS_PATH.exists():
            raise FileNotFoundError("final_dataset.csv not found")
        _df_cache = pd.read_csv(DS_PATH)
    return _df_cache


@router.get("/api/validate", tags=["Validation"])
async def validate_predictions(
    district: str   = Query(...,  description="District name"),
    crop:     str   = Query(...,  description="Crop name"),
    years:    int   = Query(5,    description="How many recent years to validate", ge=1, le=23),
):
    """
    Run the trained model on historical (district, crop, year) rows
    and compare against actual recorded yield.
    Returns row-by-row actual vs predicted and aggregate metrics.
    """
    if district not in DISTRICT_COORDINATES:
        raise HTTPException(status_code=400, detail=f"Unknown district: {district}")
    if crop not in VALID_CROPS:
        raise HTTPException(status_code=400, detail=f"Unknown crop: {crop}")

    store = get_store()
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    df = _load_df()
    mask = (df["district"].str.strip() == district) & (df["crop"].str.strip() == crop)
    rows = df[mask].sort_values("year", ascending=False).head(years).sort_values("year")

    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data for {crop} in {district}."
        )

    results = []
    actuals, preds = [], []

    for _, row in rows.iterrows():
        year    = int(row["year"])
        actual  = float(row["yield_value"])

        ndvi_profile = [
            float(row.get("ndvi_jun", 0.3)),
            float(row.get("ndvi_jul", 0.42)),
            float(row.get("ndvi_aug", 0.55)),
            float(row.get("ndvi_sep", 0.58)),
            float(row.get("ndvi_oct", 0.45)),
            float(row.get("ndvi_nov", 0.3)),
        ]

        # Build fake weather array from dataset columns
        weather_data = _build_weather_from_row(row)
        soil_ph      = float(row.get("soil_ph", 6.5))

        images   = fetch_satellite_images(district, year)
        cnn_feat = extract_cnn_features(images, store.cnn_model)

        try:
            pred_out = run_prediction(
                store=store,
                crop=crop,
                district=district,
                year=year,
                ndvi_profile=ndvi_profile,
                weather_data=weather_data,
                cnn_features=cnn_feat,
                soil_ph=soil_ph,
            )
            predicted = round(pred_out["yield"], 3)
        except Exception as e:
            predicted = None

        error   = round(predicted - actual, 3) if predicted is not None else None
        pct_err = round(abs(error) / actual * 100, 1) if error is not None and actual > 0 else None

        results.append({
            "year":      year,
            "actual":    round(actual, 3),
            "predicted": predicted,
            "error":     error,
            "pct_error": pct_err,
            "within_15pct": (pct_err is not None and pct_err <= 15),
        })

        if predicted is not None:
            actuals.append(actual)
            preds.append(predicted)

    # Aggregate metrics
    summary = {}
    if actuals:
        errs    = [abs(p - a) for p, a in zip(preds, actuals)]
        ss_res  = sum((p - a)**2 for p, a in zip(preds, actuals))
        ss_tot  = sum((a - np.mean(actuals))**2 for a in actuals)
        r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        summary = {
            "n_years": len(actuals),
            "mae":     round(float(np.mean(errs)), 3),
            "rmse":    round(float(np.sqrt(np.mean([e**2 for e in errs]))), 3),
            "r2":      round(float(r2), 4),
            "within_15pct_pct": round(
                sum(1 for r in results if r["within_15pct"]) / len(results) * 100, 1
            ),
            "actual_mean":  round(float(np.mean(actuals)), 3),
            "predicted_mean": round(float(np.mean(preds)), 3),
        }

    return {
        "district":   district,
        "crop":       crop,
        "yield_unit": "Bales/Hectare" if "cotton" in crop.lower() else "Tonnes/Hectare",
        "rows":       results,
        "summary":    summary,
    }


def _build_weather_from_row(row) -> np.ndarray:
    """
    Reconstructs the weather matrix from dataset columns (22×3).
    """
    data = np.zeros((22, 3), dtype=np.float32)
    for w in range(22):
        i = w + 1
        data[w, 0] = float(row.get(f"week_{i}_temp_mean", 28.0) or 28.0)
        data[w, 1] = float(row.get(f"week_{i}_temp_max",  33.0) or 33.0)
        data[w, 2] = float(row.get(f"week_{i}_rain",       0.0) or 0.0)
    return data
