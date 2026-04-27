"""
FastAPI backend — redesigned for 3-input user experience.

POST /api/predict  { crop, district, year }
GET  /api/crops
GET  /api/districts
GET  /api/districts/{district}/crops
GET  /api/results
GET  /api/health
"""

from __future__ import annotations
import time, sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.routes.validate import router as validate_router
from api.routes.auth import router as auth_router, router_compat as auth_router_compat
from api.routes.history import router as history_router
from api.routes.security import require_farmer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    PredictRequest, PredictResponse, CropListResponse,
    DistrictListResponse, DistrictInfo
)
from api.config import DISTRICT_COORDINATES, DISTRICT_CROPS, VALID_CROPS
from api.services.ndvi_service    import fetch_ndvi_timeseries, get_peak_month, get_health_status
from api.services.weather_service import (
    fetch_weather_data, compute_thermal_stress,
    compute_water_stress, summarize_weather
)
from api.services.soil_service      import get_soil_ph
from api.services.satellite_service import fetch_satellite_images, extract_cnn_features
from api.services.stress_service    import get_stress_level, generate_stress_description
from api.services.prediction_service import (
    get_store, run_prediction, compute_yield_range, estimate_confidence
)
from api.services.mongo_service import ensure_database_initialized, save_prediction_history

TABLES_DIR = PROJECT_ROOT / "results" / "tables"

# ---- Lifespan ---------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_database_initialized()
    try:
        store = get_store()
        store.load()
    except Exception as e:
        # Keep API alive for auth/history endpoints even if model init fails.
        print(f"[Startup] Model loading skipped: {e}")
    yield

app = FastAPI(
    title="Crop Yield Prediction API",
    description="Physics-Informed Multimodal Crop Yield Prediction for Maharashtra",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(validate_router)
app.include_router(auth_router)
app.include_router(auth_router_compat)
app.include_router(history_router)

# ---- Health -----------------------------------------------------------------

@app.get("/api/health", tags=["Health"])
def health():
    store = get_store()
    return {
        "status": "healthy",
        "model_loaded": store.model is not None,
        "model_version": store.model_name or "None",
        "version": "v2",
    }

# kept for backward compat
@app.get("/", tags=["Health"])
def health_root():
    return health()

# ---- Metadata ---------------------------------------------------------------

@app.get("/api/crops", response_model=CropListResponse, tags=["Metadata"])
def list_crops():
    return {"crops": VALID_CROPS}

# backward compat
@app.get("/crops", response_model=CropListResponse, tags=["Metadata"])
def list_crops_compat():
    return list_crops()


@app.get("/api/districts", response_model=DistrictListResponse, tags=["Metadata"])
def list_districts():
    result = []
    for name, (lat, lon) in sorted(DISTRICT_COORDINATES.items()):
        result.append(DistrictInfo(
            name=name, lat=lat, lon=lon,
            available_crops=DISTRICT_CROPS.get(name, VALID_CROPS)
        ))
    return {"districts": result}

# backward compat
@app.get("/districts", tags=["Metadata"])
def list_districts_compat():
    return {"districts": sorted(DISTRICT_COORDINATES.keys())}


@app.get("/api/districts/{district}/crops", tags=["Metadata"])
def get_district_crops(district: str):
    if district not in DISTRICT_COORDINATES:
        raise HTTPException(status_code=404, detail=f"Unknown district: {district}")
    crops = DISTRICT_CROPS.get(district, VALID_CROPS)
    return {"district": district, "crops": crops}


# ---- Results ----------------------------------------------------------------

@app.get("/api/results", tags=["Metrics"])
@app.get("/results", tags=["Metrics"])
def get_results(experiment: Optional[str] = None):
    """Return experiment OOF metrics for the Results page."""
    all_rows = []
    files = {
        "exp1": "model_metrics.csv",
        "exp2": "model_metrics_exp2.csv",
        "exp3": "model_metrics_exp3.csv",
        "exp4": "model_metrics_exp4.csv",
        "exp5": "model_metrics_exp5.csv",
        "exp6": "model_metrics_exp6_ablation.csv",
    }
    for exp_id, fname in files.items():
        if experiment and exp_id != experiment:
            continue
        fpath = TABLES_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            df["experiment"] = exp_id
            all_rows.append(df)
    if not all_rows:
        raise HTTPException(status_code=404, detail="No results found. Run training first.")

    combined = pd.concat(all_rows, ignore_index=True)
    oof_rows = combined[combined["split"] == "oof"].replace({float("nan"): None})
    return oof_rows.to_dict(orient="records")


# ---- Main Prediction Endpoint -----------------------------------------------

@app.post("/api/predict", tags=["Prediction"])
async def predict(request: PredictRequest, farmer=Depends(require_farmer)):
    t_start = time.time()

    # -- Validate ---
    if request.crop not in VALID_CROPS:
        raise HTTPException(status_code=400, detail=f"Unknown crop '{request.crop}'. Choose from: {VALID_CROPS}")
    if request.district not in DISTRICT_COORDINATES:
        raise HTTPException(status_code=400, detail=f"Unknown district '{request.district}'.")

    available = DISTRICT_CROPS.get(request.district, VALID_CROPS)
    if request.crop not in available:
        raise HTTPException(
            status_code=400,
            detail=f"'{request.crop}' is not typically grown in {request.district}. "
                   f"Available crops: {available}"
        )

    if request.year < 2000:
        raise HTTPException(status_code=400, detail="Satellite data not available before 2000.")

    store = get_store()
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    lat, lon = DISTRICT_COORDINATES[request.district]

    # -- Fetch data ---
    ndvi_values  = fetch_ndvi_timeseries(request.district, request.year)
    weather_data = await fetch_weather_data(lat, lon, request.year)
    weather_ok   = weather_data is not None
    soil_ph      = get_soil_ph(request.district)
    images       = fetch_satellite_images(request.district, request.year)
    cnn_feat     = extract_cnn_features(images, store.cnn_model)

    # -- Physics stress ---
    thermal = float(compute_thermal_stress(weather_data, request.crop))
    water   = float(compute_water_stress(weather_data, lat, request.crop))
    combined_stress = round(0.4 * thermal + 0.6 * water, 3)

    # -- Model inference ---
    try:
        pred = run_prediction(
            store=store,
            crop=request.crop,
            district=request.district,
            year=request.year,
            ndvi_profile=ndvi_values,
            weather_data=weather_data,
            cnn_features=cnn_feat,
            soil_ph=soil_ph,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {e}")

    yield_pred   = pred["yield"]
    model_stress = pred["stress"]  # from multi-task head if Exp5
    stress_final = float(model_stress) if model_stress is not None else combined_stress

    # -- Confidence ---
    confidence = estimate_confidence(
        crop=request.crop, district=request.district, year=request.year,
        ndvi_values=ndvi_values, weather_ok=weather_ok
    )

    # -- Yield range ---
    yield_range  = compute_yield_range(yield_pred, request.crop, confidence["level"])

    # -- NDVI summary ---
    peak_month  = get_peak_month(ndvi_values)
    health      = get_health_status(ndvi_values)

    # -- Stress labels ---
    stress_level = get_stress_level(stress_final)
    stress_desc  = generate_stress_description(thermal, water)

    # -- Weather summary ---
    wx_summary = summarize_weather(weather_data)

    # -- Yield unit ---
    yield_unit = "Bales/Hectare" if "cotton" in request.crop.lower() else "Tonnes/Hectare"

    elapsed_ms = int((time.time() - t_start) * 1000)

    save_prediction_history(
        farmer_id=farmer["_id"],
        crop=request.crop,
        district=request.district,
        year=request.year,
        predicted_yield=yield_pred,
        yield_unit=yield_unit,
        stress_level=stress_level,
        stress_index=stress_final,
    )

    return {
        "status": "success",
        "prediction": {
            "crop": request.crop,
            "district": request.district,
            "year": request.year,
            "predicted_yield": round(yield_pred, 3),
            "yield_unit": yield_unit,
            "yield_range": yield_range,
        },
        "stress": {
            "overall_index": round(stress_final, 3),
            "level": stress_level,
            "thermal_stress": round(thermal, 3),
            "water_stress": round(water, 3),
            "description": stress_desc,
        },
        "ndvi_profile": {
            "months": ["June", "July", "August", "September", "October", "November"],
            "values": [round(float(v), 4) for v in ndvi_values],
            "peak_month": peak_month,
            "health_status": health,
        },
        "weather_summary": wx_summary,
        "confidence": confidence,
        "metadata": {
            "model_version": store.model_name or "Unknown",
            "data_sources": ["MODIS (NDVI)", "Open-Meteo (Weather)", "Soil Health Card (pH)"],
            "processing_time_ms": elapsed_ms,
        }
    }


# ---- Embeddings (kept for About page) ---------------------------------------

@app.get("/embeddings/districts", tags=["Embeddings"])
def district_embeddings():
    try:
        import tensorflow as tf
        from sklearn.manifold import TSNE

        MODEL_CANDIDATES_PATHS = [
            PROJECT_ROOT / "api" / "models" / "experiment5_pinn_multitask.keras",
            PROJECT_ROOT / "api" / "models" / "experiment4_pinn.keras",
        ]
        DATA_DIR = PROJECT_ROOT / "data" / "processed"

        weights = None
        for mp in MODEL_CANDIDATES_PATHS:
            if not mp.exists(): continue
            try:
                model  = tf.keras.models.load_model(mp)
                layer  = model.get_layer("district_embedding")
                weights = layer.get_weights()[0]
                break
            except Exception:
                continue
        if weights is None:
            raise HTTPException(status_code=503, detail="No model with district embeddings found.")

        ds_path = DATA_DIR / "final_dataset.csv"
        if not ds_path.exists():
            raise HTTPException(status_code=404, detail="final_dataset.csv not found.")

        df = pd.read_csv(ds_path, usecols=["district", "district_id"])
        dist_map = df.drop_duplicates("district_id").sort_values("district_id")
        n = min(len(dist_map), weights.shape[0])
        emb = weights[:n]
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, n - 1))
        coords = tsne.fit_transform(emb)
        result = [
            {"district": row["district"], "x": float(coords[i, 0]), "y": float(coords[i, 1])}
            for i, (_, row) in enumerate(dist_map.head(n).iterrows())
        ]
        return {"embeddings": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
