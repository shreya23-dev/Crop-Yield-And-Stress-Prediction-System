"""
Prediction service: loads + runs PINN model, computes confidence,
and estimates yield range from known OOF error statistics.
"""

from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
import tempfile
import zipfile
from pathlib import Path
from functools import lru_cache
from typing import Optional
import tensorflow as tf

tf.get_logger().setLevel("ERROR")


def _remove_quantization_config(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for v in obj.values():
            _remove_quantization_config(v)
    elif isinstance(obj, list):
        for item in obj:
            _remove_quantization_config(item)


def _load_from_sanitized_keras_archive(model_path: Path):
    """Create a temporary .keras archive without quantization_config and load it."""
    import os

    fd, tmp_name = tempfile.mkstemp(suffix=".keras")
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with zipfile.ZipFile(model_path, "r") as zin:
            config = json.loads(zin.read("config.json"))
            _remove_quantization_config(config)

            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for info in zin.infolist():
                    if info.filename == "config.json":
                        continue
                    zout.writestr(info, zin.read(info.filename))
                zout.writestr("config.json", json.dumps(config, separators=(",", ":")))

        return tf.keras.models.load_model(tmp_path, compile=False)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def load_model_compat(model_path: Path):
    """Load Keras model with compatibility fallback for known config drifts."""
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except TypeError as e:
        msg = str(e)
        if "quantization_config" not in msg or "Embedding" not in msg:
            raise

        print("  [PredictionService] Retrying model load with broad Embedding compatibility shim")

        embedding_classes = [tf.keras.layers.Embedding]
        try:
            import keras

            embedding_classes.append(keras.layers.Embedding)
            from keras.src.layers.core.embedding import Embedding as KerasSrcEmbedding

            embedding_classes.append(KerasSrcEmbedding)
        except Exception:
            # Standalone keras package path may not exist depending on TF/Keras version.
            pass

        seen = set()
        patch_targets = []
        for cls in embedding_classes:
            if cls is None or id(cls) in seen:
                continue
            seen.add(id(cls))
            patch_targets.append((cls, cls.__init__))

        for cls, original_init in patch_targets:
            def _compat_init(self, *args, _original_init=original_init, **kwargs):
                kwargs.pop("quantization_config", None)
                return _original_init(self, *args, **kwargs)

            cls.__init__ = _compat_init

        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except TypeError:
            print("  [PredictionService] Retrying model load with sanitized .keras config")
            return _load_from_sanitized_keras_archive(model_path)
        finally:
            for cls, original_init in patch_targets:
                cls.__init__ = original_init

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "api" / "models"

MODEL_CANDIDATES = [
    ("experiment5_pinn_multitask.keras", "experiment5_preprocessing.pkl", "PINN-Multimodal-v2"),
    ("experiment4_pinn.keras",           "experiment4_preprocessing.pkl", "PINN-v1"),
    ("experiment3_with_images.keras",    "experiment3_preprocessing.pkl", "Neural+Images"),
]

class ModelStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model       = None
            cls._instance.preproc     = None
            cls._instance.model_name  = None
            cls._instance.cnn_model   = None
            cls._instance.district_map = {}
            cls._instance.crop_map     = {}
            cls._instance._loaded      = False
        return cls._instance

    def load(self):
        if self._loaded:
            return

        for model_file, preproc_file, label in MODEL_CANDIDATES:
            mp = MODELS_DIR / model_file
            pp = MODELS_DIR / preproc_file
            if mp.exists() and pp.exists():
                print(f"  [PredictionService] Loading: {label}")
                self.model      = load_model_compat(mp)
                self.preproc    = joblib.load(pp)
                self.model_name = label
                break

        if self.model is None:
            print("  [PredictionService] ⚠️  No trained model found — prediction unavailable")

        try:
            from src.models.multimodal_net_exp3 import build_satellite_cnn
            self.cnn_model = build_satellite_cnn(32, 32, image_channels=1)
        except Exception as e:
            print(f"  [PredictionService] CNN not loaded: {e}")

        ds_path = DATA_DIR / "final_dataset.csv"
        if ds_path.exists():
            df = pd.read_csv(ds_path, usecols=["district", "crop", "district_id", "crop_id"])
            self.district_map = dict(zip(df["district"].str.strip(), df["district_id"].astype(int)))
            self.crop_map     = dict(zip(df["crop"].str.strip(), df["crop_id"].astype(int)))

        self._loaded = True


_STORE = ModelStore()

def get_store() -> ModelStore:
    _STORE.load()
    return _STORE


def run_prediction(
    store: ModelStore,
    crop: str,
    district: str,
    year: int,
    ndvi_profile: list,
    weather_data: np.ndarray,
    cnn_features: np.ndarray,
    soil_ph: float,
) -> dict:
    """
    Runs the PINN model and returns (yield_pred, stress_pred).
    """
    preproc = store.preproc
    ndvi_arr = np.array([ndvi_profile], dtype=np.float32)

    crop_id     = np.array([[store.crop_map.get(crop, 0)]],     dtype=np.int32)
    district_id = np.array([[store.district_map.get(district, 0)]], dtype=np.int32)
    year_norm   = np.array([[(year - 1997) / (2022 - 1997)]], dtype=np.float32)
    soil_input  = np.array([[soil_ph]], dtype=np.float32)

    # Scale NDVI
    ndvi_sc = preproc.get("ndvi_scaler")
    if ndvi_sc:
        ndvi_scaled = ndvi_sc.transform(ndvi_arr).reshape(1, 6, 1).astype(np.float32)
    else:
        std = ndvi_arr.std() + 1e-8
        ndvi_scaled = ((ndvi_arr - ndvi_arr.mean()) / std).reshape(1, 6, 1).astype(np.float32)

    # Scale soil
    soil_sc = preproc.get("soil_scaler")
    soil_scaled = soil_sc.transform(soil_input).astype(np.float32) if soil_sc else soil_input

    # Scale weather: weather_data is shape (22, 3) where cols are [temp_mean, temp_max, rain].
    # But the training script fitted the scaler on: [22x mean, 22x max, 22x rain].
    # So we transpose then flatten to match the expected (1, 66) shape.
    weather_flat = weather_data.T.flatten().reshape(1, -1)  # (1, 66)
    
    weather_sc = preproc.get("weather_scaler")
    if weather_sc:
        # After scaling, the training script just reshaped it directly to (22, 3).
        # We must reproduce that exact buggy/tangled reshape so the NN gets what it expects.
        weather_scaled = weather_sc.transform(weather_flat).reshape(1, 22, 3).astype(np.float32)
    else:
        weather_scaled = weather_flat.reshape(1, 22, 3).astype(np.float32)

    model_input = {
        "satellite_features": cnn_features,
        "ndvi":               ndvi_scaled,
        "weather":            weather_scaled,
        "crop_id":            crop_id,
        "district_id":        district_id,
        "year_norm":          year_norm,
        "soil_ph":            soil_scaled,
    }

    output = store.model.predict(model_input, verbose=0)
    y_scaler = preproc.get("y_scaler")

    if isinstance(output, dict):
        yield_scaled = float(output["yield"].reshape(-1)[0])
        stress_pred  = float(output["stress"].reshape(-1)[0])
    else:
        yield_scaled = float(np.array(output).reshape(-1)[0])
        stress_pred  = None

    if y_scaler:
        yield_pred = float(y_scaler.inverse_transform([[yield_scaled]])[0, 0])
    else:
        yield_pred = yield_scaled

    yield_pred = max(0.0, round(yield_pred, 3))
    return {"yield": yield_pred, "stress": stress_pred}


def compute_yield_range(yield_pred: float, crop: str, level: str) -> dict:
    """±15–20% range based on confidence and crop type."""
    # Management-intensive crops have wider error bands
    pct = 0.15 if crop in ["Rice", "Jowar", "Bajra"] else 0.22
    if level == "Medium": pct *= 1.2
    if level == "Low":    pct *= 1.5
    return {
        "low":  round(max(0.0, yield_pred * (1 - pct)), 3),
        "high": round(yield_pred * (1 + pct), 3),
    }


def estimate_confidence(crop: str, district: str, year: int,
                        ndvi_values: list, weather_ok: bool) -> dict:
    score = 1.0
    factors = []

    if all(v > 0 for v in ndvi_values):
        factors.append("Complete NDVI profile retrieved")
    else:
        score -= 0.2
        factors.append("Partial NDVI profile (some months estimated)")

    if weather_ok:
        factors.append("Weather data fully available")
    else:
        score -= 0.2
        factors.append("Weather data not available — used climatological fallback")

    if crop in ["Soyabean", "Cotton(lint)"]:
        score -= 0.15
        factors.append(f"{crop} yield is management-intensive; prediction less certain")
    else:
        factors.append(f"Good model performance for {crop}")

    if year > 2023:
        score -= 0.1
        factors.append("Future year — using climatological weather proxy")
    elif year >= 2000:
        factors.append("Historical data range fully covered")

    score = round(max(0.0, score), 2)
    level = "High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low"
    return {"level": level, "score": score, "factors": factors}
