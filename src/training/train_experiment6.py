"""
Experiment 6: Ablation Study.

Tests the full PINN model (Exp4 architecture) with one modality removed at a time:
  A. No satellite images   (satellite_features = zeros)
  B. No NDVI time series   (ndvi = zeros)
  C. No weather data       (weather = zeros)
  D. No soil pH            (soil_ph = zeros)
  E. No physics loss       (all lambdas = 0, reduces to Exp3)
  F. No crop/district emb  (replace embeddings with fixed zero vectors)

Each ablation runs full 5-fold CV. Results saved to model_metrics_exp6_ablation.csv.

Usage:
    python src/training/train_experiment6.py
    python src/training/train_experiment6.py --epochs 40 --skip-ablation E F
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.multimodal_net_exp3 import build_experiment3_fast_model, build_satellite_cnn
from src.models.physics_loss import compute_all_physics_labels

# ---- paths ----------------------------------------------------------------
DATA_CSV        = PROJECT_ROOT / "data" / "processed" / "final_dataset.csv"
SAT_DIR         = PROJECT_ROOT / "data" / "processed" / "satellite_images"
RESULTS_PATH    = PROJECT_ROOT / "results" / "tables" / "model_metrics_exp6_ablation.csv"
SAT_FEAT_CACHE  = PROJECT_ROOT / "data" / "processed" / "satellite_cnn_features.npy"

SAT_MONTHS = [6, 7, 8, 9, 10, 11]
IMAGE_H    = 32
IMAGE_W    = 32

LAMBDA_GROWTH = 0.10
LAMBDA_TEMP   = 0.05
LAMBDA_WATER  = 0.05

SEED = 42

# All ablation configs: id -> (label, description)
ABLATION_CONFIGS = {
    "A": ("No Satellite Images",     "Zero out satellite_features input"),
    "B": ("No NDVI Time Series",     "Zero out ndvi input"),
    "C": ("No Weather Data",         "Zero out weather input"),
    "D": ("No Soil pH",              "Zero out soil_ph input"),
    "E": ("No Physics Loss",         "Lambda1=Lambda2=Lambda3=0 (standard MSE)"),
    "F": ("No Crop/District Embeds", "Zero out crop_id and district_id embeddings"),
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_ndvi_columns(df) -> List[str]:
    return ["ndvi_jun", "ndvi_jul", "ndvi_aug", "ndvi_sep", "ndvi_oct", "ndvi_nov"]


def get_weather_columns(df, var) -> List[str]:
    return [f"week_{w}_{var}" for w in range(1, 23)]


def make_stratify_labels(df, min_count=5) -> pd.Series:
    combo  = df["crop"].astype(str) + "__" + df["district"].astype(str)
    counts = combo.value_counts()
    return combo.where(combo.map(counts) >= min_count, "__RARE__")


def evaluate_row(ablation_id, label, split, crop, y_true, y_pred) -> dict:
    return {
        "ablation_id": ablation_id,
        "ablation_label": label,
        "split": split,
        "crop": crop,
        "r2":   r2_score(y_true, y_pred),
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Satellite loading + CNN features
# ---------------------------------------------------------------------------

def load_raw_images(df) -> np.ndarray:
    N = len(df)
    images = np.zeros((N, 6, IMAGE_H, IMAGE_W, 1), dtype=np.float32)
    loaded = 0
    print(f"  Loading satellite images for {N} rows ...")
    for i, (_, row) in enumerate(df.iterrows()):
        district, year = row["district"], int(row["year"])
        arrs, ok = [], True
        for month in SAT_MONTHS:
            fpath = SAT_DIR / f"{district}_{year}_{month:02d}.npy"
            if not fpath.exists(): ok = False; break
            arr = np.load(fpath).astype(np.float32)
            if arr.ndim == 3: arr = arr[:, :, 0]
            if arr.shape != (IMAGE_H, IMAGE_W):
                arr = tf.image.resize(arr[..., np.newaxis], [IMAGE_H, IMAGE_W]).numpy()[..., 0]
            arrs.append(arr)
        if ok:
            for m, arr in enumerate(arrs): images[i, m, :, :, 0] = arr
            loaded += 1
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{N} processed ({loaded} images)", flush=True)
    print(f"  Done. {loaded}/{N} with images.")
    return images


def get_cnn_features(images) -> np.ndarray:
    if SAT_FEAT_CACHE.exists():
        print("  Loading CNN feature cache ...")
        feats = np.load(SAT_FEAT_CACHE)
        if feats.shape[0] == len(images):
            return feats
    print("  Extracting CNN features ...")
    t0  = time.time()
    cnn = build_satellite_cnn(IMAGE_H, IMAGE_W, image_channels=1)
    N   = len(images)
    feats = np.zeros((N, 6, 128), dtype=np.float32)
    mn    = images.min(); rng = max(images.max() - mn, 1e-6)
    imgs  = np.clip((images - mn) / rng, 0.0, 1.0)
    for m in range(6):
        feats[:, m, :] = cnn.predict(imgs[:, m, :, :, :], batch_size=64, verbose=0)
        print(f"    Month {m+1}/6  ({time.time()-t0:.1f}s)", flush=True)
    np.save(SAT_FEAT_CACHE, feats)
    return feats


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_inputs(df, cnn_feats, ndvi_sc=None, weather_sc=None, soil_sc=None, fit=False):
    ndvi_cols    = get_ndvi_columns(df)
    weather_cols = (get_weather_columns(df, "temp_mean") +
                    get_weather_columns(df, "temp_max") +
                    get_weather_columns(df, "rain"))
    ndvi        = df[ndvi_cols].to_numpy(dtype=np.float32)
    weather_flat = df[weather_cols].to_numpy(dtype=np.float32)
    soil        = df[["soil_ph"]].to_numpy(dtype=np.float32)
    year        = df[["year_normalized"]].to_numpy(dtype=np.float32)
    crop_id     = df[["crop_id"]].to_numpy(dtype=np.int32)
    district_id = df[["district_id"]].to_numpy(dtype=np.int32)

    if fit or ndvi_sc is None:    ndvi_sc    = StandardScaler().fit(ndvi)
    if fit or weather_sc is None: weather_sc = StandardScaler().fit(weather_flat)
    if fit or soil_sc is None:    soil_sc    = StandardScaler().fit(soil)

    ndvi_s    = ndvi_sc.transform(ndvi).reshape(-1, 6, 1).astype(np.float32)
    weather_s = weather_sc.transform(weather_flat).reshape(-1, 22, 3).astype(np.float32)
    soil_s    = soil_sc.transform(soil).astype(np.float32)

    return (
        {"satellite_features": cnn_feats, "ndvi": ndvi_s, "weather": weather_s,
         "crop_id": crop_id, "district_id": district_id, "year_norm": year, "soil_ph": soil_s},
        ndvi_sc, weather_sc, soil_sc,
    )


def apply_ablation(inputs: dict, ablation_id: str) -> dict:
    """Zero out the specified modality in an input dict (making a copy)."""
    inp = {k: v.copy() for k, v in inputs.items()}
    if ablation_id == "A":
        inp["satellite_features"] = np.zeros_like(inp["satellite_features"])
    elif ablation_id == "B":
        inp["ndvi"] = np.zeros_like(inp["ndvi"])
    elif ablation_id == "C":
        inp["weather"] = np.zeros_like(inp["weather"])
    elif ablation_id == "D":
        inp["soil_ph"] = np.zeros_like(inp["soil_ph"])
    elif ablation_id == "F":
        inp["crop_id"]     = np.zeros_like(inp["crop_id"])
        inp["district_id"] = np.zeros_like(inp["district_id"])
    # E (No Physics Loss) handled separately via lambda=0
    return inp


# ---------------------------------------------------------------------------
# PINN trainer (same as Exp4)
# ---------------------------------------------------------------------------

class PINNTrainer:
    def __init__(self, model, y_scaler, lambda1=LAMBDA_GROWTH, lambda2=LAMBDA_TEMP, lambda3=LAMBDA_WATER):
        self.model = model; self.y_scaler = y_scaler
        self.l1 = lambda1; self.l2 = lambda2; self.l3 = lambda3

    def pinn_loss(self, y_true_s, y_pred_s, gp, tp, wp):
        l_yield = tf.reduce_mean(tf.square(y_pred_s - y_true_s))
        y_min   = tf.reduce_min(y_pred_s)
        y_range = tf.maximum(tf.reduce_max(y_pred_s) - y_min, 1e-6)
        y_norm  = (y_pred_s - y_min) / y_range
        l_grow  = tf.reduce_mean(tf.square(y_norm - gp))
        l_temp  = tf.reduce_mean(tf.square(y_norm - tp))
        l_water = tf.reduce_mean(tf.square(y_norm - wp))
        return l_yield + self.l1*l_grow + self.l2*l_temp + self.l3*l_water

    def fit(self, train_in, y_tr, physics_tr, val_in, y_va, physics_va,
            epochs, batch_size, patience):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        N   = len(y_tr); idx = np.arange(N)

        def _t(a): return tf.constant(a.astype(np.float32))
        gp_tr = _t(physics_tr["growth_proxy"]); tp_tr = _t(1.0 - physics_tr["thermal_stress"]); wp_tr = _t(1.0 - physics_tr["water_stress"])
        gp_va = _t(physics_va["growth_proxy"]); tp_va = _t(1.0 - physics_va["thermal_stress"]); wp_va = _t(1.0 - physics_va["water_stress"])
        y_tr_tf = _t(y_tr); y_va_tf = _t(y_va)
        tr_tf   = {k: tf.constant(v) for k, v in train_in.items()}
        va_tf   = {k: tf.constant(v) for k, v in val_in.items()}

        best_val = float("inf"); best_w = self.model.get_weights(); no_imp = 0

        for epoch in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, N, batch_size):
                e = min(s + batch_size, N); bidx = idx[s:e]
                bi   = {k: tf.gather(v, bidx) for k, v in tr_tf.items()}
                y_b  = tf.gather(y_tr_tf, bidx)
                gp_b = tf.gather(gp_tr, bidx); tp_b = tf.gather(tp_tr, bidx); wp_b = tf.gather(wp_tr, bidx)
                with tf.GradientTape() as tape:
                    pred = tf.squeeze(self.model(bi, training=True), axis=1)
                    lv   = self.pinn_loss(y_b, pred, gp_b, tp_b, wp_b)
                grads = tape.gradient(lv, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            vp       = tf.squeeze(self.model(va_tf, training=False), axis=1)
            val_loss = float(self.pinn_loss(y_va_tf, vp, gp_va, tp_va, wp_va).numpy())

            if val_loss < best_val - 1e-6:
                best_val = val_loss; best_w = self.model.get_weights(); no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience: break

        self.model.set_weights(best_w)


# ---------------------------------------------------------------------------
# Progress logger
# ---------------------------------------------------------------------------

class ProgressLogger(tf.keras.callbacks.Callback):
    def __init__(self, tag, fold, total_folds):
        super().__init__()
        self.tag = tag; self.fold = fold; self.total_folds = total_folds
        self._t0 = None
    def on_train_begin(self, logs=None):
        self._t0 = time.time()
        print(f"  {'Epoch':>6}  {'TrainLoss':>10}  {'Elapsed':>8}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*8}")
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        t = logs.get("loss", float("nan"))
        e = time.time() - self._t0; m, s = divmod(int(e), 60)
        print(f"  {epoch+1:>6}  {t:>10.5f}  {m:02d}m{s:02d}s", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds",         type=int, default=5)
    p.add_argument("--epochs",        type=int, default=40)
    p.add_argument("--patience",      type=int, default=8)
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--skip-ablation", nargs="+", default=[],
                   help="Ablation IDs to skip, e.g. --skip-ablation E F")
    p.add_argument("--force-cnn",     action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed()
    tf.get_logger().setLevel("ERROR")

    print("=" * 70)
    print("Experiment 6: Ablation Study (6 configurations × 5 folds)")
    print("=" * 70)

    # ---- Load data --------------------------------------------------------
    df = pd.read_csv(DATA_CSV).copy()
    if "has_satellite_images" in df.columns:
        df = df[df["has_satellite_images"] == True].reset_index(drop=True)

    ndvi_cols      = get_ndvi_columns(df)
    temp_mean_cols  = get_weather_columns(df, "temp_mean")
    temp_max_cols   = get_weather_columns(df, "temp_max")
    rain_cols       = get_weather_columns(df, "rain")
    all_weather     = temp_mean_cols + temp_max_cols + rain_cols

    df = df.dropna(subset=ndvi_cols + all_weather + ["soil_ph", "year_normalized", "yield_value"]).reset_index(drop=True)
    print(f"  Rows: {len(df):,}")

    images    = load_raw_images(df)
    cnn_feats = get_cnn_features(images)

    y        = df["yield_value"].to_numpy(dtype=np.float32)
    y_scaler = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).reshape(-1).astype(np.float32)

    print("\n  Pre-computing physics labels ...")
    ndvi_mat  = df[ndvi_cols].to_numpy(dtype=np.float32)
    tmean_mat = df[temp_mean_cols].to_numpy(dtype=np.float32)
    tmax_mat  = df[temp_max_cols].to_numpy(dtype=np.float32)
    rain_mat  = df[rain_cols].to_numpy(dtype=np.float32)
    lat_arr   = df["latitude"].to_numpy(dtype=np.float32) if "latitude" in df.columns else None
    crop_names = df["crop"].to_numpy()

    physics_all = compute_all_physics_labels(ndvi_mat, tmean_mat, tmax_mat, rain_mat, crop_names, lat_arr)

    crops           = df["crop"].astype(str)
    stratify_labels = make_stratify_labels(df, min_count=max(2, args.folds))
    num_crops       = int(df["crop_id"].max() + 1)
    num_districts   = int(df["district_id"].max() + 1)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    all_rows: List[dict] = []

    # ---- Run each ablation -----------------------------------------------
    for abl_id, (abl_label, abl_desc) in ABLATION_CONFIGS.items():
        if abl_id in args.skip_ablation:
            print(f"\n[Ablation {abl_id}] SKIPPED: {abl_label}")
            continue

        # For ablation E (no physics), zero all lambdas
        l1 = 0.0 if abl_id == "E" else LAMBDA_GROWTH
        l2 = 0.0 if abl_id == "E" else LAMBDA_TEMP
        l3 = 0.0 if abl_id == "E" else LAMBDA_WATER

        print(f"\n{'='*60}")
        print(f"[Ablation {abl_id}] {abl_label}")
        print(f"  {abl_desc}")
        print(f"{'='*60}")

        oof_pred = np.full(len(df), np.nan, dtype=np.float32)
        abl_rows: List[dict] = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(df, stratify_labels), start=1):
            print(f"\n  Fold {fold}/{args.folds}  (train={len(tr_idx)}, val={len(va_idx)})")
            t_fold = time.time()

            train_df = df.iloc[tr_idx].reset_index(drop=True)
            val_df   = df.iloc[va_idx].reset_index(drop=True)

            (train_in, nsc, wsc, ssc) = prepare_inputs(train_df, cnn_feats[tr_idx], fit=True)
            (val_in, _, _, _)         = prepare_inputs(val_df, cnn_feats[va_idx], nsc, wsc, ssc)

            # Apply ablation (zero out modality)
            train_in_abl = apply_ablation(train_in, abl_id)
            val_in_abl   = apply_ablation(val_in,   abl_id)

            physics_tr = {k: v[tr_idx] for k, v in physics_all.items()}
            physics_va = {k: v[va_idx] for k, v in physics_all.items()}
            y_tr = y_scaled[tr_idx]; y_va = y_scaled[va_idx]

            model   = build_experiment3_fast_model(num_crops=num_crops, num_districts=num_districts)
            trainer = PINNTrainer(model=model, y_scaler=y_scaler, lambda1=l1, lambda2=l2, lambda3=l3)

            trainer.fit(
                train_in_abl, y_tr, physics_tr,
                val_in_abl,   y_va, physics_va,
                epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
            )

            va_tf   = {k: tf.constant(v) for k, v in val_in_abl.items()}
            pred_s  = model.predict(va_tf, verbose=0).reshape(-1, 1)
            pred    = y_scaler.inverse_transform(pred_s).reshape(-1)
            oof_pred[va_idx] = pred

            fold_row = evaluate_row(abl_id, abl_label, f"fold_{fold}", "overall", y[va_idx], pred)
            abl_rows.append(fold_row)
            elapsed = time.time() - t_fold
            print(f"  R2={fold_row['r2']:.4f} | MAE={fold_row['mae']:.4f}  ({elapsed:.0f}s)")

        # OOF for this ablation
        oof_row = evaluate_row(abl_id, abl_label, "oof", "overall", y, oof_pred)
        abl_rows.append(oof_row)
        for crop in sorted(crops.unique()):
            mask = (crops == crop).to_numpy()
            abl_rows.append(evaluate_row(abl_id, abl_label, "oof", crop, y[mask], oof_pred[mask]))

        oof_r2 = oof_row["r2"]
        print(f"\n  [{abl_id}] {abl_label}  OOF R2={oof_r2:.4f} | MAE={oof_row['mae']:.4f}")
        all_rows.extend(abl_rows)

    # ---- Save results ----------------------------------------------------
    results_df = pd.DataFrame(all_rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n  Ablation results saved -> {RESULTS_PATH}")

    # ---- Summary table ---------------------------------------------------
    print("\n" + "="*60)
    print("ABLATION SUMMARY (OOF R² — lower = modality was important)")
    print("="*60)

    # Load full model R² from Exp4 for comparison
    exp4_path = PROJECT_ROOT / "results" / "tables" / "model_metrics_exp4.csv"
    full_r2 = None
    if exp4_path.exists():
        tmp = pd.read_csv(exp4_path)
        row = tmp[(tmp["split"] == "oof") & (tmp["crop"] == "overall")]
        if not row.empty:
            full_r2 = row.iloc[0]["r2"]
            print(f"  {'Full PINN (Exp4)':35s}  R2={full_r2:.4f} (baseline)")

    oof_summary = results_df[(results_df["split"] == "oof") & (results_df["crop"] == "overall")]
    for _, r in oof_summary.iterrows():
        delta = f"  Δ={r['r2'] - full_r2:+.4f}" if full_r2 is not None else ""
        print(f"  {r['ablation_id']}. {r['ablation_label']:33s}  R2={r['r2']:.4f}{delta}")


if __name__ == "__main__":
    main()
