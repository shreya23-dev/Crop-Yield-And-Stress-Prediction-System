"""
Experiment 5 training: PINN + Multi-task (Yield + Stress prediction).

Extends Exp4 by adding a stress prediction head and L_stress loss term.

L_total = L_yield + λ1*L_growth + λ2*L_temp + λ3*L_water + λ4*L_stress

Pre-computed CNN features and fold checkpointing are included for speed/resilience.

Usage:
    python src/training/train_experiment5.py
    python src/training/train_experiment5.py --epochs 60 --skip-final-train --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.multimodal_net_exp5 import build_experiment5_model
from src.models.multimodal_net_exp3 import build_satellite_cnn
from src.models.physics_loss import compute_all_physics_labels

# ---- paths ----------------------------------------------------------------
DATA_CSV       = PROJECT_ROOT / "data" / "processed" / "final_dataset.csv"
SAT_DIR        = PROJECT_ROOT / "data" / "processed" / "satellite_images"
RESULTS_PATH   = PROJECT_ROOT / "results" / "tables" / "model_metrics_exp5.csv"
CKPT_PATH      = PROJECT_ROOT / "results" / "tables" / "exp5_checkpoint.json"
SAT_FEAT_CACHE = PROJECT_ROOT / "data" / "processed" / "satellite_cnn_features.npy"
MODEL_PATH     = PROJECT_ROOT / "api" / "models" / "experiment5_pinn_multitask.keras"
PREPROC_PATH   = PROJECT_ROOT / "api" / "models" / "experiment5_preprocessing.pkl"

SAT_MONTHS = [6, 7, 8, 9, 10, 11]
IMAGE_H    = 32
IMAGE_W    = 32

# Lambda hyperparameters
LAMBDA_GROWTH  = 0.10
LAMBDA_TEMP    = 0.05
LAMBDA_WATER   = 0.05
LAMBDA_STRESS  = 0.10   # weight for stress prediction task

SEED = 42


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_stratify_labels(df: pd.DataFrame, min_count: int = 5) -> pd.Series:
    combo  = df["crop"].astype(str) + "__" + df["district"].astype(str)
    counts = combo.value_counts()
    return combo.where(combo.map(counts) >= min_count, "__RARE__")


def get_ndvi_columns(df: pd.DataFrame) -> List[str]:
    return ["ndvi_jun", "ndvi_jul", "ndvi_aug", "ndvi_sep", "ndvi_oct", "ndvi_nov"]


def get_weather_columns(df: pd.DataFrame, var: str) -> List[str]:
    return [f"week_{w}_{var}" for w in range(1, 23)]


def evaluate_row(model_name, split, crop, y_true, y_pred) -> dict:
    return {
        "model": model_name, "split": split, "crop": crop,
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Satellite images + CNN features
# ---------------------------------------------------------------------------

def load_raw_images(df: pd.DataFrame) -> np.ndarray:
    N      = len(df)
    images = np.zeros((N, 6, IMAGE_H, IMAGE_W, 1), dtype=np.float32)
    loaded = 0
    print(f"  Loading satellite images for {N} rows ...")
    for i, (_, row) in enumerate(df.iterrows()):
        district, year = row["district"], int(row["year"])
        month_arrs, ok = [], True
        for month in SAT_MONTHS:
            fpath = SAT_DIR / f"{district}_{year}_{month:02d}.npy"
            if not fpath.exists():
                ok = False; break
            arr = np.load(fpath).astype(np.float32)
            if arr.ndim == 3:     arr = arr[:, :, 0]
            if arr.shape != (IMAGE_H, IMAGE_W):
                arr = tf.image.resize(arr[..., np.newaxis], [IMAGE_H, IMAGE_W]).numpy()[..., 0]
            month_arrs.append(arr)
        if ok:
            for m, arr in enumerate(month_arrs):
                images[i, m, :, :, 0] = arr
            loaded += 1
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{N} processed ({loaded} with images)", flush=True)
    print(f"  Done. {loaded}/{N} rows with full satellite coverage.")
    return images


def extract_cnn_features(images: np.ndarray, force: bool = False) -> np.ndarray:
    """Load from cache (shared with Exp3/Exp4) or recompute."""
    if SAT_FEAT_CACHE.exists() and not force:
        print(f"  Loading CNN feature cache ...")
        feats = np.load(SAT_FEAT_CACHE)
        if feats.shape[0] == len(images):
            print(f"  Cache hit! Shape: {feats.shape}")
            return feats
    print(f"  Extracting CNN features ...")
    t0  = time.time()
    cnn = build_satellite_cnn(IMAGE_H, IMAGE_W, image_channels=1)
    N   = len(images)
    feats = np.zeros((N, 6, 128), dtype=np.float32)
    sat_min   = images.min()
    sat_range = max(images.max() - sat_min, 1e-6)
    imgs_norm = np.clip((images - sat_min) / sat_range, 0.0, 1.0)
    for m in range(6):
        feats[:, m, :] = cnn.predict(imgs_norm[:, m, :, :, :], batch_size=64, verbose=0)
        print(f"    Month {m+1}/6 done  ({time.time()-t0:.1f}s)", flush=True)
    np.save(SAT_FEAT_CACHE, feats)
    return feats


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def prepare_inputs(df, cnn_feats, ndvi_sc=None, weather_sc=None, soil_sc=None, fit=False):
    ndvi_cols      = get_ndvi_columns(df)
    weather_cols   = (get_weather_columns(df, "temp_mean") +
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


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class ProgressLogger(tf.keras.callbacks.Callback):
    def __init__(self, fold, total_folds):
        super().__init__()
        self.fold = fold; self.total_folds = total_folds
        self._t0 = None
    def on_train_begin(self, logs=None):
        self._t0 = time.time()
        print(f"  {'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'Elapsed':>8}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        t = logs.get("loss", float("nan")); v = logs.get("val_loss", float("nan"))
        e = time.time() - self._t0; m, s = divmod(int(e), 60)
        print(f"  {epoch+1:>6}  {t:>10.5f}  {v:>10.5f}  {m:02d}m{s:02d}s", flush=True)
    def on_train_end(self, logs=None):
        e = time.time() - self._t0; m, s = divmod(int(e), 60)
        print(f"  --- Fold {self.fold}/{self.total_folds} done in {m}m {s}s ---")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict:
    if CKPT_PATH.exists():
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {"completed_folds": [], "rows": []}


def save_checkpoint(ckpt: dict) -> None:
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CKPT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)


# ---------------------------------------------------------------------------
# PINN Multi-task Trainer
# ---------------------------------------------------------------------------

class PINNMultiTaskTrainer:
    """Custom GradientTape loop for multi-task PINN loss."""

    def __init__(self, model, y_scaler, lambda1=LAMBDA_GROWTH, lambda2=LAMBDA_TEMP,
                 lambda3=LAMBDA_WATER, lambda4=LAMBDA_STRESS):
        self.model    = model
        self.y_scaler = y_scaler
        self.l1 = lambda1; self.l2 = lambda2
        self.l3 = lambda3; self.l4 = lambda4

    def combined_loss(self, y_true_s, y_pred_s, stress_true, stress_pred,
                      gp, tp, wp):
        """
        L_total = L_yield + λ1*L_growth + λ2*L_temp + λ3*L_water + λ4*L_stress
        """
        l_yield = tf.reduce_mean(tf.square(y_pred_s - y_true_s))

        y_min = tf.reduce_min(y_pred_s)
        y_range = tf.maximum(tf.reduce_max(y_pred_s) - y_min, 1e-6)
        y_norm = (y_pred_s - y_min) / y_range

        l_growth = tf.reduce_mean(tf.square(y_norm - gp))
        l_temp   = tf.reduce_mean(tf.square(y_norm - tp))
        l_water  = tf.reduce_mean(tf.square(y_norm - wp))
        l_stress = tf.reduce_mean(tf.square(stress_pred - stress_true))

        return l_yield + self.l1*l_growth + self.l2*l_temp + self.l3*l_water + self.l4*l_stress

    def fit(self, train_inputs, y_train, stress_train, physics_train,
            val_inputs, y_val, stress_val, physics_val,
            epochs, batch_size, patience, callbacks_extra=None):

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        N = len(y_train)
        idx = np.arange(N)

        def _t(a): return tf.constant(a.astype(np.float32))

        gp_tr = _t(physics_train["growth_proxy"])
        tp_tr = _t(1.0 - physics_train["thermal_stress"])
        wp_tr = _t(1.0 - physics_train["water_stress"])
        gp_va = _t(physics_val["growth_proxy"])
        tp_va = _t(1.0 - physics_val["thermal_stress"])
        wp_va = _t(1.0 - physics_val["water_stress"])

        y_tr_tf  = _t(y_train);  y_va_tf  = _t(y_val)
        st_tr_tf = _t(stress_train); st_va_tf = _t(stress_val)

        train_tf = {k: tf.constant(v) for k, v in train_inputs.items()}
        val_tf   = {k: tf.constant(v) for k, v in val_inputs.items()}

        best_val = float("inf")
        best_w   = self.model.get_weights()
        no_imp   = 0
        history  = []

        if callbacks_extra:
            for cb in callbacks_extra: cb.on_train_begin()

        for epoch in range(epochs):
            np.random.shuffle(idx)
            epoch_losses = []

            for s in range(0, N, batch_size):
                e    = min(s + batch_size, N)
                bidx = idx[s:e]
                bi   = {k: tf.gather(v, bidx) for k, v in train_tf.items()}
                y_b  = tf.gather(y_tr_tf, bidx)
                st_b = tf.gather(st_tr_tf, bidx)
                gp_b = tf.gather(gp_tr, bidx)
                tp_b = tf.gather(tp_tr, bidx)
                wp_b = tf.gather(wp_tr, bidx)

                with tf.GradientTape() as tape:
                    out    = self.model(bi, training=True)
                    y_pred = tf.squeeze(out["yield"], axis=1)
                    s_pred = tf.squeeze(out["stress"], axis=1)
                    loss   = self.combined_loss(y_b, y_pred, st_b, s_pred, gp_b, tp_b, wp_b)

                grads = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                epoch_losses.append(float(loss.numpy()))

            tr_loss = float(np.mean(epoch_losses))

            # Validation
            val_out   = self.model(val_tf, training=False)
            y_vp      = tf.squeeze(val_out["yield"],  axis=1)
            s_vp      = tf.squeeze(val_out["stress"], axis=1)
            val_loss  = float(self.combined_loss(y_va_tf, y_vp, st_va_tf, s_vp, gp_va, tp_va, wp_va).numpy())

            history.append((tr_loss, val_loss))

            if callbacks_extra:
                for cb in callbacks_extra:
                    cb.on_epoch_end(epoch, logs={"loss": tr_loss, "val_loss": val_loss})

            if val_loss < best_val - 1e-6:
                best_val = val_loss; best_w = self.model.get_weights(); no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience: break

        self.model.set_weights(best_w)
        if callbacks_extra:
            for cb in callbacks_extra: cb.on_train_end()
        return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds",      type=int,   default=5)
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lambda1",    type=float, default=LAMBDA_GROWTH)
    p.add_argument("--lambda2",    type=float, default=LAMBDA_TEMP)
    p.add_argument("--lambda3",    type=float, default=LAMBDA_WATER)
    p.add_argument("--lambda4",    type=float, default=LAMBDA_STRESS)
    p.add_argument("--skip-final-train", action="store_true")
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--force-cnn",  action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed()
    tf.get_logger().setLevel("ERROR")

    print("=" * 70)
    print("Experiment 5: PINN Multi-task (Yield + Stress)")
    print(f"  λ1={args.lambda1} λ2={args.lambda2} λ3={args.lambda3} λ4={args.lambda4}")
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

    # ---- Satellite CNN features ------------------------------------------
    images    = load_raw_images(df)
    cnn_feats = extract_cnn_features(images, force=args.force_cnn)

    # ---- Targets ----------------------------------------------------------
    y        = df["yield_value"].to_numpy(dtype=np.float32)
    y_scaler = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).reshape(-1).astype(np.float32)

    # ---- Physics labels + stress target ----------------------------------
    print("\n  Pre-computing physics labels + stress target ...")
    ndvi_mat  = df[ndvi_cols].to_numpy(dtype=np.float32)
    tmean_mat = df[temp_mean_cols].to_numpy(dtype=np.float32)
    tmax_mat  = df[temp_max_cols].to_numpy(dtype=np.float32)
    rain_mat  = df[rain_cols].to_numpy(dtype=np.float32)
    lat_arr   = df["latitude"].to_numpy(dtype=np.float32) if "latitude" in df.columns else None
    crop_names = df["crop"].to_numpy()

    physics_all = compute_all_physics_labels(ndvi_mat, tmean_mat, tmax_mat, rain_mat, crop_names, lat_arr)
    stress_all  = physics_all["combined_stress"]   # shape (N,) in [0,1]

    print(f"  combined_stress: mean={stress_all.mean():.3f}  std={stress_all.std():.3f}")

    crops           = df["crop"].astype(str)
    stratify_labels = make_stratify_labels(df, min_count=max(2, args.folds))
    num_crops       = int(df["crop_id"].max() + 1)
    num_districts   = int(df["district_id"].max() + 1)

    # ---- Checkpointing ---------------------------------------------------
    ckpt = load_checkpoint() if args.resume else {"completed_folds": [], "rows": []}
    completed_folds: List[int] = ckpt["completed_folds"]
    rows: List[dict]           = ckpt.get("rows", [])
    if completed_folds:
        print(f"\n  Resuming from folds done: {completed_folds}")

    # ---- 5-Fold CV -------------------------------------------------------
    skf      = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    oof_pred = np.full(len(df), np.nan, dtype=np.float32)
    oof_stress_pred = np.full(len(df), np.nan, dtype=np.float32)
    rows     = rows if rows else []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, stratify_labels), start=1):
        if fold in completed_folds:
            print(f"\nFold {fold}/{args.folds} -- SKIPPED (checkpoint)")
            continue

        print(f"\nFold {fold}/{args.folds}  (train={len(tr_idx)}, val={len(va_idx)})")

        train_df  = df.iloc[tr_idx].reset_index(drop=True)
        val_df    = df.iloc[va_idx].reset_index(drop=True)

        (train_in, ndvi_sc, weather_sc, soil_sc) = prepare_inputs(train_df, cnn_feats[tr_idx], fit=True)
        (val_in, _, _, _)                         = prepare_inputs(
            val_df, cnn_feats[va_idx], ndvi_sc, weather_sc, soil_sc
        )

        physics_tr = {k: v[tr_idx] for k, v in physics_all.items()}
        physics_va = {k: v[va_idx] for k, v in physics_all.items()}

        y_tr = y_scaled[tr_idx]; y_va = y_scaled[va_idx]
        st_tr = stress_all[tr_idx]; st_va = stress_all[va_idx]

        model   = build_experiment5_model(num_crops=num_crops, num_districts=num_districts)
        trainer = PINNMultiTaskTrainer(
            model=model, y_scaler=y_scaler,
            lambda1=args.lambda1, lambda2=args.lambda2,
            lambda3=args.lambda3, lambda4=args.lambda4,
        )
        trainer.fit(
            train_inputs=train_in, y_train=y_tr, stress_train=st_tr, physics_train=physics_tr,
            val_inputs=val_in,   y_val=y_va,   stress_val=st_va,   physics_val=physics_va,
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
            callbacks_extra=[ProgressLogger(fold=fold, total_folds=args.folds)],
        )

        # Evaluate
        val_tf  = {k: tf.constant(v) for k, v in val_in.items()}
        val_out = model(val_tf, training=False)
        pred_s  = tf.squeeze(val_out["yield"],  axis=1).numpy().reshape(-1, 1)
        pred_st = tf.squeeze(val_out["stress"], axis=1).numpy()

        pred    = y_scaler.inverse_transform(pred_s).reshape(-1)
        oof_pred[va_idx]        = pred
        oof_stress_pred[va_idx] = pred_st

        fold_row = evaluate_row("experiment5_pinn_multitask", f"fold_{fold}", "overall", y[va_idx], pred)
        fold_row["stress_mae"] = float(mean_absolute_error(st_va, pred_st))
        fold_row["stress_r2"]  = float(r2_score(st_va, pred_st))
        rows.append(fold_row)
        print(f"  Yield  R2={fold_row['r2']:.4f} | MAE={fold_row['mae']:.4f}")
        print(f"  Stress R2={fold_row['stress_r2']:.4f} | MAE={fold_row['stress_mae']:.4f}")

        completed_folds.append(fold)
        ckpt["completed_folds"] = completed_folds
        ckpt["rows"]            = rows
        save_checkpoint(ckpt)
        print(f"  Checkpoint saved (folds done: {completed_folds})")

    # ---- OOF metrics -----------------------------------------------------
    if not np.isnan(oof_pred).any():
        rows.append(evaluate_row("experiment5_pinn_multitask", "oof", "overall", y, oof_pred))
        for crop in sorted(crops.unique()):
            mask = (crops == crop).to_numpy()
            rows.append(evaluate_row("experiment5_pinn_multitask", "oof", crop, y[mask], oof_pred[mask]))

        results_df = pd.DataFrame(rows)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(RESULTS_PATH, index=False)
        stress_mae_oof = float(mean_absolute_error(stress_all, oof_stress_pred))
        stress_r2_oof  = float(r2_score(stress_all, oof_stress_pred))
    else:
        results_df = pd.read_csv(RESULTS_PATH)
        stress_mae_oof = 0.0
        stress_r2_oof = 0.0

    oof_row = results_df[(results_df["split"] == "oof") & (results_df["crop"] == "overall")].iloc[0]

    print(f"\nOOF Yield  -> R2={oof_row['r2']:.4f} | MAE={oof_row['mae']:.4f} | RMSE={oof_row['rmse']:.4f}")
    print(f"OOF Stress -> R2={stress_r2_oof:.4f} | MAE={stress_mae_oof:.4f}")

    # Experiment comparison
    print("\n  Experiment comparison:")
    exp_files = [
        ("Exp1 Tabular Baseline", "model_metrics.csv",     "xgboost_best"),
        ("Exp2 Neural (no imgs)", "model_metrics_exp2.csv", "experiment2_neural"),
        ("Exp3 Neural + Images",  "model_metrics_exp3.csv", "experiment3_with_images"),
        ("Exp4 PINN",             "model_metrics_exp4.csv", "experiment4_pinn"),
    ]
    for label, fname, _ in exp_files:
        fpath = PROJECT_ROOT / "results" / "tables" / fname
        if fpath.exists():
            tmp = pd.read_csv(fpath)
            row = tmp[(tmp["split"] == "oof") & (tmp["crop"] == "overall")]
            if not row.empty:
                r = row.iloc[0]
                print(f"    {label:30s}  R2={r['r2']:.4f}  MAE={r['mae']:.4f}")
    print(f"    {'Exp5 PINN+Stress':30s}  R2={oof_row['r2']:.4f}  MAE={oof_row['mae']:.4f}  <-- current")

    if CKPT_PATH.exists():
        CKPT_PATH.unlink()
    print(f"\n  Results saved -> {RESULTS_PATH}")

    # ---- Save final model ------------------------------------------------
    if not args.skip_final_train:
        print("\n--- Final full-data training ---")
        (full_in, ndvi_sc, weather_sc, soil_sc) = prepare_inputs(df, cnn_feats, fit=True)
        final_model   = build_experiment5_model(num_crops=num_crops, num_districts=num_districts)
        final_trainer = PINNMultiTaskTrainer(
            model=final_model, y_scaler=y_scaler,
            lambda1=args.lambda1, lambda2=args.lambda2,
            lambda3=args.lambda3, lambda4=args.lambda4,
        )
        final_trainer.fit(
            train_inputs=full_in, y_train=y_scaled, stress_train=stress_all, physics_train=physics_all,
            val_inputs=full_in,   y_val=y_scaled,   stress_val=stress_all,   physics_val=physics_all,
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
        )
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_model.save(MODEL_PATH)
        joblib.dump(
            {"ndvi_scaler": ndvi_sc, "weather_scaler": weather_sc, "soil_scaler": soil_sc,
             "y_scaler": y_scaler, "num_crops": num_crops, "num_districts": num_districts},
            PREPROC_PATH,
        )
        print(f"  Model saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
