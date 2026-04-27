"""
Experiment 4: PINN — Full 4-branch model WITH physics-informed loss.

Adds three physics regularization terms on top of Experiment 3:
  L_total = L_yield + lambda1*L_growth + lambda2*L_temperature + lambda3*L_water

Architecture is identical to Exp3. Only the loss changes.
Physics labels are pre-computed from NDVI + weather data before training starts.

Usage:
    python src/training/train_experiment4.py
    python src/training/train_experiment4.py --epochs 60 --skip-final-train
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

from src.models.multimodal_net_exp3 import (
    build_satellite_cnn,
    build_experiment3_fast_model,   # fast model takes (N, 6, 128) CNN features
)
from src.models.physics_loss import compute_all_physics_labels

# ---- paths ----------------------------------------------------------------
DATA_CSV     = PROJECT_ROOT / "data" / "processed" / "final_dataset.csv"
SAT_DIR      = PROJECT_ROOT / "data" / "processed" / "satellite_images"
RESULTS_PATH = PROJECT_ROOT / "results" / "tables" / "model_metrics_exp4.csv"
CKPT_PATH    = PROJECT_ROOT / "results" / "tables" / "exp4_checkpoint.json"
SAT_FEAT_CACHE = PROJECT_ROOT / "data" / "processed" / "satellite_cnn_features.npy"
MODEL_PATH   = PROJECT_ROOT / "api" / "models" / "experiment4_pinn.keras"
PREPROC_PATH = PROJECT_ROOT / "api" / "models" / "experiment4_preprocessing.pkl"

# Satellite image constants
SAT_MONTHS  = [6, 7, 8, 9, 10, 11]
IMAGE_H     = 32
IMAGE_W     = 32

# Physics lambda hyperparameters (from Section 4.7 of project spec)
LAMBDA_GROWTH = 0.10
LAMBDA_TEMP   = 0.05
LAMBDA_WATER  = 0.05

SEED = 42


# ---------------------------------------------------------------------------
# Utilities (same as Exp3)
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_stratify_labels(df: pd.DataFrame, min_count: int = 5) -> pd.Series:
    combo  = df["crop"].astype(str) + "__" + df["district"].astype(str)
    counts = combo.value_counts()
    return combo.where(combo.map(counts) >= min_count, "__RARE__")


def get_ndvi_columns(df: pd.DataFrame) -> List[str]:
    cols = ["ndvi_jun", "ndvi_jul", "ndvi_aug", "ndvi_sep", "ndvi_oct", "ndvi_nov"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing NDVI columns: {missing}")
    return cols


def get_weather_columns(df: pd.DataFrame, var: str) -> List[str]:
    """Get all 22 weekly columns for a given weather variable (temp_mean, temp_max, rain)."""
    return [f"week_{w}_{var}" for w in range(1, 23)]


def load_satellite_images(df: pd.DataFrame) -> np.ndarray:
    """Load 32x32 satellite images. Returns zeros for missing."""
    N = len(df)
    images = np.zeros((N, 6, IMAGE_H, IMAGE_W, 1), dtype=np.float32)
    loaded = 0

    print(f"  Loading satellite images ({N} rows)...")
    for i, (_, row) in enumerate(df.iterrows()):
        district, year = row["district"], int(row["year"])
        month_arrays, all_found = [], True

        for month in SAT_MONTHS:
            fpath = SAT_DIR / f"{district}_{year}_{month:02d}.npy"
            if not fpath.exists():
                all_found = False
                break
            arr = np.load(fpath).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            if arr.shape != (IMAGE_H, IMAGE_W):
                arr = tf.image.resize(arr[..., np.newaxis], [IMAGE_H, IMAGE_W]).numpy()[..., 0]
            month_arrays.append(arr)

        if all_found:
            for m, arr in enumerate(month_arrays):
                images[i, m, :, :, 0] = arr
            loaded += 1

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{N}  ({loaded} with images)", flush=True)

    print(f"  Done. {loaded}/{N} rows have complete satellite coverage.")
    return images


def extract_cnn_features(images: np.ndarray, force: bool = False) -> np.ndarray:
    """
    Pre-extract CNN features once for all images.
    Reuses the cache from Exp3 if available (same CNN architecture).
    """
    if SAT_FEAT_CACHE.exists() and not force:
        print(f"  Loading cached CNN features from {SAT_FEAT_CACHE.name} ...")
        features = np.load(SAT_FEAT_CACHE)
        if features.shape[0] == len(images):
            print(f"  Cache hit! Shape: {features.shape}")
            return features
        print("  Cache shape mismatch, recomputing...")

    print(f"  Extracting CNN features for {len(images)} rows x 6 months ...")
    t0  = time.time()
    cnn = build_satellite_cnn(IMAGE_H, IMAGE_W, image_channels=1)

    N        = len(images)
    features = np.zeros((N, 6, 128), dtype=np.float32)
    sat_min   = images.min()
    sat_range = max(images.max() - sat_min, 1e-6)
    images_norm = np.clip((images - sat_min) / sat_range, 0.0, 1.0)

    for m in range(6):
        feats = cnn.predict(images_norm[:, m, :, :, :], batch_size=64, verbose=0)
        features[:, m, :] = feats
        print(f"    Month {m+1}/6 done  ({time.time()-t0:.1f}s)", flush=True)

    np.save(SAT_FEAT_CACHE, features)
    print(f"  CNN features cached -> {SAT_FEAT_CACHE}")
    return features


def prepare_tabular_inputs(
    df: pd.DataFrame,
    cnn_feats: np.ndarray,
    ndvi_scaler=None, weather_scaler=None, soil_scaler=None,
    fit: bool = False,
):
    ndvi_cols    = get_ndvi_columns(df)
    weather_cols = (
        get_weather_columns(df, "temp_mean") +
        get_weather_columns(df, "temp_max") +
        get_weather_columns(df, "rain")
    )
    ndvi        = df[ndvi_cols].to_numpy(dtype=np.float32)
    weather_flat = df[weather_cols].to_numpy(dtype=np.float32)
    soil        = df[["soil_ph"]].to_numpy(dtype=np.float32)
    year        = df[["year_normalized"]].to_numpy(dtype=np.float32)
    crop_id     = df[["crop_id"]].to_numpy(dtype=np.int32)
    district_id = df[["district_id"]].to_numpy(dtype=np.int32)

    if fit or ndvi_scaler is None:
        ndvi_scaler    = StandardScaler().fit(ndvi)
    if fit or weather_scaler is None:
        weather_scaler = StandardScaler().fit(weather_flat)
    if fit or soil_scaler is None:
        soil_scaler    = StandardScaler().fit(soil)

    ndvi_scaled    = ndvi_scaler.transform(ndvi).reshape(-1, 6, 1).astype(np.float32)
    weather_scaled = weather_scaler.transform(weather_flat).reshape(-1, 22, 3).astype(np.float32)
    soil_scaled    = soil_scaler.transform(soil).astype(np.float32)

    return (
        ndvi_scaled, weather_scaled, soil_scaled, year, crop_id, district_id,
        ndvi_scaler, weather_scaler, soil_scaler
    )


def evaluate_rows(model_name, split, crop, y_true, y_pred) -> dict:
    return {
        "model": model_name,
        "split": split,
        "crop":  crop,
        "r2":    r2_score(y_true, y_pred),
        "mae":   mean_absolute_error(y_true, y_pred),
        "rmse":  rmse(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Per-epoch progress callback
# ---------------------------------------------------------------------------

class ProgressLogger(tf.keras.callbacks.Callback):
    def __init__(self, fold: int, total_folds: int):
        super().__init__()
        self.fold = fold
        self.total_folds = total_folds
        self._t0 = None

    def on_train_begin(self, logs=None):
        self._t0 = time.time()
        print(f"  {'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'Elapsed':>8}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")

    def on_epoch_end(self, epoch, logs=None):
        logs   = logs or {}
        t_loss = logs.get("loss", float("nan"))
        v_loss = logs.get("val_loss", float("nan"))
        elapsed = time.time() - self._t0
        mins, secs = divmod(int(elapsed), 60)
        print(f"  {epoch+1:>6}  {t_loss:>10.5f}  {v_loss:>10.5f}  {mins:02d}m{secs:02d}s", flush=True)

    def on_train_end(self, logs=None):
        total = time.time() - self._t0
        mins, secs = divmod(int(total), 60)
        print(f"  --- Fold {self.fold}/{self.total_folds} done in {mins}m {secs}s ---")


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
# PINN Custom Training Loop
# ---------------------------------------------------------------------------

class PINNTrainer:
    """
    Custom training loop that adds physics losses on top of MSE.

    Because physics losses are computed from NDVI/weather features (not
    from the Keras tensors), we inject them as sample-level labels and use
    Keras's loss API with a custom combined_loss function.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        y_scaler: StandardScaler,
        lambda1: float = LAMBDA_GROWTH,
        lambda2: float = LAMBDA_TEMP,
        lambda3: float = LAMBDA_WATER,
    ):
        self.model    = model
        self.y_scaler = y_scaler
        self.lambda1  = lambda1
        self.lambda2  = lambda2
        self.lambda3  = lambda3

    def pinn_loss(
        self,
        y_true_scaled:   tf.Tensor,   # (N,)
        y_pred_scaled:   tf.Tensor,   # (N,)
        growth_proxy:    tf.Tensor,   # (N,)
        thermal_proxy:   tf.Tensor,   # (N,)  = 1 - thermal_stress
        water_proxy:     tf.Tensor,   # (N,)  = 1 - water_stress
    ) -> tf.Tensor:
        """Combined PINN loss."""
        # Yield MSE (in scaled space)
        l_yield = tf.reduce_mean(tf.square(y_pred_scaled - y_true_scaled))

        # Normalize predictions to [0,1] for physics terms
        y_min = tf.reduce_min(y_pred_scaled)
        y_max = tf.reduce_max(y_pred_scaled)
        y_range = tf.maximum(y_max - y_min, 1e-6)
        y_pred_norm = (y_pred_scaled - y_min) / y_range

        l_growth = tf.reduce_mean(tf.square(y_pred_norm - growth_proxy))
        l_temp   = tf.reduce_mean(tf.square(y_pred_norm - thermal_proxy))
        l_water  = tf.reduce_mean(tf.square(y_pred_norm - water_proxy))

        return l_yield + self.lambda1 * l_growth + self.lambda2 * l_temp + self.lambda3 * l_water

    def fit(
        self,
        train_inputs: Dict,
        y_train: np.ndarray,
        physics_train: dict,
        val_inputs: Dict,
        y_val: np.ndarray,
        physics_val: dict,
        epochs: int,
        batch_size: int,
        patience: int,
        callbacks_extra: list = None,
    ) -> list:
        """
        Manual mini-batch training loop with PINN loss.

        Returns list of (train_loss, val_loss) per epoch.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
        )

        N_train = len(y_train)
        indices = np.arange(N_train)

        # Convert physics labels to tensors
        def _to_tensor(arr):
            return tf.constant(arr.astype(np.float32))

        gp_train  = _to_tensor(physics_train["growth_proxy"])
        tp_train  = _to_tensor(1.0 - physics_train["thermal_stress"])
        wp_train  = _to_tensor(1.0 - physics_train["water_stress"])
        gp_val    = _to_tensor(physics_val["growth_proxy"])
        tp_val    = _to_tensor(1.0 - physics_val["thermal_stress"])
        wp_val    = _to_tensor(1.0 - physics_val["water_stress"])

        y_train_tf = tf.constant(y_train.astype(np.float32))
        y_val_tf   = tf.constant(y_val.astype(np.float32))

        # Pre-build full input tensors
        def _dict_to_tf(d):
            return {k: tf.constant(v) for k, v in d.items()}

        train_inputs_tf = _dict_to_tf(train_inputs)
        val_inputs_tf   = _dict_to_tf(val_inputs)

        best_val_loss   = float("inf")
        best_weights    = self.model.get_weights()
        no_improve      = 0
        history         = []

        # Print header via callback if provided
        if callbacks_extra:
            for cb in callbacks_extra:
                cb.on_train_begin()

        for epoch in range(epochs):
            # Shuffle
            np.random.shuffle(indices)

            epoch_losses = []
            for start in range(0, N_train, batch_size):
                end  = min(start + batch_size, N_train)
                bidx = indices[start:end]

                batch_inputs = {k: tf.gather(v, bidx) for k, v in train_inputs_tf.items()}
                y_b    = tf.gather(y_train_tf, bidx)
                gp_b   = tf.gather(gp_train, bidx)
                tp_b   = tf.gather(tp_train, bidx)
                wp_b   = tf.gather(wp_train, bidx)

                with tf.GradientTape() as tape:
                    pred    = self.model(batch_inputs, training=True)
                    pred    = tf.squeeze(pred, axis=1)
                    loss_val = self.pinn_loss(y_b, pred, gp_b, tp_b, wp_b)

                grads = tape.gradient(loss_val, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                epoch_losses.append(float(loss_val.numpy()))

            train_loss = float(np.mean(epoch_losses))

            # Validation
            val_pred = tf.squeeze(self.model(val_inputs_tf, training=False), axis=1)
            val_loss = float(self.pinn_loss(y_val_tf, val_pred, gp_val, tp_val, wp_val).numpy())

            history.append((train_loss, val_loss))

            # Log via callbacks
            if callbacks_extra:
                for cb in callbacks_extra:
                    cb.on_epoch_end(epoch, logs={"loss": train_loss, "val_loss": val_loss})

            # Early stopping + best weights
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_weights  = self.model.get_weights()
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        self.model.set_weights(best_weights)

        if callbacks_extra:
            for cb in callbacks_extra:
                cb.on_train_end()

        return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Experiment 4: PINN with physics loss.")
    parser.add_argument("--folds",        type=int,   default=5,    help="CV folds")
    parser.add_argument("--epochs",       type=int,   default=60,   help="Max epochs")
    parser.add_argument("--patience",     type=int,   default=10,   help="Early stopping patience")
    parser.add_argument("--batch-size",   type=int,   default=32,   help="Batch size")
    parser.add_argument("--lambda1",      type=float, default=LAMBDA_GROWTH, help="Growth lambda")
    parser.add_argument("--lambda2",      type=float, default=LAMBDA_TEMP,   help="Temp lambda")
    parser.add_argument("--lambda3",      type=float, default=LAMBDA_WATER,  help="Water lambda")
    parser.add_argument("--skip-final-train", action="store_true")
    parser.add_argument("--resume",       action="store_true", help="Resume from fold checkpoint")
    parser.add_argument("--force-cnn",    action="store_true", help="Force re-extract CNN features")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed()
    tf.get_logger().setLevel("ERROR")

    print("=" * 70)
    print("Experiment 4: PINN — Multimodal + Physics Loss")
    print(f"  lambda1(growth)={args.lambda1}  lambda2(temp)={args.lambda2}  lambda3(water)={args.lambda3}")
    print("=" * 70)

    # ---- Load data --------------------------------------------------------
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV).copy()
    if "has_satellite_images" in df.columns:
        df = df[df["has_satellite_images"] == True].reset_index(drop=True)

    ndvi_cols        = get_ndvi_columns(df)
    temp_mean_cols   = get_weather_columns(df, "temp_mean")
    temp_max_cols    = get_weather_columns(df, "temp_max")
    rain_cols        = get_weather_columns(df, "rain")
    weather_all_cols = temp_mean_cols + temp_max_cols + rain_cols

    df = df.dropna(subset=ndvi_cols + weather_all_cols + ["soil_ph", "year_normalized", "yield_value"]).reset_index(drop=True)
    print(f"  Using {len(df):,} rows after filtering.")

    # ---- Load images + extract CNN features (ONCE, cached) ---------------
    images    = load_satellite_images(df)
    cnn_feats = extract_cnn_features(images, force=args.force_cnn)

    # ---- Targets ----------------------------------------------------------
    y        = df["yield_value"].to_numpy(dtype=np.float32)
    y_scaler = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).reshape(-1).astype(np.float32)

    # ---- Pre-compute physics labels for ALL rows -------------------------
    print("\n  Pre-computing physics labels ...")
    ndvi_matrix      = df[ndvi_cols].to_numpy(dtype=np.float32)
    temp_mean_matrix = df[temp_mean_cols].to_numpy(dtype=np.float32)
    temp_max_matrix  = df[temp_max_cols].to_numpy(dtype=np.float32)
    rain_matrix      = df[rain_cols].to_numpy(dtype=np.float32)
    lat_array        = df["latitude"].to_numpy(dtype=np.float32) if "latitude" in df.columns else None
    crop_names       = df["crop"].to_numpy()

    physics_all = compute_all_physics_labels(
        ndvi_matrix, temp_mean_matrix, temp_max_matrix, rain_matrix, crop_names, lat_array
    )
    print(f"  growth_proxy  : mean={physics_all['growth_proxy'].mean():.3f}  std={physics_all['growth_proxy'].std():.3f}")
    print(f"  thermal_stress: mean={physics_all['thermal_stress'].mean():.3f}  std={physics_all['thermal_stress'].std():.3f}")
    print(f"  water_stress  : mean={physics_all['water_stress'].mean():.3f}  std={physics_all['water_stress'].std():.3f}")

    crops           = df["crop"].astype(str)
    stratify_labels = make_stratify_labels(df, min_count=max(2, args.folds))
    num_crops       = int(df["crop_id"].max() + 1)
    num_districts   = int(df["district_id"].max() + 1)

    # ---- Checkpointing ---------------------------------------------------
    ckpt = load_checkpoint() if args.resume else {"completed_folds": [], "rows": []}
    completed_folds: List[int] = ckpt["completed_folds"]
    rows: List[dict]           = ckpt.get("rows", [])
    if completed_folds:
        print(f"\n  Resuming — folds already done: {completed_folds}")

    # ---- 5-Fold CV -------------------------------------------------------
    skf      = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    oof_pred = np.full(len(df), np.nan, dtype=np.float32)
    if not rows:
        rows = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, stratify_labels), start=1):
        if fold in completed_folds:
            print(f"\nFold {fold}/{args.folds} -- SKIPPED (checkpoint)")
            continue

        print(f"\nFold {fold}/{args.folds}  (train={len(tr_idx)}, val={len(va_idx)})")

        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df   = df.iloc[va_idx].reset_index(drop=True)

        # Tabular inputs
        (
            ndvi_tr, weather_tr, soil_tr, year_tr, crop_tr, dist_tr,
            ndvi_sc, weather_sc, soil_sc
        ) = prepare_tabular_inputs(train_df, cnn_feats[tr_idx], fit=True)

        (
            ndvi_va, weather_va, soil_va, year_va, crop_va, dist_va, _, _, _
        ) = prepare_tabular_inputs(
            val_df, cnn_feats[va_idx],
            ndvi_scaler=ndvi_sc, weather_scaler=weather_sc, soil_scaler=soil_sc,
            fit=False,
        )

        train_inputs_dict = {
            "satellite_features": cnn_feats[tr_idx],
            "ndvi": ndvi_tr, "weather": weather_tr,
            "crop_id": crop_tr, "district_id": dist_tr,
            "year_norm": year_tr, "soil_ph": soil_tr,
        }
        val_inputs_dict = {
            "satellite_features": cnn_feats[va_idx],
            "ndvi": ndvi_va, "weather": weather_va,
            "crop_id": crop_va, "district_id": dist_va,
            "year_norm": year_va, "soil_ph": soil_va,
        }

        # Physics labels for this fold's split
        def _slice_physics(idx):
            return {k: v[idx] for k, v in physics_all.items()}

        physics_tr = _slice_physics(tr_idx)
        physics_va = _slice_physics(va_idx)

        y_tr = y_scaled[tr_idx]
        y_va = y_scaled[va_idx]

        # Build model (fast variant)
        model = build_experiment3_fast_model(
            num_crops=num_crops, num_districts=num_districts,
        )

        # PINN trainer
        trainer = PINNTrainer(
            model=model, y_scaler=y_scaler,
            lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        )

        trainer.fit(
            train_inputs=train_inputs_dict,
            y_train=y_tr,
            physics_train=physics_tr,
            val_inputs=val_inputs_dict,
            y_val=y_va,
            physics_val=physics_va,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            callbacks_extra=[ProgressLogger(fold=fold, total_folds=args.folds)],
        )

        # Evaluate
        val_pred_scaled = model.predict(val_inputs_dict, verbose=0).reshape(-1, 1)
        val_pred        = y_scaler.inverse_transform(val_pred_scaled).reshape(-1)
        oof_pred[va_idx] = val_pred

        fold_row = evaluate_rows("experiment4_pinn", f"fold_{fold}", "overall", y[va_idx], val_pred)
        rows.append(fold_row)
        print(f"  R2={fold_row['r2']:.4f} | MAE={fold_row['mae']:.4f} | RMSE={fold_row['rmse']:.4f}")

        # Save checkpoint
        completed_folds.append(fold)
        ckpt["completed_folds"] = completed_folds
        ckpt["rows"]            = rows
        save_checkpoint(ckpt)
        print(f"  Checkpoint saved (folds done: {completed_folds})")

    # ---- OOF metrics -----------------------------------------------------
    rows.append(evaluate_rows("experiment4_pinn", "oof", "overall", y, oof_pred))
    for crop in sorted(crops.unique()):
        mask = (crops == crop).to_numpy()
        rows.append(evaluate_rows("experiment4_pinn", "oof", crop, y[mask], oof_pred[mask]))

    results_df = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved CV metrics -> {RESULTS_PATH}")

    # ---- Summary ---------------------------------------------------------
    oof_row = results_df[(results_df["split"] == "oof") & (results_df["crop"] == "overall")].iloc[0]
    print(f"\nOOF Summary -> R2={oof_row['r2']:.4f} | MAE={oof_row['mae']:.4f} | RMSE={oof_row['rmse']:.4f}")

    # Compare all experiments
    print("\n  Experiment comparison:")
    comparisons = [
        ("Exp1 Tabular Baseline", "model_metrics.csv",     "xgboost_best"),
        ("Exp2 Neural (no imgs)", "model_metrics_exp2.csv", "experiment2_neural"),
        ("Exp3 Neural + Images",  "model_metrics_exp3.csv", "experiment3_with_images"),
    ]
    for label, fname, model_name in comparisons:
        fpath = PROJECT_ROOT / "results" / "tables" / fname
        if fpath.exists():
            tmp = pd.read_csv(fpath)
            row = tmp[(tmp["split"] == "oof") & (tmp["crop"] == "overall")]
            if not row.empty:
                r = row.iloc[0]
                print(f"    {label:30s}  R2={r['r2']:.4f}  MAE={r['mae']:.4f}")
    print(f"    {'Exp4 PINN':30s}  R2={oof_row['r2']:.4f}  MAE={oof_row['mae']:.4f}  <-- current")

    # Per-crop
    print("\n  Per-crop OOF:")
    per_crop = results_df[(results_df["split"] == "oof") & (results_df["crop"] != "overall")]
    for _, r in per_crop.iterrows():
        print(f"    {r['crop']:15s}  R2={r['r2']:.4f}  MAE={r['mae']:.4f}")

    # ---- Final model save ------------------------------------------------
    if not args.skip_final_train:
        print("\n--- Final full-data training ---")
        (ndvi_full, weather_full, soil_full, year_full, crop_full, dist_full,
         ndvi_sc, weather_sc, soil_sc) = prepare_tabular_inputs(df, cnn_feats, fit=True)

        full_inputs = {
            "satellite_features": cnn_feats, "ndvi": ndvi_full, "weather": weather_full,
            "crop_id": crop_full, "district_id": dist_full,
            "year_norm": year_full, "soil_ph": soil_full,
        }

        final_model = build_experiment3_fast_model(
            num_crops=num_crops, num_districts=num_districts,
        )
        trainer_final = PINNTrainer(
            model=final_model, y_scaler=y_scaler,
            lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        )
        trainer_final.fit(
            train_inputs=full_inputs, y_train=y_scaled,
            physics_train=physics_all,
            val_inputs=full_inputs, y_val=y_scaled,   # use full data for val (final model)
            physics_val=physics_all,
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_model.save(MODEL_PATH)
        joblib.dump(
            {
                "ndvi_scaler": ndvi_sc, "weather_scaler": weather_sc,
                "soil_scaler": soil_sc, "y_scaler": y_scaler,
                "sat_min": sat_min, "sat_range": sat_range,
                "num_crops": num_crops, "num_districts": num_districts,
                "lambda1": args.lambda1, "lambda2": args.lambda2, "lambda3": args.lambda3,
                "image_h": IMAGE_H, "image_w": IMAGE_W,
            },
            PREPROC_PATH,
        )
        print(f"Saved model   -> {MODEL_PATH}")
        print(f"Saved preproc -> {PREPROC_PATH}")


if __name__ == "__main__":
    main()
