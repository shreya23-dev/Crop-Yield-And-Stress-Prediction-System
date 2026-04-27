"""
Experiment 2 training: neural multimodal model without satellite images.

Usage:
    python src/training/train_experiment2.py
"""

from __future__ import annotations

import argparse
import os
import sys
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

from src.models.multimodal_net import build_experiment2_model

DATA_CSV = PROJECT_ROOT / "data" / "processed" / "final_dataset.csv"
RESULTS_PATH = PROJECT_ROOT / "results" / "tables" / "model_metrics_exp2.csv"
MODEL_PATH = PROJECT_ROOT / "api" / "models" / "experiment2_multimodal.keras"
PREPROC_PATH = PROJECT_ROOT / "api" / "models" / "experiment2_preprocessing.pkl"

SEED = 42


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_stratify_labels(df: pd.DataFrame, min_count: int = 5) -> pd.Series:
    combo = df["crop"].astype(str) + "__" + df["district"].astype(str)
    counts = combo.value_counts()
    return combo.where(combo.map(counts) >= min_count, "__RARE__")


def get_ndvi_columns(df: pd.DataFrame) -> List[str]:
    cols = ["ndvi_jun", "ndvi_jul", "ndvi_aug", "ndvi_sep", "ndvi_oct", "ndvi_nov"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing NDVI columns: {missing}")
    return cols


def get_weather_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for week in range(1, 23):
        cols.extend(
            [
                f"week_{week}_temp_mean",
                f"week_{week}_temp_max",
                f"week_{week}_rain",
            ]
        )
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing weather columns: {missing[:6]} ... total missing={len(missing)}")
    return cols


def prepare_inputs(
    df: pd.DataFrame,
    ndvi_scaler: StandardScaler | None = None,
    weather_scaler: StandardScaler | None = None,
    soil_scaler: StandardScaler | None = None,
    fit: bool = False,
) -> Tuple[Dict[str, np.ndarray], StandardScaler, StandardScaler, StandardScaler]:
    ndvi_cols = get_ndvi_columns(df)
    weather_cols = get_weather_columns(df)

    ndvi = df[ndvi_cols].to_numpy(dtype=np.float32)  # (N, 6)
    weather_flat = df[weather_cols].to_numpy(dtype=np.float32)  # (N, 66)
    soil = df[["soil_ph"]].to_numpy(dtype=np.float32)
    year = df[["year_normalized"]].to_numpy(dtype=np.float32)
    crop_id = df[["crop_id"]].to_numpy(dtype=np.int32)
    district_id = df[["district_id"]].to_numpy(dtype=np.int32)

    if fit or ndvi_scaler is None:
        ndvi_scaler = StandardScaler().fit(ndvi)
    if fit or weather_scaler is None:
        weather_scaler = StandardScaler().fit(weather_flat)
    if fit or soil_scaler is None:
        soil_scaler = StandardScaler().fit(soil)

    ndvi_scaled = ndvi_scaler.transform(ndvi).reshape(-1, 6, 1).astype(np.float32)
    weather_scaled = weather_scaler.transform(weather_flat).reshape(-1, 22, 3).astype(np.float32)
    soil_scaled = soil_scaler.transform(soil).astype(np.float32)

    inputs = {
        "ndvi": ndvi_scaled,
        "weather": weather_scaled,
        "crop_id": crop_id,
        "district_id": district_id,
        "year_norm": year,
        "soil_ph": soil_scaled,
    }
    return inputs, ndvi_scaler, weather_scaler, soil_scaler


def slice_inputs(inputs: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[indices] for k, v in inputs.items()}


def evaluate_rows(model_name: str, split: str, crop: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": model_name,
        "split": split,
        "crop": crop,
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Experiment 2 neural multimodal model.")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=80, help="Max epochs per training run")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--skip-final-train",
        action="store_true",
        help="Skip final full-data training and save only CV metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    tf.get_logger().setLevel("ERROR")

    print("=" * 70)
    print("Experiment 2: Neural Multimodal (NDVI + Weather + Static)")
    print("=" * 70)

    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV).copy()
    required = ["yield_value", "crop", "district", "crop_id", "district_id", "year_normalized", "soil_ph"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Remove rows with missing core features.
    ndvi_cols = get_ndvi_columns(df)
    weather_cols = get_weather_columns(df)
    df = df.dropna(subset=ndvi_cols + weather_cols + ["soil_ph", "year_normalized", "yield_value"]).reset_index(drop=True)

    y = df["yield_value"].to_numpy(dtype=np.float32)
    y_scaler = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).reshape(-1).astype(np.float32)

    crops = df["crop"].astype(str)
    stratify_labels = make_stratify_labels(df, min_count=max(2, args.folds))

    num_crops = int(df["crop_id"].max() + 1)
    num_districts = int(df["district_id"].max() + 1)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    oof_pred = np.full(len(df), np.nan, dtype=np.float32)
    rows: List[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, stratify_labels), start=1):
        print(f"\nFold {fold}/{args.folds}")
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)

        train_inputs, ndvi_scaler, weather_scaler, soil_scaler = prepare_inputs(train_df, fit=True)
        val_inputs, _, _, _ = prepare_inputs(
            val_df,
            ndvi_scaler=ndvi_scaler,
            weather_scaler=weather_scaler,
            soil_scaler=soil_scaler,
            fit=False,
        )

        y_train = y_scaled[tr_idx]
        y_val = y_scaled[va_idx]

        model = build_experiment2_model(num_crops=num_crops, num_districts=num_districts)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
            )
        ]

        model.fit(
            train_inputs,
            y_train,
            validation_data=(val_inputs, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
            callbacks=callbacks,
        )

        pred_scaled = model.predict(val_inputs, verbose=0).reshape(-1, 1)
        pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)
        oof_pred[va_idx] = pred

        fold_row = evaluate_rows(
            "experiment2_neural",
            f"fold_{fold}",
            "overall",
            y[va_idx],
            pred,
        )
        rows.append(fold_row)
        print(
            f"  R2={fold_row['r2']:.4f} | MAE={fold_row['mae']:.4f} | RMSE={fold_row['rmse']:.4f}"
        )

    # OOF overall + per-crop.
    rows.append(evaluate_rows("experiment2_neural", "oof", "overall", y, oof_pred))
    for crop in sorted(crops.unique()):
        mask = crops == crop
        rows.append(
            evaluate_rows(
                "experiment2_neural",
                "oof",
                crop,
                y[mask.to_numpy()],
                oof_pred[mask.to_numpy()],
            )
        )

    results_df = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved metrics: {RESULTS_PATH}")

    if not args.skip_final_train:
        # Train final model on full dataset and save it for serving.
        full_inputs, ndvi_scaler, weather_scaler, soil_scaler = prepare_inputs(df, fit=True)
        final_model = build_experiment2_model(num_crops=num_crops, num_districts=num_districts)
        final_model.fit(
            full_inputs,
            y_scaled,
            validation_split=0.1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=args.patience,
                    restore_best_weights=True,
                )
            ],
        )

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_model.save(MODEL_PATH)
        joblib.dump(
            {
                "ndvi_scaler": ndvi_scaler,
                "weather_scaler": weather_scaler,
                "soil_scaler": soil_scaler,
                "y_scaler": y_scaler,
                "num_crops": num_crops,
                "num_districts": num_districts,
            },
            PREPROC_PATH,
        )
        print(f"Saved model: {MODEL_PATH}")
        print(f"Saved preprocessing: {PREPROC_PATH}")

    oof_overall = results_df[(results_df["split"] == "oof") & (results_df["crop"] == "overall")].iloc[0]
    print(
        "\nOOF summary -> "
        f"R2={oof_overall['r2']:.4f} | "
        f"MAE={oof_overall['mae']:.4f} | "
        f"RMSE={oof_overall['rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
