"""
Experiment 1: Tabular baseline models for crop yield prediction.

Usage:
    python src/models/baseline_tabular.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = PROJECT_ROOT / "data" / "processed" / "final_dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
MODEL_DIR = PROJECT_ROOT / "api" / "models"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical = ["crop", "district"]
    numeric_candidates = [
        "year_normalized",
        "soil_ph",
        "ndvi_mean",
        "ndvi_max",
        "ndvi_min",
        "ndvi_range",
        "ndvi_std",
        "ndvi_peak_month",
        "ndvi_slope",
        "weather_temp_mean_avg",
        "weather_temp_max_avg",
        "weather_total_rainfall",
        "weather_rain_variance",
        "weather_dry_weeks",
        "weather_max_consec_dry_weeks",
        "satellite_image_count",
        "has_satellite_images",
    ]
    numeric = [c for c in numeric_candidates if c in df.columns]
    return categorical, numeric


def build_preprocessor(categorical: List[str], numeric: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
    )


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "random_forest": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        random_state=42,
                        n_jobs=1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        )
    }

    try:
        from xgboost import XGBRegressor

        models["xgboost"] = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=600,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    except Exception:
        pass

    return models


def evaluate_cv(
    model_name: str,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    crops: pd.Series,
    stratify_labels: pd.Series,
) -> Tuple[List[dict], List[dict], np.ndarray]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_rows: List[dict] = []
    oof_pred = np.full(len(X), np.nan)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, stratify_labels), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        oof_pred[va_idx] = pred

        fold_rows.append(
            {
                "model": model_name,
                "split": f"fold_{fold}",
                "crop": "overall",
                "r2": r2_score(y_va, pred),
                "mae": mean_absolute_error(y_va, pred),
                "rmse": rmse(y_va.to_numpy(), pred),
            }
        )

    crop_rows: List[dict] = []
    for crop in sorted(crops.unique()):
        mask = crops == crop
        y_c = y[mask].to_numpy()
        p_c = oof_pred[mask]
        crop_rows.append(
            {
                "model": model_name,
                "split": "oof",
                "crop": crop,
                "r2": r2_score(y_c, p_c),
                "mae": mean_absolute_error(y_c, p_c),
                "rmse": rmse(y_c, p_c),
            }
        )

    crop_rows.append(
        {
            "model": model_name,
            "split": "oof",
            "crop": "overall",
            "r2": r2_score(y, oof_pred),
            "mae": mean_absolute_error(y, oof_pred),
            "rmse": rmse(y.to_numpy(), oof_pred),
        }
    )

    return fold_rows, crop_rows, oof_pred


def make_stratify_labels(df: pd.DataFrame, min_count: int = 5) -> pd.Series:
    combo = df["crop"].astype(str) + "__" + df["district"].astype(str)
    counts = combo.value_counts()
    # Pool sparse crop+district combos into one shared stratum for stable 5-fold CV.
    return combo.where(combo.map(counts) >= min_count, "__RARE__")


def main() -> None:
    print("=" * 70)
    print("Experiment 1: Tabular Baselines (5-fold CV)")
    print("=" * 70)

    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    if "yield_value" not in df.columns:
        raise ValueError("Expected target column `yield_value` in final_dataset.csv")

    categorical, numeric = get_feature_columns(df)
    feature_cols = categorical + numeric
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected feature columns: {missing_features}")

    X = df[feature_cols].copy()
    y = df["yield_value"].astype(float)
    crops = df["crop"].astype(str)
    stratify_labels = make_stratify_labels(df, min_count=5)

    # Convert bool column to int for consistent model behavior.
    if "has_satellite_images" in X.columns:
        X["has_satellite_images"] = X["has_satellite_images"].astype(int)

    preprocessor = build_preprocessor(categorical, numeric)
    models = build_models(preprocessor)

    if not models:
        raise RuntimeError("No baseline models are available.")

    all_rows: List[dict] = []
    model_summary: List[Tuple[str, float]] = []

    for model_name, model in models.items():
        print(f"\nTraining: {model_name}")
        fold_rows, crop_rows, _ = evaluate_cv(model_name, model, X, y, crops, stratify_labels)
        all_rows.extend(fold_rows)
        all_rows.extend(crop_rows)

        oof_overall = [r for r in crop_rows if r["crop"] == "overall"][0]
        model_summary.append((model_name, oof_overall["rmse"]))
        print(
            f"  OOF -> R2: {oof_overall['r2']:.4f} | "
            f"MAE: {oof_overall['mae']:.4f} | RMSE: {oof_overall['rmse']:.4f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(all_rows)
    metrics_path = RESULTS_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics: {metrics_path}")

    best_model_name = sorted(model_summary, key=lambda x: x[1])[0][0]
    print(f"Best model by OOF RMSE: {best_model_name}")

    best_model = models[best_model_name]
    best_model.fit(X, y)
    model_path = MODEL_DIR / f"{best_model_name}_baseline.pkl"
    joblib.dump(best_model, model_path)
    print(f"Saved model: {model_path}")

    summary = (
        metrics_df[(metrics_df["split"] == "oof") & (metrics_df["crop"] == "overall")]
        .sort_values("rmse")
        .reset_index(drop=True)
    )
    print("\nOverall OOF summary:")
    for _, row in summary.iterrows():
        print(
            f"  {row['model']:14s} -> R2: {row['r2']:.4f}, "
            f"MAE: {row['mae']:.4f}, RMSE: {row['rmse']:.4f}"
        )


if __name__ == "__main__":
    main()
