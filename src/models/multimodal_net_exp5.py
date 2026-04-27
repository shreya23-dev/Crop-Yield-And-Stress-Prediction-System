"""
Experiment 5 model: PINN + Multi-task (Yield + Stress prediction).

Extends Exp4 by adding a second output head for stress prediction:
  - yield head   -> Dense(1)               predicts crop yield
  - stress head  -> Dense(48) -> Dense(1)  predicts combined stress index [0,1]

The model is identical to the fast variant (pre-computed CNN features input).
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models


def build_experiment5_model(
    num_crops: int,
    num_districts: int,
    ndvi_timesteps: int = 6,
    weather_timesteps: int = 22,
    weather_features: int = 3,
    cnn_feature_dim: int = 128,
) -> tf.keras.Model:
    """
    Multi-task model with yield + stress prediction heads.

    Inputs: same as Exp3/Exp4 fast model (pre-computed satellite CNN features).
    Outputs: [yield_output (N,1), stress_output (N,1)]
    """
    # Branch 1 — Satellite CNN features -> LSTM
    sat_input = layers.Input(shape=(ndvi_timesteps, cnn_feature_dim), name="satellite_features")
    x_sat = layers.LSTM(64, name="sat_lstm")(sat_input)

    # Branch 2 — NDVI Conv1D
    ndvi_input = layers.Input(shape=(ndvi_timesteps, 1), name="ndvi")
    x_ndvi = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(ndvi_input)
    x_ndvi = layers.Dropout(0.3)(x_ndvi)
    x_ndvi = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x_ndvi)
    x_ndvi = layers.GlobalAveragePooling1D()(x_ndvi)
    x_ndvi = layers.Dense(32, activation="relu")(x_ndvi)

    # Branch 3 — Weather LSTM
    weather_input = layers.Input(shape=(weather_timesteps, weather_features), name="weather")
    x_weather = layers.LSTM(64, return_sequences=False)(weather_input)
    x_weather = layers.Dense(64, activation="relu")(x_weather)

    # Branch 4 — Static
    crop_input     = layers.Input(shape=(1,), dtype="int32",   name="crop_id")
    district_input = layers.Input(shape=(1,), dtype="int32",   name="district_id")
    year_input     = layers.Input(shape=(1,), dtype="float32", name="year_norm")
    soil_input     = layers.Input(shape=(1,), dtype="float32", name="soil_ph")

    crop_emb     = layers.Flatten()(layers.Embedding(num_crops,     8,  name="crop_embedding")(crop_input))
    district_emb = layers.Flatten()(layers.Embedding(num_districts, 16, name="district_embedding")(district_input))

    x_static = layers.Concatenate()([crop_emb, district_emb, year_input, soil_input])
    x_static = layers.Dense(32, activation="relu")(x_static)

    # Shared fusion: 64 + 32 + 64 + 32 = 192
    fused = layers.Concatenate(name="fusion")([x_sat, x_ndvi, x_weather, x_static])
    fused = layers.Dense(96, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)
    shared = layers.Dense(48, activation="relu", name="shared_48")(fused)

    # ---------- Task 1: Yield head ----------
    yield_output = layers.Dense(1, name="yield")(shared)

    # ---------- Task 2: Stress head ----------
    # Separate branch from shared layer to allow task-specific learning
    x_stress = layers.Dense(32, activation="relu", name="stress_dense")(shared)
    x_stress = layers.Dropout(0.2)(x_stress)
    stress_output = layers.Dense(1, activation="sigmoid", name="stress")(x_stress)
    # sigmoid constrains stress to [0, 1], matching physics-derived labels

    model = models.Model(
        inputs=[sat_input, ndvi_input, weather_input, crop_input, district_input, year_input, soil_input],
        outputs={"yield": yield_output, "stress": stress_output},
        name="experiment5_pinn_multitask",
    )
    return model
