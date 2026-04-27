"""
Keras multimodal model for Experiment 2 (no satellite image branch).
Inputs:
  - NDVI monthly sequence: (6, 1)
  - Weather weekly sequence: (22, 3)
  - Static features: crop_id, district_id, year_normalized, soil_ph_scaled
Outputs:
  - Yield prediction (regression)
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_experiment2_model(
    num_crops: int,
    num_districts: int,
    ndvi_timesteps: int = 6,
    weather_timesteps: int = 22,
    weather_features: int = 3,
) -> tf.keras.Model:
    # NDVI branch: Conv1D over 6 monthly values.
    ndvi_input = layers.Input(shape=(ndvi_timesteps, 1), name="ndvi")
    x_ndvi = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(ndvi_input)
    x_ndvi = layers.Dropout(0.3)(x_ndvi)
    x_ndvi = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x_ndvi)
    x_ndvi = layers.GlobalAveragePooling1D()(x_ndvi)
    x_ndvi = layers.Dense(32, activation="relu")(x_ndvi)

    # Weather branch: LSTM over 22 weeks.
    weather_input = layers.Input(shape=(weather_timesteps, weather_features), name="weather")
    x_weather = layers.LSTM(64, return_sequences=False)(weather_input)
    x_weather = layers.Dense(64, activation="relu")(x_weather)

    # Static branch with embeddings.
    crop_input = layers.Input(shape=(1,), dtype="int32", name="crop_id")
    district_input = layers.Input(shape=(1,), dtype="int32", name="district_id")
    year_input = layers.Input(shape=(1,), dtype="float32", name="year_norm")
    soil_input = layers.Input(shape=(1,), dtype="float32", name="soil_ph")

    crop_emb = layers.Embedding(num_crops, 8, name="crop_embedding")(crop_input)
    crop_emb = layers.Flatten()(crop_emb)
    district_emb = layers.Embedding(num_districts, 16, name="district_embedding")(district_input)
    district_emb = layers.Flatten()(district_emb)

    x_static = layers.Concatenate()([crop_emb, district_emb, year_input, soil_input])
    x_static = layers.Dense(32, activation="relu")(x_static)

    # Fusion + output.
    fused = layers.Concatenate()([x_ndvi, x_weather, x_static])
    fused = layers.Dense(96, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)
    fused = layers.Dense(48, activation="relu")(fused)

    yield_output = layers.Dense(1, name="yield")(fused)

    model = models.Model(
        inputs=[ndvi_input, weather_input, crop_input, district_input, year_input, soil_input],
        outputs=yield_output,
        name="experiment2_multimodal_no_images",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model

