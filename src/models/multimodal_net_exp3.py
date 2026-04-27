"""
Experiment 3 model: Full 4-branch neural multimodal (WITH satellite images).

Architecture:
  Branch 1 — Satellite: TimeDistributed CNN -> LSTM  -> 64-dim
  Branch 2 — NDVI:      Conv1D               -> Dense -> 32-dim
  Branch 3 — Weather:   LSTM                          -> 64-dim
  Branch 4 — Static:    Embeddings + Dense            -> 32-dim
  Fusion:               Concat(192) -> Dense(96) -> Dense(48) -> yield (1)

Satellite input shape: (batch, 6, 64, 64, C)
  6 monthly images, each 64x64 with C channels (C=1 for single-band NDVI raster).

Loss: MSE only (no physics — that comes in Experiment 4).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_satellite_cnn(image_h: int, image_w: int, image_channels: int) -> tf.keras.Model:
    """
    Shared CNN that maps one month's image (H, W, C) -> 128-dim feature.
    Applied via TimeDistributed to all 6 months.
    Works for 32x32 input (2 MaxPool layers -> 8x8 -> GlobalAvgPool).
    """
    inp = layers.Input(shape=(image_h, image_w, image_channels))

    x = layers.Conv2D(32, kernel_size=3, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)           # 32->16

    x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)           # 16->8

    x = layers.Conv2D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)             # -> (128,)

    x = layers.Dropout(0.3)(x)

    return models.Model(inputs=inp, outputs=x, name="satellite_cnn")


def build_experiment3_model(
    num_crops: int,
    num_districts: int,
    ndvi_timesteps: int = 6,
    weather_timesteps: int = 22,
    weather_features: int = 3,
    image_h: int = 32,
    image_w: int = 32,
    image_channels: int = 1,
) -> tf.keras.Model:
    """
    Full 4-branch multimodal model for Experiment 3.

    Args:
        num_crops:       Number of unique crops (for embedding).
        num_districts:   Number of unique districts (for embedding).
        ndvi_timesteps:  Number of monthly NDVI values (6).
        weather_timesteps: Number of weekly weather steps (22).
        weather_features:  Number of weather variables per step (3).
        image_h, image_w: Spatial size of each satellite patch.
        image_channels:  Number of raster channels per image.

    Returns:
        Compiled Keras model.
    """

    # ------------------------------------------------------------------
    # Branch 1 — Satellite CNN + LSTM
    # Input: (batch, 6, H, W, C)
    # ------------------------------------------------------------------
    sat_input = layers.Input(
        shape=(ndvi_timesteps, image_h, image_w, image_channels),
        name="satellite",
    )

    shared_cnn = build_satellite_cnn(image_h, image_w, image_channels)

    # Apply shared CNN to each of the 6 monthly images -> (batch, 6, 128)
    x_sat = layers.TimeDistributed(shared_cnn, name="td_cnn")(sat_input)

    # LSTM over months to capture seasonal evolution of spatial features
    x_sat = layers.LSTM(64, name="sat_lstm")(x_sat)      # -> (batch, 64)

    # ------------------------------------------------------------------
    # Branch 2 — NDVI Conv1D
    # Input: (batch, 6, 1)
    # ------------------------------------------------------------------
    ndvi_input = layers.Input(shape=(ndvi_timesteps, 1), name="ndvi")
    x_ndvi = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(ndvi_input)
    x_ndvi = layers.Dropout(0.3)(x_ndvi)
    x_ndvi = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x_ndvi)
    x_ndvi = layers.GlobalAveragePooling1D()(x_ndvi)
    x_ndvi = layers.Dense(32, activation="relu")(x_ndvi)  # -> (batch, 32)

    # ------------------------------------------------------------------
    # Branch 3 — Weather LSTM
    # Input: (batch, 22, 3)
    # ------------------------------------------------------------------
    weather_input = layers.Input(shape=(weather_timesteps, weather_features), name="weather")
    x_weather = layers.LSTM(64, return_sequences=False)(weather_input)
    x_weather = layers.Dense(64, activation="relu")(x_weather)  # -> (batch, 64)

    # ------------------------------------------------------------------
    # Branch 4 — Static Embeddings + Dense
    # Inputs: crop_id (int), district_id (int), year_norm (float), soil_ph (float)
    # ------------------------------------------------------------------
    crop_input     = layers.Input(shape=(1,), dtype="int32",   name="crop_id")
    district_input = layers.Input(shape=(1,), dtype="int32",   name="district_id")
    year_input     = layers.Input(shape=(1,), dtype="float32", name="year_norm")
    soil_input     = layers.Input(shape=(1,), dtype="float32", name="soil_ph")

    crop_emb     = layers.Flatten()(layers.Embedding(num_crops,     8,  name="crop_embedding")(crop_input))
    district_emb = layers.Flatten()(layers.Embedding(num_districts, 16, name="district_embedding")(district_input))

    # Concat all static: 8 + 16 + 1 + 1 = 26
    x_static = layers.Concatenate()([crop_emb, district_emb, year_input, soil_input])
    x_static = layers.Dense(32, activation="relu")(x_static)  # -> (batch, 32)

    # ------------------------------------------------------------------
    # Fusion: 64 + 32 + 64 + 32 = 192
    # ------------------------------------------------------------------
    fused = layers.Concatenate(name="fusion")([x_sat, x_ndvi, x_weather, x_static])
    fused = layers.Dense(96, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)
    fused = layers.Dense(48, activation="relu", name="fusion_48")(fused)

    # Yield output head
    yield_output = layers.Dense(1, name="yield")(fused)

    model = models.Model(
        inputs=[
            sat_input,
            ndvi_input,
            weather_input,
            crop_input,
            district_input,
            year_input,
            soil_input,
        ],
        outputs=yield_output,
        name="experiment3_multimodal_with_images",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model


# ---------------------------------------------------------------------------
# FAST variant: accepts pre-computed CNN features instead of raw images
# ---------------------------------------------------------------------------

def build_experiment3_fast_model(
    num_crops: int,
    num_districts: int,
    ndvi_timesteps: int = 6,
    weather_timesteps: int = 22,
    weather_features: int = 3,
    cnn_feature_dim: int = 128,   # output dim of the satellite CNN
) -> tf.keras.Model:
    """
    Fast training model: takes PRE-COMPUTED satellite CNN features (N, 6, 128).

    This avoids running the expensive CNN on every epoch. Instead:
      1. Run build_satellite_cnn() once on ALL images -> cache (N, 6, 128)
      2. Use this model for training which only needs the smaller LSTM + Dense ops.

    Architecturally equivalent to build_experiment3_model(), just cached at
    the CNN boundary. ~10-20x faster on CPU.
    """
    # Branch 1 — Pre-computed satellite features -> LSTM
    # Input: (batch, 6, 128) — already extracted by CNN
    sat_feat_input = layers.Input(
        shape=(ndvi_timesteps, cnn_feature_dim),
        name="satellite_features",
    )
    x_sat = layers.LSTM(64, name="sat_lstm")(sat_feat_input)   # -> (batch, 64)

    # Branch 2 — NDVI Conv1D
    ndvi_input = layers.Input(shape=(ndvi_timesteps, 1), name="ndvi")
    x_ndvi = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu")(ndvi_input)
    x_ndvi = layers.Dropout(0.3)(x_ndvi)
    x_ndvi = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x_ndvi)
    x_ndvi = layers.GlobalAveragePooling1D()(x_ndvi)
    x_ndvi = layers.Dense(32, activation="relu")(x_ndvi)        # -> (batch, 32)

    # Branch 3 — Weather LSTM
    weather_input = layers.Input(shape=(weather_timesteps, weather_features), name="weather")
    x_weather = layers.LSTM(64, return_sequences=False)(weather_input)
    x_weather = layers.Dense(64, activation="relu")(x_weather)  # -> (batch, 64)

    # Branch 4 — Static Embeddings + Dense
    crop_input     = layers.Input(shape=(1,), dtype="int32",   name="crop_id")
    district_input = layers.Input(shape=(1,), dtype="int32",   name="district_id")
    year_input     = layers.Input(shape=(1,), dtype="float32", name="year_norm")
    soil_input     = layers.Input(shape=(1,), dtype="float32", name="soil_ph")

    crop_emb     = layers.Flatten()(layers.Embedding(num_crops,     8,  name="crop_embedding")(crop_input))
    district_emb = layers.Flatten()(layers.Embedding(num_districts, 16, name="district_embedding")(district_input))

    x_static = layers.Concatenate()([crop_emb, district_emb, year_input, soil_input])
    x_static = layers.Dense(32, activation="relu")(x_static)   # -> (batch, 32)

    # Fusion: 64 + 32 + 64 + 32 = 192
    fused = layers.Concatenate(name="fusion")([x_sat, x_ndvi, x_weather, x_static])
    fused = layers.Dense(96, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)
    fused = layers.Dense(48, activation="relu", name="fusion_48")(fused)

    yield_output = layers.Dense(1, name="yield")(fused)

    model = models.Model(
        inputs=[
            sat_feat_input,
            ndvi_input,
            weather_input,
            crop_input,
            district_input,
            year_input,
            soil_input,
        ],
        outputs=yield_output,
        name="experiment3_fast_multimodal",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model
