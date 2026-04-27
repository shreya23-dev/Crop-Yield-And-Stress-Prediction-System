"""
physics_loss.py — Physics-informed loss terms for crop yield prediction.

Implements three agricultural science constraints as differentiable TF ops:
  L_growth     — NDVI -> LAI -> fIPAR growth proxy (Monteith radiation-use efficiency)
  L_temperature — Crop-specific thermal stress (trapezoidal response curve)
  L_water      — FAO Hargreaves water stress (ETa / ETm ratio)

Each function takes numpy arrays and returns a scalar float (compatible with
custom Keras training loops via tf.py_function wrappers).

Reference:
  Growth:      Monteith (1977) radiation-use efficiency
  Temperature: DSSAT crop model thermal response
  Water:       FAO Irrigation & Drainage Paper No. 33 (Doorenbos & Kassam 1979)
               Hargreaves & Samani (1985) ETo equation
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# Crop thermal parameters (T_base, T_opt, T_ceil) in degrees Celsius
# ---------------------------------------------------------------------------
THERMAL_PARAMS: Dict[str, tuple] = {
    "Rice":         (10.0, 30.0, 42.0),
    "Jowar":        (8.0,  32.0, 44.0),
    "Bajra":        (10.0, 33.0, 45.0),
    "Soyabean":     (10.0, 28.0, 40.0),
    "Cotton(lint)": (15.0, 30.0, 40.0),
    # fallback
    "default":      (10.0, 30.0, 42.0),
}

# FAO crop yield response factor to water deficit (Ky)
KY_VALUES: Dict[str, float] = {
    "Rice":         1.05,
    "Jowar":        0.90,
    "Bajra":        0.70,
    "Soyabean":     0.85,
    "Cotton(lint)": 0.85,
    "default":      0.90,
}

# Light extinction coefficient
K_LIGHT: float = 0.5
# Radiation-use efficiency (g/MJ PAR) — relative, used for normalization
RUE: float = 1.0
# Latitude of Maharashtra centroid (used for Ra when lat not available)
DEFAULT_LAT_DEG: float = 19.7


# ---------------------------------------------------------------------------
# 1. L_growth — NDVI → LAI → fIPAR → growth proxy
# ---------------------------------------------------------------------------

def compute_growth_proxy(ndvi_matrix: np.ndarray) -> np.ndarray:
    """
    Compute seasonal growth proxy for each sample from monthly NDVI sequence.

    Args:
        ndvi_matrix: shape (N, 6)  — monthly NDVI Jun-Nov, values in [0, 1]

    Returns:
        growth_proxy: shape (N,) — normalized seasonal growth proxy [0, 1]
    """
    ndvi = np.clip(ndvi_matrix, 0.001, 0.999)   # avoid log(0) and log(1→0)

    # LAI from Beer-Lambert inversion
    lai = -(1.0 / K_LIGHT) * np.log(1.0 - ndvi)

    # Fraction of intercepted PAR
    fipar = 1.0 - np.exp(-K_LIGHT * lai)

    # Season-integrated growth proxy (sum over 6 months × RUE)
    growth_proxy = fipar.sum(axis=1) * RUE        # shape (N,)

    # Normalize to [0, 1] using dataset range
    gp_min, gp_max = growth_proxy.min(), growth_proxy.max()
    if gp_max - gp_min > 1e-6:
        growth_proxy = (growth_proxy - gp_min) / (gp_max - gp_min)

    return growth_proxy.astype(np.float32)


def loss_growth(
    y_pred_norm: np.ndarray,
    ndvi_matrix: np.ndarray,
) -> float:
    """
    L_growth = MSE(normalized_predicted_yield, normalized_growth_proxy)

    Args:
        y_pred_norm: shape (N,) — yield predictions normalized to [0,1]
        ndvi_matrix: shape (N, 6) — monthly NDVI values

    Returns:
        Scalar MSE loss.
    """
    growth_proxy = compute_growth_proxy(ndvi_matrix)
    return float(np.mean((y_pred_norm - growth_proxy) ** 2))


# ---------------------------------------------------------------------------
# 2. L_temperature — Trapezoidal thermal response
# ---------------------------------------------------------------------------

def thermal_response(temp: np.ndarray, t_base: float, t_opt: float, t_ceil: float) -> np.ndarray:
    """
    Trapezoidal thermal response curve [0, 1].

    0               if T < T_base or T > T_ceil
    linear rise     if T_base <= T < T_opt
    linear fall     if T_opt <= T <= T_ceil
    """
    result = np.zeros_like(temp, dtype=np.float32)

    # Rising slope
    rise_mask = (temp >= t_base) & (temp < t_opt)
    result[rise_mask] = (temp[rise_mask] - t_base) / (t_opt - t_base)

    # Falling slope
    fall_mask = (temp >= t_opt) & (temp <= t_ceil)
    result[fall_mask] = (t_ceil - temp[fall_mask]) / (t_ceil - t_opt)

    return result


def compute_thermal_stress(
    temp_mean_matrix: np.ndarray,
    crop_names: np.ndarray,
) -> np.ndarray:
    """
    Compute season-mean thermal stress score for each sample.

    Args:
        temp_mean_matrix: shape (N, 22) — weekly mean temperatures
        crop_names:       shape (N,) — crop name strings

    Returns:
        thermal_stress: shape (N,) — mean stress score in [0, 1]
                         (0 = no stress, 1 = full stress)
    """
    N = len(temp_mean_matrix)
    stress = np.zeros(N, dtype=np.float32)

    for i in range(N):
        crop = crop_names[i] if crop_names[i] in THERMAL_PARAMS else "default"
        t_base, t_opt, t_ceil = THERMAL_PARAMS[crop]
        weekly_temps = temp_mean_matrix[i]
        resp = thermal_response(weekly_temps, t_base, t_opt, t_ceil)
        # Stress = 1 - response (0 = optimal, 1 = complete stress)
        stress[i] = 1.0 - resp.mean()

    return stress


def loss_temperature(
    y_pred_norm: np.ndarray,
    temp_mean_matrix: np.ndarray,
    crop_names: np.ndarray,
) -> float:
    """
    L_temperature penalizes high predicted yield when thermal stress is high.

    Specifically: MSE(y_pred_norm, 1 - thermal_stress)
    i.e., if temperatures were very stressful, we expect low yield.

    Args:
        y_pred_norm:      shape (N,) — normalized yield predictions
        temp_mean_matrix: shape (N, 22) — weekly mean temps
        crop_names:       shape (N,) — crop strings

    Returns:
        Scalar MSE loss.
    """
    thermal_stress = compute_thermal_stress(temp_mean_matrix, crop_names)
    # Expected yield direction: lower stress -> higher yield
    expected_yield_proxy = 1.0 - thermal_stress
    return float(np.mean((y_pred_norm - expected_yield_proxy) ** 2))


# ---------------------------------------------------------------------------
# 3. L_water — FAO Hargreaves water stress
# ---------------------------------------------------------------------------

def _extraterrestrial_radiation(lat_deg: float, doy: int) -> float:
    """
    Compute extraterrestrial radiation Ra [MJ/m²/day] via FAO-56 formula.

    Args:
        lat_deg: latitude in degrees
        doy:     day of year (1-365)

    Returns:
        Ra in MJ/m²/day
    """
    lat_rad = math.radians(lat_deg)
    dr = 1.0 + 0.033 * math.cos(2 * math.pi * doy / 365.0)
    delta = 0.409 * math.sin(2 * math.pi * doy / 365.0 - 1.39)
    ws = math.acos(-math.tan(lat_rad) * math.tan(delta))
    Ra = (24.0 * 60.0 / math.pi) * 0.082 * dr * (
        ws * math.sin(lat_rad) * math.sin(delta)
        + math.cos(lat_rad) * math.cos(delta) * math.sin(ws)
    )
    return Ra


def _hargreaves_eto(t_mean: float, t_max: float, ra: float) -> float:
    """
    Hargreaves & Samani (1985) reference evapotranspiration [mm/day].

    ETo = 0.0023 × (T_mean + 17.8) × (T_max - T_mean)^0.5 × Ra
    Note: using T_max - T_mean as proxy for DTR when T_min is unavailable.
    """
    dtr = max(t_max - t_mean, 0.0)
    return 0.0023 * (t_mean + 17.8) * math.sqrt(dtr) * ra


# Pre-compute mean Ra for Maharashtra kharif season (Jun–Nov, doys 152-335)
_KHARIF_DOYS = list(range(152, 336, 7))   # weekly for 26 weeks but we use 22
_DEFAULT_RA = [_extraterrestrial_radiation(DEFAULT_LAT_DEG, d) for d in _KHARIF_DOYS[:22]]


def compute_water_stress(
    temp_mean_matrix: np.ndarray,
    temp_max_matrix: np.ndarray,
    rain_matrix: np.ndarray,
    lat_array: np.ndarray | None = None,
    crop_names: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute seasonal water stress (ETa / ETm deficit) for each sample.

    ETm (potential ET) from Hargreaves equation using temperature.
    ETa (actual ET) approximated from weekly rainfall with simple water balance.

    Returns:
        water_stress: shape (N,) — stress in [0, 1]
                      (0 = no water stress, 1 = complete water deficit)
    """
    N = len(temp_mean_matrix)
    stress = np.zeros(N, dtype=np.float32)

    for i in range(N):
        lat = float(lat_array[i]) if lat_array is not None else DEFAULT_LAT_DEG
        # Weekly Ra values
        ra_vals = [_extraterrestrial_radiation(lat, d) for d in _KHARIF_DOYS[:22]]

        etm_total = 0.0
        eta_total  = 0.0

        for w in range(22):
            t_mean = float(temp_mean_matrix[i, w])
            t_max  = float(temp_max_matrix[i, w])
            rain   = float(rain_matrix[i, w])
            ra     = ra_vals[w]

            # Weekly ETm (mm/week)
            eto_daily = _hargreaves_eto(t_mean, t_max, ra)
            etm_week  = eto_daily * 7.0

            # ETa approximated as min(rainfall, ETm) — simple bucket model
            eta_week = min(rain, etm_week)

            etm_total += etm_week
            eta_total  += eta_week

        # Avoid division by zero
        if etm_total < 1e-3:
            stress[i] = 0.0
        else:
            eta_ratio = eta_total / etm_total
            stress[i] = max(0.0, 1.0 - eta_ratio)

    return stress.astype(np.float32)


def loss_water(
    y_pred_norm: np.ndarray,
    temp_mean_matrix: np.ndarray,
    temp_max_matrix: np.ndarray,
    rain_matrix: np.ndarray,
    lat_array: np.ndarray | None = None,
    crop_names: np.ndarray | None = None,
) -> float:
    """
    L_water penalizes high predicted yield when water deficit is high.

    MSE(y_pred_norm, 1 - water_stress)

    Returns:
        Scalar MSE loss.
    """
    water_stress = compute_water_stress(
        temp_mean_matrix, temp_max_matrix, rain_matrix, lat_array, crop_names
    )
    expected_yield_proxy = 1.0 - water_stress
    return float(np.mean((y_pred_norm - expected_yield_proxy) ** 2))


# ---------------------------------------------------------------------------
# 4. Combined stress label (for the stress prediction head in Exp5)
# ---------------------------------------------------------------------------

def compute_combined_stress(
    temp_mean_matrix: np.ndarray,
    temp_max_matrix: np.ndarray,
    rain_matrix: np.ndarray,
    crop_names: np.ndarray,
    lat_array: np.ndarray | None = None,
    w_thermal: float = 0.4,
    w_water:   float = 0.6,
) -> np.ndarray:
    """
    Combined stress label = 0.4 * thermal_stress + 0.6 * water_stress

    Used as the target for the stress head in Experiment 5 (multi-task PINN).

    Returns:
        combined_stress: shape (N,) in [0, 1]
    """
    thermal = compute_thermal_stress(temp_mean_matrix, crop_names)
    water   = compute_water_stress(
        temp_mean_matrix, temp_max_matrix, rain_matrix, lat_array, crop_names
    )
    return (w_thermal * thermal + w_water * water).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Convenience: compute all physics labels in one call
# ---------------------------------------------------------------------------

def compute_all_physics_labels(
    ndvi_matrix:      np.ndarray,
    temp_mean_matrix: np.ndarray,
    temp_max_matrix:  np.ndarray,
    rain_matrix:      np.ndarray,
    crop_names:       np.ndarray,
    lat_array:        np.ndarray | None = None,
) -> dict:
    """
    Pre-compute all physics constraint labels for a dataset split.
    Call once before training to avoid recomputing per batch.

    Returns dict with keys:
        growth_proxy    shape (N,) in [0,1]
        thermal_stress  shape (N,) in [0,1]
        water_stress    shape (N,) in [0,1]
        combined_stress shape (N,) in [0,1]
    """
    growth_proxy    = compute_growth_proxy(ndvi_matrix)
    thermal_stress  = compute_thermal_stress(temp_mean_matrix, crop_names)
    water_stress    = compute_water_stress(
        temp_mean_matrix, temp_max_matrix, rain_matrix, lat_array, crop_names
    )
    combined_stress = (0.4 * thermal_stress + 0.6 * water_stress).astype(np.float32)

    return {
        "growth_proxy":    growth_proxy,
        "thermal_stress":  thermal_stress,
        "water_stress":    water_stress,
        "combined_stress": combined_stress,
    }
