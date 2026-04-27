# COMPREHENSIVE PROJECT PROMPT (v2 — Updated with Satellite Image Processing)
# Multimodal Physics-Informed Crop Yield Prediction System
# Maharashtra, India — Final Year Project

---

## PROJECT TITLE
**Physics-Informed Multimodal Deep Learning for District-Level Kharif Crop Yield Prediction Using Satellite Imagery, Meteorological Time Series, and Soil Data in Maharashtra, India**

---

## 1. PROBLEM STATEMENT

Predict district-level annual crop yield (tonnes/hectare) for 5 Kharif season crops (Rice, Jowar, Bajra, Soyabean, Cotton) across 34 districts of Maharashtra, India, using a genuinely multimodal approach that fuses:
- **Satellite imagery** (spatial features from raw image patches)
- **Vegetation time series** (temporal NDVI profiles)
- **Meteorological time series** (weekly weather sequences)
- **Pedological data** (soil properties)
- **Static categorical features** (crop identity, district identity, year)

The system must also predict crop stress levels as a secondary output using physics-informed constraints for actionable agricultural insights.

---

## 2. DATASET

### 2.1 Ground Truth — Yield Data
- **Source**: Maharashtra Government agricultural reports (1997–2023)
- **File**: `maharashtra_kharif_yield_clean.csv`
- **Total samples**: 3,240 rows (after cleaning)
- **Columns**: state, district, latitude, longitude, year, year_label, crop, season, area_hectare, production, yield_value, yield_unit
- **Crops & counts**: Rice (699), Soyabean (699), Jowar (665), Cotton/lint (643), Bajra (534)
- **Districts**: 34 districts with lat/lon coordinates
- **Years**: 1997–2022 (26 years)
- **Season**: Kharif only (June–November growing window)
- **Note**: Cotton yield is in Bales/Hectare (1 bale = 170 kg), others in Tonnes/Hectare
- **Cleaning applied**: Removed 3 extreme outliers (data entry errors), 13 zero-yield entries, and rows with area < 50 hectares

### 2.2 Satellite Image Data (MODALITY 1 — Spatial)
- **Source**: Google Earth Engine — MODIS (MOD13Q1, 250m resolution, available from 2000) or Sentinel-2 (10m, from 2015)
- **Recommendation**: Use MODIS for full year range (2000–2022). For 1997–1999, use AVHRR or handle as missing.
- **What to download**: Monthly composite image patches for each district, for each month of Kharif season (June, July, August, September, October, November)
- **Image specification**:
  - Patch size: 64×64 pixels centered on district centroid
  - At MODIS 250m resolution: covers ~16km × 16km per patch
  - Bands to include: Red, NIR, Blue, SWIR1, NDVI (computed), EVI (computed)
  - Channels: 4-6 bands per image
- **Shape per sample**: (6, C, 64, 64) — 6 months × C channels × 64 × 64 spatial
- **Total images to download**: 859 district-year pairs × 6 months = ~5,150 images
- **Storage**: ~100-150 MB total (each image ~16-25 KB as compressed GeoTIFF or NPY)
- **Download method**: Use `ee.Image.clip()` with district boundary or buffer around centroid, export as NumPy arrays or GeoTIFF
- **Why images, not just NDVI mean**: Raw image patches capture within-district spatial heterogeneity — healthy crops in one area, stressed crops in another, water bodies, bare soil. A mean value destroys this spatial information.

### 2.3 NDVI Time Series (MODALITY 2 — Temporal Vegetation Signal)
- **Source**: Google Earth Engine (same MODIS product)
- **What to extract**: District-mean monthly NDVI for June, July, August, September, October, November
- **Shape per sample**: (6, 1) — 6 monthly NDVI values
- **This is SEPARATE from the image branch**: The NDVI time series is a clean, noise-free temporal signal showing the overall vegetation trajectory. The image branch captures spatial patterns. Both are needed.
- **District-year pairs**: 859 unique combinations from `district_year_lookup.csv`

### 2.4 Weather Data (MODALITY 3 — Meteorological Time Series)
- **Source**: Open-Meteo Historical Weather API (free, no key needed)
- **API endpoint**: `https://archive-api.open-meteo.com/v1/archive`
- **What to extract**: Weekly aggregates for Kharif season (June 1 – November 30):
  - temperature_mean (°C)
  - temperature_max (°C)
  - rainfall_sum (mm)
- **Shape per sample**: (22, 3) — approximately 22 weeks × 3 weather variables
- **Rate limiting**: Respect Open-Meteo rate limits, add delays between requests

### 2.5 Soil Data (MODALITY 4 — Pedological)
- **Source**: India Soil Health Card data / existing soil_ph.csv
- **Feature**: Soil pH per district (static, one value per district)
- **Shape**: scalar per sample

### 2.6 Final Feature Set Per Sample
```
MODALITY 1 — Satellite Images (Spatial):
  satellite_images: shape (6, C, 64, 64)  — 6 monthly image patches, C spectral bands

MODALITY 2 — NDVI Time Series (Temporal Vegetation):
  ndvi_sequence: shape (6, 1)  — monthly district-mean NDVI Jun–Nov

MODALITY 3 — Weather Time Series (Temporal Meteorological):
  weather_sequence: shape (22, 3)  — weekly temp_mean, temp_max, rainfall

MODALITY 4 — Static Features (Tabular):
  crop: categorical → learned embedding (dim=8)
  district: categorical → learned embedding (dim=16)
  year: normalized scalar
  soil_ph: normalized scalar

TARGET (Primary):
  yield_value: continuous (tonnes/hectare or bales/hectare)

TARGET (Secondary):
  stress_index: continuous (0–1), derived from physics equations
```

---

## 3. MODEL ARCHITECTURE

### 3.1 Overview
A unified multi-crop, multi-task model with FOUR input branches, physics-informed loss, and two output heads. This is genuinely multimodal — it processes raw satellite images (spatial), vegetation time series (temporal), weather sequences (temporal), and static tabular features, each with architecture appropriate to that data type.

### 3.2 Branch 1 — Satellite Image Branch (CNN + LSTM) — SPATIAL FEATURES
```
Input: (batch, 6, C, 64, 64)  — 6 monthly satellite image patches

Shared CNN applied to each month's image independently:
  Conv2D(C, 32, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool2d(2)    → (32, 32, 32)
  Conv2D(32, 64, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool2d(2)   → (64, 16, 16)
  Conv2D(64, 128, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool2d(2)  → (128, 8, 8)
  AdaptiveAvgPool2d(1) → (128,)
  Dropout(0.3)

Each monthly image → 128-dim feature vector
Stack 6 months → (batch, 6, 128)

Temporal aggregation via LSTM:
  LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
  Take final hidden state → (batch, 64)

Output: (batch, 64)
```
**Why this design**: The shared CNN extracts spatial features from each month's image (vegetation patterns, heterogeneity, land use). The LSTM then captures how these spatial features evolve over the growing season. Weight sharing across months keeps parameters manageable.

### 3.3 Branch 2 — NDVI Time Series Branch (Conv1D) — TEMPORAL VEGETATION SIGNAL
```
Input: (batch, 6, 1)  — monthly district-mean NDVI

→ Conv1D(in=1, out=32, kernel=3, padding=1) + ReLU + Dropout(0.3)
→ Conv1D(in=32, out=64, kernel=3, padding=1) + ReLU
→ AdaptiveAvgPool1d(1)
→ Flatten → Linear(64, 32)

Output: (batch, 32)
```
**Why keep this separate from images**: District-mean NDVI is a clean, noise-free temporal signal. The image branch captures spatial patterns but is noisy. Having both gives the model a reliable temporal backbone plus rich spatial features.

### 3.4 Branch 3 — Weather Temporal Branch (LSTM) — METEOROLOGICAL SIGNAL
```
Input: (batch, 22, 3)  — weekly temp_mean, temp_max, rainfall

→ LSTM(input_size=3, hidden_size=64, num_layers=1, batch_first=True)
→ Take final hidden state → (batch, 64)
→ Linear(64, 64)

Output: (batch, 64)
```
**Why LSTM**: 22 weekly timesteps is enough for LSTM to learn temporal dependencies like "prolonged heat followed by heavy rain" or "dry spell during critical growth stage."

### 3.5 Branch 4 — Static Feature Branch (Embeddings + Dense) — TABULAR FEATURES
```
Input: crop_id, district_id, year, soil_ph

→ crop_embedding = Embedding(num_crops=5, embedding_dim=8)
→ district_embedding = Embedding(num_districts=34, embedding_dim=16)
→ year_normalized = (year - 1997) / 25
→ soil_ph_normalized = standard scaling
→ Concat [crop_emb(8), district_emb(16), year(1), soil_ph(1)] → dim=26
→ Linear(26, 32) + ReLU

Output: (batch, 32)
```
**Why learned embeddings**: Districts that are agro-ecologically similar (e.g., Kolhapur and Sangli — both southern, similar rainfall) learn similar embedding vectors. Same for crops. The model discovers these relationships without manual encoding. Embeddings can be visualized via t-SNE for the paper.

### 3.6 Fusion Layer + Output Heads
```
Concat all branches:
  satellite_features (64) + ndvi_features (32) + weather_features (64) + static_features (32)
  = 192-dimensional fused vector

Fusion network:
→ Linear(192, 96) + ReLU + Dropout(0.3)
→ Linear(96, 48) + ReLU

Yield Prediction Head:
→ Linear(48, 1) → predicted_yield (unbounded)

Stress Prediction Head:
→ Linear(48, 1) + Sigmoid → stress_index (0 to 1)
```

### 3.7 Total Parameters (approximate)
```
Satellite CNN:      ~200K parameters (shared across 6 months)
Satellite LSTM:     ~50K
NDVI Conv1D:        ~6K
Weather LSTM:       ~17K
Static branch:      ~2K
Fusion + heads:     ~15K
Embeddings:         ~600
────────────────────────────
Total:              ~290K parameters
```
This is appropriate for ~3,240 training samples — roughly 11 samples per parameter with regularization.

---

## 4. PHYSICS-INFORMED NEURAL NETWORK (PINN) COMPONENTS

### 4.1 Total Loss Function
```
L_total = L_yield + λ₁ × L_growth + λ₂ × L_temperature + λ₃ × L_water + λ₄ × L_stress
```
The physics losses act as regularizers — they constrain the model to respect known agricultural science, preventing overfitting in a data-scarce setting.

### 4.2 L_yield — Standard Data-Driven Loss
```
L_yield = MSE(predicted_yield, actual_yield)
```

### 4.3 L_growth — Crop Growth Physics Constraint
Based on the established relationship: biomass accumulation is proportional to intercepted photosynthetically active radiation, which correlates with NDVI through Leaf Area Index.
```
LAI = -1/k × ln(1 - NDVI)          where k ≈ 0.5 (light extinction coefficient)
fIPAR = 1 - exp(-k × LAI)          fraction of intercepted PAR
growth_proxy = Σ(fIPAR_month) × RUE  summed over growing season

L_growth = MSE(normalized_predicted_yield, normalized_growth_proxy)
```
This penalizes the model when predicted yield is inconsistent with what the NDVI trajectory physically implies about biomass accumulation.

### 4.4 L_temperature — Thermal Stress Constraint
Every crop has a well-documented optimal temperature range. Growth rate follows a trapezoidal response curve:
```
Crop-specific thermal parameters:
  ┌──────────┬────────┬───────┬────────┐
  │ Crop     │ T_base │ T_opt │ T_ceil │
  ├──────────┼────────┼───────┼────────┤
  │ Rice     │ 10°C   │ 30°C  │ 42°C   │
  │ Jowar    │ 8°C    │ 32°C  │ 44°C   │
  │ Bajra    │ 10°C   │ 33°C  │ 45°C   │
  │ Soyabean │ 10°C   │ 28°C  │ 40°C   │
  │ Cotton   │ 15°C   │ 30°C  │ 40°C   │
  └──────────┴────────┴───────┴────────┘

thermal_response(T) =
  0                                    if T < T_base or T > T_ceil
  (T - T_base) / (T_opt - T_base)     if T_base ≤ T < T_opt
  (T_ceil - T) / (T_ceil - T_opt)     if T_opt ≤ T ≤ T_ceil

weekly_thermal_stress = 1 - thermal_response(T_weekly_mean)
season_thermal_stress = mean(weekly_thermal_stress)

L_temperature penalizes if model predicts high yield when temperatures
indicate significant thermal stress periods.
```

### 4.5 L_water — Water Stress Constraint (FAO Method)
Based on FAO Irrigation & Drainage Paper 33:
```
(1 - Ya/Ym) = Ky × (1 - ETa/ETm)

Ya = actual yield, Ym = maximum yield potential
Ky = crop-specific yield response factor to water deficit
ETa = actual evapotranspiration (approximated from rainfall using water balance)
ETm = maximum evapotranspiration (estimated via Hargreaves equation from temperature)

Hargreaves ET₀ = 0.0023 × (T_mean + 17.8) × (T_max - T_min)^0.5 × Ra
  where Ra = extraterrestrial radiation (function of latitude and day of year)

Crop Ky values (FAO):
  Rice=1.05, Jowar=0.9, Bajra=0.7, Soyabean=0.85, Cotton=0.85

L_water penalizes predictions that ignore water deficit signals in the rainfall data.
```

### 4.6 L_stress — Stress Prediction Loss
```
Combined stress label derivation:
  thermal_stress_score = season_thermal_stress        (from Section 4.4)
  water_stress_score = max(0, 1 - ETa/ETm)           (from Section 4.5)
  combined_stress = 0.4 × thermal_stress + 0.6 × water_stress  (water weighted higher)

L_stress = MSE(predicted_stress_index, combined_stress_label)
```

### 4.7 Lambda Hyperparameters
```
λ₁ (growth):       0.1  — tune range [0.01, 0.5]
λ₂ (temperature):  0.05 — tune range [0.01, 0.2]
λ₃ (water):        0.05 — tune range [0.01, 0.2]
λ₄ (stress):       0.1  — tune range [0.05, 0.3]

Tuning strategy: Start with these defaults. Use validation loss to find optimal balance.
Gradually increase physics lambdas during training (curriculum approach).
```

---

## 5. TRAINING STRATEGY

### 5.1 Data Split
- **5-fold cross-validation** stratified by (crop × district)
- Each fold preserves crop and district distribution
- Report mean ± std for all metrics across folds
- For temporal validation: also test with leave-last-3-years-out (2020-2022 as test)

### 5.2 Training Configuration
```
Optimizer:    AdamW (lr=1e-3, weight_decay=1e-4)
Scheduler:    CosineAnnealingLR (T_max=100)
Epochs:       200 (with early stopping, patience=20)
Batch size:   32 (smaller due to image data memory)
Dropout:      0.3 (after conv layers and fusion layers)
```

### 5.3 Data Augmentation (for satellite images)
```
Training time augmentations:
  - Random horizontal flip (p=0.5)
  - Random vertical flip (p=0.5)
  - Random rotation (±15°)
  - Gaussian noise on pixel values (σ=0.02)
  - Random brightness/contrast adjustment (±10%)
No augmentation on NDVI time series, weather, or static features.
```

### 5.4 Metrics
- **Overall**: R², MAE, RMSE
- **Per-crop**: R², MAE, RMSE for each of the 5 crops separately
- **Stress prediction**: MAE, R² on stress index
- **Statistical significance**: Report confidence intervals from 5-fold CV

---

## 6. EXPERIMENTS TO RUN (for paper)

### Experiment 1: Tabular Baseline (No Deep Learning)
```
Features: Flatten all temporal features into summary statistics
  NDVI:    mean, max, min, range, month_of_peak, slope
  Weather: mean_temp, max_temp, total_rainfall, rainfall_variance, dry_weeks_count, 
           max_consecutive_dry_weeks
  Image:   mean_pixel_ndvi, spatial_std_ndvi (simple image statistics)
  Static:  crop (one-hot), district (one-hot), year, soil_ph

Models: XGBoost, Random Forest, LightGBM
Purpose: Establish non-deep-learning baseline
```

### Experiment 2: Neural Multimodal Model (Without Physics, Without Images)
```
Architecture: NDVI Conv1D + Weather LSTM + Static Embeddings (3 branches)
Loss: Standard MSE only
Purpose: Show the value of temporal modeling vs tabular baseline
```

### Experiment 3: Neural Multimodal Model (Without Physics, With Images)
```
Architecture: Full 4-branch (Satellite CNN+LSTM, NDVI Conv1D, Weather LSTM, Static)
Loss: Standard MSE only
Purpose: Show the value of adding satellite image spatial features
```

### Experiment 4: PINN Model (With Physics Constraints)
```
Architecture: Full 4-branch + physics loss terms
Loss: L_yield + λ₁L_growth + λ₂L_temperature + λ₃L_water
Purpose: Show improvement from physics-informed regularization
```

### Experiment 5: PINN + Multi-task (Yield + Stress Prediction)
```
Architecture: Full 4-branch + physics loss + stress head
Loss: Full L_total (all terms)
Purpose: Show multi-task learning benefit + interpretable stress outputs
```

### Experiment 6: Ablation Study
```
Remove one modality at a time from the full model:
  A. No satellite images (remove Branch 1)
  B. No NDVI time series (remove Branch 2)
  C. No weather data (remove Branch 3)
  D. No soil pH (remove from Branch 4)
  E. No physics constraints (remove PINN losses)
  F. No crop/district embeddings (use one-hot instead)

Purpose: Quantify relative importance of each component
```

---

## 7. BACKEND (FastAPI) — PREDICTION PIPELINE

### 7.1 API Endpoint
```
GET /predict?crop=rice&district=kolhapur&year=2026
```

### 7.2 Prediction Flow
```
1. Receive request (crop, district, year)
2. Fetch real-time satellite image patch from GEE for latest available months
3. Fetch real-time NDVI monthly time series from GEE → shape (6,)
4. Fetch real-time weather weekly data from Open-Meteo → shape (22, 3)
5. Look up soil pH for district → scalar
6. Encode crop → embedding lookup
7. Encode district → embedding lookup
8. Normalize year, soil_ph
9. Forward pass through PINN model
10. Return JSON response
```

### 7.3 Response Format
```json
{
  "crop": "Rice",
  "district": "Kolhapur",
  "year": 2026,
  "predicted_yield": 1.85,
  "yield_unit": "Tonnes/Hectare",
  "stress_index": 0.32,
  "stress_level": "Moderate",
  "stress_factors": {
    "thermal_stress": 0.15,
    "water_stress": 0.49
  },
  "confidence": "Medium",
  "ndvi_profile": [0.25, 0.42, 0.58, 0.61, 0.45, 0.30],
  "model": "PINN-Multimodal-v2"
}
```

### 7.4 CORS & Frontend
- CORS configured for frontend origin
- Frontend: React-based UI (Lovable/custom)
- Shows: yield prediction, stress gauge, NDVI curve, weather summary, satellite image thumbnails

---

## 8. KEY RESEARCH CONTRIBUTIONS

1. **Physics-Informed Multimodal Fusion**: First application of PINN to district-level crop yield prediction in Maharashtra that genuinely integrates satellite imagery (spatial), vegetation indices (temporal), weather (temporal), and soil data (tabular) with agricultural physics constraints.

2. **Genuine Image Processing Pipeline**: Unlike prior work that reduces satellite data to single statistics, this system processes raw multi-band image patches through CNNs to capture within-district spatial vegetation heterogeneity.

3. **Learned Agro-Ecological Embeddings**: District and crop embeddings that capture latent geographic and agronomic similarities — visualizable via t-SNE, showing that geographically proximate or climatically similar districts cluster together.

4. **Cross-Crop Transfer Learning**: Unified single model that borrows environmental learning across crops, improving predictions for data-scarce crop-district combinations compared to separate per-crop models.

5. **Crop Stress Diagnostics**: Multi-task prediction of yield and stress index from physics-derived labels, providing interpretable and actionable outputs beyond point predictions.

6. **Climate-Driven vs Management-Driven Crop Analysis**: Systematic quantitative evidence showing environmental remote sensing data predicts climate-sensitive crops (Rice, Jowar, Bajra) well but struggles with management-intensive crops (Cotton, Soyabean).

---

## 9. EXPECTED RESULTS

### 9.1 Model Performance (estimated)
```
┌─────────────────────────────┬────────────┬───────┬───────┬───────┬──────────┬────────┐
│ Model                       │ Overall R² │ Rice  │ Jowar │ Bajra │ Soyabean │ Cotton │
├─────────────────────────────┼────────────┼───────┼───────┼───────┼──────────┼────────┤
│ Exp 1: Tabular (XGBoost)    │ 0.55-0.65  │ 0.60  │ 0.65  │ 0.70  │ 0.35     │ 0.25   │
│ Exp 2: Neural (no images)   │ 0.58-0.68  │ 0.65  │ 0.68  │ 0.72  │ 0.38     │ 0.28   │
│ Exp 3: Neural (with images) │ 0.62-0.70  │ 0.68  │ 0.72  │ 0.75  │ 0.40     │ 0.30   │
│ Exp 4: PINN                 │ 0.65-0.73  │ 0.72  │ 0.75  │ 0.80  │ 0.44     │ 0.33   │
│ Exp 5: PINN + Stress        │ 0.66-0.74  │ 0.73  │ 0.76  │ 0.81  │ 0.45     │ 0.34   │
└─────────────────────────────┴────────────┴───────┴───────┴───────┴──────────┴────────┘
```

### 9.2 Key Figures for Paper
1. System architecture diagram (showing all 4 branches + fusion + PINN)
2. Sample satellite image patches showing spatial vegetation patterns
3. NDVI temporal profiles by crop and district
4. Model comparison bar chart (R² per crop per experiment)
5. t-SNE of learned district embeddings (colored by geographic region)
6. t-SNE of learned crop embeddings
7. Physics loss convergence curves (showing PINN loss decreasing)
8. Stress index vs actual yield scatter plot (showing correlation)
9. Ablation study results (bar chart of R² when removing each modality)
10. Predicted vs actual yield scatter plots per crop
11. Spatial map of prediction errors across Maharashtra districts
12. Feature importance / attention visualization

---

## 10. TECHNOLOGY STACK

```
Language:           Python 3.10+
Deep Learning:      PyTorch (model, training, PINN)
Image Processing:   torchvision (transforms, augmentation)
ML Baselines:       scikit-learn, XGBoost, LightGBM
Satellite Data:     Google Earth Engine Python API (ee), geemap
Weather Data:       Open-Meteo API (requests library)
Geospatial:         rasterio, geopandas (for image handling)
Backend:            FastAPI + Uvicorn
Frontend:           React (Lovable UI)
Visualization:      matplotlib, seaborn, plotly
Data:               pandas, numpy
Experiment Tracking: (optional) wandb or tensorboard
```

---

## 11. DISTRICT COORDINATES (for data fetching)

```json
{
  "Ahilyanagar": [19.0948, 74.7480],
  "Akola": [20.7002, 77.0082],
  "Amravati": [20.9374, 77.7796],
  "Beed": [18.9891, 75.7601],
  "Bhandara": [21.1669, 79.6436],
  "Buldhana": [20.5293, 76.1843],
  "Chandrapur": [19.9500, 79.2961],
  "Chhatrapati Sambhajinagar": [19.8762, 75.3433],
  "Dharashiv": [18.1807, 76.0447],
  "Dhule": [20.9042, 74.7749],
  "Gadchiroli": [20.1057, 80.0000],
  "Gondia": [21.4602, 80.1920],
  "Hingoli": [19.7173, 77.1510],
  "Jalgaon": [21.0077, 75.5626],
  "Jalna": [19.8347, 75.8804],
  "Kolhapur": [16.7050, 74.2433],
  "Latur": [18.3968, 76.5604],
  "Nagpur": [21.1458, 79.0882],
  "Nanded": [19.1383, 77.3210],
  "Nandurbar": [21.3690, 74.2394],
  "Nashik": [20.0000, 73.7800],
  "Palghar": [19.6967, 72.7699],
  "Parbhani": [19.2610, 76.7784],
  "Pune": [18.5204, 73.8567],
  "Raigad": [18.5158, 73.1822],
  "Ratnagiri": [16.9944, 73.3000],
  "Sangli": [16.8524, 74.5815],
  "Satara": [17.6805, 74.0183],
  "Sindhudurg": [16.3489, 73.7556],
  "Solapur": [17.6599, 75.9064],
  "Thane": [19.2183, 72.9781],
  "Wardha": [20.7453, 78.6022],
  "Washim": [20.1041, 77.1478],
  "Yavatmal": [20.3899, 78.1307]
}
```

---

## 12. IMPORTANT NOTES & CONSTRAINTS

- **Cotton yield unit**: Bales/Hectare (1 bale = 170 kg). Model handles different scales via crop embedding — no manual conversion needed.
- **Missing data**: Not all districts grow all crops. Bajra has 534 samples (26 districts), Cotton has 643 (28 districts). The model handles this naturally through the unified architecture.
- **MODIS temporal range**: MODIS data starts from February 2000. For years 1997-1999 (~300 samples), options are: (a) use AVHRR NDVI (lower resolution), (b) exclude these years, (c) fill with seasonal NDVI climatology. Recommendation: exclude 1997-1999 for image branch, keep NDVI mean from alternative source.
- **Sentinel-2 option**: If higher resolution images are desired, Sentinel-2 (10m) is available from 2015. Could train a separate high-res model for 2015-2022 period as an additional experiment.
- **Weather API**: Open-Meteo historical data goes back to 1940. No API key needed, but respect rate limits (add 0.5s delay between requests).
- **Data-scarce setting**: ~3,240 total samples. Every architecture choice must respect this: heavy regularization (dropout=0.3, weight decay=1e-4, early stopping), simple CNN (3 layers, not ResNet-50), physics constraints as inductive bias, data augmentation for images.
- **Year normalization**: Normalize as (year - 1997) / (2022 - 1997) to [0, 1] range.
- **Image preprocessing**: Normalize each band to [0, 1] using per-band min-max or z-score normalization. Handle cloud-masked pixels (set to 0 or interpolate).

---

## 13. DATA FETCHING SPECIFICATIONS

### 13.1 GEE — Satellite Image Download Script Requirements
```
For each (district, year) in district_year_lookup.csv:
  For each month in [June, July, August, September, October, November]:
    1. Create point geometry from (latitude, longitude)
    2. Create buffer region (~8km radius for 64×64 at 250m)
    3. Filter MODIS MOD13Q1 collection by date range (month start to month end)
    4. Compute monthly composite (median to remove clouds)
    5. Select bands: ['sur_refl_b01', 'sur_refl_b02', 'NDVI', 'EVI']
       (Red, NIR, NDVI, EVI — 4 channels)
    6. Clip to buffer region
    7. Export as 64×64 NumPy array
    8. Save as: data/processed/satellite_images/{district}_{year}_{month}.npy

  Also extract district-mean NDVI for time series:
    Save as row in: data/processed/ndvi_timeseries.csv
    Columns: district, year, ndvi_jun, ndvi_jul, ndvi_aug, ndvi_sep, ndvi_oct, ndvi_nov
```

### 13.2 Open-Meteo — Weather Download Script Requirements
```
For each (district, year) in district_year_lookup.csv:
  1. Call Open-Meteo archive API with:
     latitude, longitude
     start_date = {year}-06-01
     end_date = {year}-11-30
     daily variables: temperature_2m_mean, temperature_2m_max, precipitation_sum
  2. Aggregate daily → weekly (sum rainfall, mean temperature)
  3. Pad/trim to exactly 22 weeks
  4. Save as row in: data/processed/weather_timeseries.csv
     Columns: district, year, week_1_temp_mean, week_1_temp_max, week_1_rain, 
              week_2_temp_mean, ..., week_22_rain
```

---

## 14. PIPELINE EXECUTION ORDER

```
Step 1:  ✅ DONE — Clean yield data → maharashtra_kharif_yield_clean.csv (3,240 rows)
Step 2:  NEXT  — Extract satellite image patches + NDVI time series from GEE
Step 3:  TODO  — Fetch weekly weather data from Open-Meteo for all district-year pairs
Step 4:  TODO  — Map soil pH per district
Step 5:  TODO  — Merge all features into final_multimodal_dataset (+ save image arrays)
Step 6:  TODO  — Train tabular baseline (XGBoost/RF) — Experiment 1
Step 7:  TODO  — Train neural multimodal without images — Experiment 2
Step 8:  TODO  — Train neural multimodal with images — Experiment 3
Step 9:  TODO  — Add PINN physics loss — Experiment 4
Step 10: TODO  — Add stress prediction multi-task head — Experiment 5
Step 11: TODO  — Run ablation study — Experiment 6
Step 12: TODO  — Generate all figures, tables, results
Step 13: TODO  — Build FastAPI backend with real-time prediction
Step 14: TODO  — Connect frontend, deploy
Step 15: TODO  — Write research paper
```

---

## 15. PROJECT FOLDER STRUCTURE

```
crop-yield-prediction/
├── data/
│   ├── raw/
│   │   └── horizontal_crop_vertical_year_report.xls
│   ├── processed/
│   │   ├── maharashtra_kharif_yield_clean.csv
│   │   ├── district_year_lookup.csv
│   │   ├── ndvi_timeseries.csv
│   │   ├── weather_timeseries.csv
│   │   ├── soil_ph.csv
│   │   ├── final_multimodal_dataset.csv
│   │   └── satellite_images/          ← 64×64 .npy files per district/year/month
│   └── district_coordinates.json
├── notebooks/                         ← Jupyter notebooks for each step
├── src/
│   ├── data/                          ← Data loading, fetching, dataset classes
│   ├── models/                        ← All model architectures
│   ├── physics/                       ← PINN loss functions, stress computation
│   ├── training/                      ← Training loop, evaluation, cross-validation
│   └── utils/                         ← Config, visualization, helpers
├── api/                               ← FastAPI backend
├── frontend/                          ← React UI
├── results/                           ← Figures, tables, logs
├── paper/                             ← Research manuscript
├── scripts/                           ← Pipeline automation scripts
├── requirements.txt
└── README.md
```

---

*This prompt (v2) contains everything needed to build the entire project from scratch, including the satellite image processing pipeline. Any AI assistant or collaborator can use this to understand the full system, reproduce results, or continue development from any step.*rfdsXCZsdXC