# 🌾 AgroPINN — Physics-Informed Multimodal Crop Yield Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

**District-level Kharif crop yield prediction for Maharashtra using Physics-Informed Neural Networks (PINNs), satellite imagery, weather time series, and learnable spatial embeddings.**

[Architecture](#architecture) · [Experiments](#experiments) · [Setup](#setup) · [API](#api) · [Results](#results)

</div>

---

## 📌 Overview

AgroPINN is an end-to-end deep learning system that predicts district-level crop yields across Maharashtra for five Kharif crops (Rice, Jowar, Bajra, Soyabean, Cotton). It fuses four heterogeneous data sources through a unified multimodal neural architecture, regularised by physics-derived loss terms from agronomic science.

**Key features:**
- 🛰️ **Satellite CNN-LSTM** — processes monthly 32×32 MODIS image patches (Jun–Nov)
- 🌦️ **Weather LSTM** — 22 weeks of daily temperature & rainfall sequences
- 📈 **NDVI Time Series** — Conv1D branch capturing seasonal vegetation dynamics
- 🗺️ **Learnable Embeddings** — 35 districts × 5 crops encoded as spatial lookup tables
- ⚛️ **Physics Loss** — Monteith growth, DSSAT thermal, FAO Hargreaves water balance
- 🎯 **Multi-task Head** — simultaneous yield + crop stress index prediction (Exp5)
- 🌐 **React + FastAPI UI** — 3-input prediction interface (crop, district, year)

---

## 🏗️ Architecture

```
Satellite Images (N, 6, 32, 32, 1)               NDVI Sequence (N, 6, 1)
  └─ CNN [Conv2D×3 → GAP] per month                └─ Conv1D(32,64) → GAP → Dense(32)
  └─ LSTM(64) ──────────────────────────────────────────────────────────┐
                                                                         │
Weather (N, 22, 3)                                                       │
  └─ BiLSTM(64) → Dense(64) ───────────────────────────────────────────►├─ Concat(192)
                                                                         │   Dense(96,48)
Static: crop_emb(8) + dist_emb(16) + year_norm + soil_pH                │       ↙      ↘
  └─ Dense(32) ────────────────────────────────────────────────────────►┘  Yield(1)  Stress(1,σ)
```

**Physics loss suite:**

| Term | Equation | Source |
|---|---|---|
| `L_growth` | NDVI → LAI → fIPAR (RUE) | Monteith (1977) |
| `L_temperature` | Trapezoidal thermal response | DSSAT crop models |
| `L_water` | Hargreaves ETo → ETa/ETm deficit | FAO-56 (Doorenbos & Kassam 1979) |
| `L_stress` | 0.4×thermal + 0.6×water | Multi-task target (Exp5) |

**Full loss:**  
`L_total = L_yield + λ₁L_growth + λ₂L_temp + λ₃L_water + λ₄L_stress`

---

## 📂 Project Structure

```
Crop Yield Prediction/
├── api/                             # FastAPI backend
│   ├── main.py                      # Routes: POST /api/predict, GET /api/districts, etc.
│   ├── schemas.py                   # Pydantic request/response models
│   ├── config.py                    # District coordinates + DISTRICT_CROPS map
│   ├── models/                      # Trained model weights + preprocessing PKL
│   │   ├── experiment5_pinn_multitask.keras
│   │   └── experiment5_preprocessing.pkl
│   └── services/
│       ├── ndvi_service.py          # NDVI lookup / summarization
│       ├── weather_service.py       # Open-Meteo live fetch + stress computation
│       ├── soil_service.py          # Soil pH lookup
│       ├── satellite_service.py     # Satellite image loading + CNN features
│       ├── stress_service.py        # Stress level labels + descriptions
│       └── prediction_service.py   # Model loading, inference, confidence, yield range
│
├── src/
│   ├── data/
│   │   ├── fetch_ndvi.py            # GEE MODIS NDVI download (monthly)
│   │   ├── fetch_weather.py         # Open-Meteo historical weather download
│   │   ├── fetch_satellite.py       # GEE satellite image patch export
│   │   └── merge_features.py        # Builds final_dataset.csv
│   ├── models/
│   │   ├── multimodal_net_exp3.py   # 4-branch architecture (Exp3/4)
│   │   ├── multimodal_net_exp5.py   # + multi-task stress head (Exp5)
│   │   └── physics_loss.py          # All physics label computations
│   ├── training/
│   │   ├── train_experiment1.py     # XGBoost / Random Forest tabular baseline
│   │   ├── train_experiment2.py     # Neural (no satellite images)
│   │   ├── train_experiment3.py     # Neural + satellite images
│   │   ├── train_experiment4.py     # PINN (custom gradient tape loop)
│   │   ├── train_experiment5.py     # PINN + multi-task stress head
│   │   └── train_experiment6.py     # Ablation study (6 configurations)
│   └── utils/
│       └── generate_results.py      # Figures + LaTeX/CSV tables for paper
│
├── frontend/                        # React UI
│   ├── public/index.html
│   └── src/
│       ├── App.js                   # 3-input form, arc gauge, NDVI chart, weather summary
│       ├── index.css                # Dark glassmorphism design system
│       └── index.js
│
├── data/
│   ├── raw/                         # Original yield CSV from Maharashtra Govt
│   └── processed/
│       ├── final_dataset.csv        # Merged features (yield + NDVI + weather + soil)
│       ├── satellite_images/        # Per-district per-month .npy image patches
│       └── satellite_cnn_features.npy  # Pre-computed CNN features (cache)
│
├── results/
│   ├── figures/                     # All generated paper figures (PNG)
│   └── tables/                      # OOF metric CSV + LaTeX tables
│
├── research_paper.md                # Full research paper draft
├── requirements.txt
└── README.md
```

---

## 📊 Experiments

| # | Name | Key Addition | OOF R² (est.) |
|---|---|---|---|
| Exp1 | Tabular Baseline | XGBoost / Random Forest | ~0.58 |
| Exp2 | Neural (No Images) | NDVI Conv1D + Weather LSTM | ~0.63 |
| Exp3 | Neural + Images | + Satellite CNN-LSTM branch | ~0.67 |
| Exp4 | PINN | + Physics loss (L1+L2+L3) | ~0.71 |
| **Exp5** | **PINN + Multi-task** | **+ Stress head (L4)** | **~0.72** |
| Exp6 | Ablation Study | Remove one component at a time | — |

*Actual values are in `results/tables/model_metrics_exp5.csv` after training.*

---

## ⚙️ Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/<your-username>/crop-yield-prediction.git
cd "crop-yield-prediction"

# Create conda environment
conda create -n agropinn python=3.10 -y
conda activate agropinn

# Install backend dependencies
pip install -r requirements.txt
```

### 2. Requirements (`requirements.txt`)

```
tensorflow>=2.13
fastapi>=0.110
uvicorn[standard]
pydantic>=2.0
joblib
scikit-learn
pandas
numpy
httpx
matplotlib
seaborn
openpyxl
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

---

## 🚀 Running the Application

### Start the Backend

```bash
# From project root
uvicorn api.main:app --port 8000
```

Backend will be at `http://localhost:8000`

### Start the Frontend

```bash
cd frontend
npm start
```

Frontend will be at `http://localhost:3000`

> ⚠️ **Model Required:** The backend needs a trained model in `api/models/`. Run Exp5 training first (see below), or place your `.keras` and `.pkl` files there manually.

---

## 🏋️ Training

Run experiments in order. Each saves metrics to `results/tables/` and the final model to `api/models/`.

```bash
# Exp1 — Tabular Baseline
python src/training/train_experiment1.py

# Exp2 — Neural (no satellite)
python src/training/train_experiment2.py

# Exp3 — Neural + Images (requires satellite_images/)
python src/training/train_experiment3.py

# Exp4 — PINN
python src/training/train_experiment4.py

# Exp5 — PINN + Multi-task (saves final model to api/models/)
python src/training/train_experiment5.py

# Exp6 — Ablation Study
python src/training/train_experiment6.py
```

**Resume a interrupted training:**
```bash
python src/training/train_experiment5.py --resume --epochs 20
```

**Generate all paper figures and tables:**
```bash
python src/utils/generate_results.py
```

---

## 🌐 API Reference

### `POST /api/predict`

**Request:**
```json
{
  "crop": "Rice",
  "district": "Kolhapur",
  "year": 2024
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "predicted_yield": 1.85,
    "yield_unit": "Tonnes/Hectare",
    "yield_range": { "low": 1.57, "high": 2.13 }
  },
  "stress": {
    "overall_index": 0.32,
    "level": "Moderate",
    "thermal_stress": 0.15,
    "water_stress": 0.44
  },
  "ndvi_profile": {
    "months": ["June","July","August","September","October","November"],
    "values": [0.25, 0.42, 0.58, 0.61, 0.45, 0.30],
    "peak_month": "September",
    "health_status": "Good"
  },
  "weather_summary": { "avg_temperature": 28.5, "total_rainfall": 820.5, ... },
  "confidence": { "level": "High", "score": 0.82, "factors": [...] },
  "metadata": { "model_version": "PINN-Multimodal-v2", "processing_time_ms": 340 }
}
```

### Supporting Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Model load status |
| `GET` | `/api/crops` | List of 5 Kharif crops |
| `GET` | `/api/districts` | All 34 districts with coordinates + available crops |
| `GET` | `/api/districts/{district}/crops` | Crops grown in a specific district |
| `GET` | `/api/results` | All experiment OOF metrics (for Results page) |
| `GET` | `/embeddings/districts` | t-SNE coordinates of district embeddings |

Interactive API docs at `http://localhost:8000/docs`

---

## 📈 Results

After training, view results at `http://localhost:3000` → **Results** tab, or run:

```bash
python src/utils/generate_results.py
# Outputs: results/figures/*.png + results/tables/*.csv + results/tables/*.tex
```

---

## 📄 Data Sources

| Data | Source | Coverage |
|---|---|---|
| Crop yield | Maharashtra Government Agricultural Census | 1997–2022, 35 districts, 5 crops |
| Satellite NDVI | MODIS MOD13Q1 (250m) via Google Earth Engine | 2000–2022 |
| Weather | Open-Meteo Historical Archive (ERA5-Land) | 1997–2022 |
| Soil pH | HWSD District-Level Averages | Static per district |

---

## 📖 Research Paper

A complete research paper draft is included at [`research_paper.md`](./research_paper.md), covering:
- Abstract, Introduction, Related Work
- Dataset description (all 4 sources)
- Model architecture with equations
- All 5 physics loss derivations (LaTeX-formatted)
- 6 experiment descriptions + ablation study
- Results tables (to be filled from actual training)
- Discussion, limitations, future work, references

---

## 🗺️ Districts Covered

All 34 Maharashtra districts: Ahilyanagar, Akola, Amravati, Beed, Bhandara, Buldhana, Chandrapur, Chhatrapati Sambhajinagar, Dharashiv, Dhule, Gadchiroli, Gondia, Hingoli, Jalgaon, Jalna, Kolhapur, Latur, Nagpur, Nanded, Nandurbar, Nashik, Palghar, Parbhani, Pune, Raigad, Ratnagiri, Sangli, Satara, Sindhudurg, Solapur, Thane, Wardha, Washim, Yavatmal.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Monteith (1977)](https://doi.org/10.1098/rstb.1977.0140) — Radiation-use efficiency growth model
- [FAO-56](https://www.fao.org/3/x0490e/x0490e00.htm) — Crop water requirements (Doorenbos & Kassam 1979)
- [DSSAT](https://dssat.net/) — Thermal response crop parameters
- [Raissi et al. (2019)](https://doi.org/10.1016/j.jcp.2018.10.045) — Physics-Informed Neural Networks
- [Google Earth Engine](https://earthengine.google.com/) — MODIS satellite data access
- [Open-Meteo](https://open-meteo.com/) — Free historical weather archive (ERA5-Land)
