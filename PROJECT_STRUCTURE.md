# Multimodal Crop Yield Prediction System - Project Structure

```
crop-yield-prediction/
│
├── README.md                              # Project overview, setup instructions
├── requirements.txt                       # Python dependencies
├── .env                                   # API keys (GEE, if needed)
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── horizontal_crop_vertical_year_report.xls   # Original govt data
│   │
│   ├── processed/
│   │   ├── maharashtra_kharif_yield_clean.csv          # Step 1 output: cleaned yield data
│   │   ├── district_year_lookup.csv                     # 859 district-year pairs
│   │   ├── ndvi_timeseries.csv                          # Step 2 output: monthly NDVI (Jun-Nov)
│   │   ├── weather_timeseries.csv                       # Step 3 output: weekly weather
│   │   ├── soil_ph.csv                                  # Soil pH per district
│   │   └── final_multimodal_dataset.csv                 # Step 4 output: merged features + yield
│   │
│   └── district_coordinates.json                        # Lat/Lon for 34 districts
│
├── notebooks/
│   ├── 01_data_exploration.ipynb           # EDA on raw yield data
│   ├── 02_ndvi_extraction.ipynb            # GEE NDVI extraction walkthrough
│   ├── 03_weather_extraction.ipynb         # Open-Meteo API calls
│   ├── 04_feature_engineering.ipynb        # Merging + feature creation
│   ├── 05_baseline_models.ipynb            # XGBoost / RF tabular baseline
│   ├── 06_neural_multimodal.ipynb          # PyTorch multimodal model
│   ├── 07_pinn_model.ipynb                 # Physics-informed neural network
│   ├── 08_stress_prediction.ipynb          # Multi-task stress analysis
│   └── 09_results_analysis.ipynb           # Final comparisons, visualizations
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── clean_yield_data.py             # Step 1: Parse & clean govt XLS
│   │   ├── fetch_ndvi.py                   # Step 2: GEE NDVI time series extraction
│   │   ├── fetch_weather.py                # Step 3: Open-Meteo API weather fetching
│   │   ├── merge_features.py              # Step 4: Combine all modalities
│   │   └── dataset.py                      # PyTorch Dataset class for multimodal data
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_tabular.py             # XGBoost / Random Forest baseline
│   │   ├── multimodal_net.py               # Neural multimodal fusion model
│   │   ├── pinn_model.py                   # Physics-informed neural network
│   │   └── stress_head.py                  # Multi-task stress prediction head
│   │
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── crop_growth.py                  # Biomass accumulation equation (dB/dt)
│   │   ├── thermal_stress.py               # Temperature stress function per crop
│   │   ├── water_stress.py                 # FAO water stress (Ky coefficients)
│   │   └── stress_index.py                 # Combined stress index computation
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                      # Training loop with physics loss
│   │   ├── evaluator.py                    # Metrics: R², MAE, RMSE per crop
│   │   └── cross_validation.py             # Stratified K-fold (crop × district)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                       # Hyperparameters, crop constants
│       ├── embeddings.py                   # Learned embedding utilities
│       └── visualization.py                # t-SNE plots, yield maps, trend charts
│
├── api/
│   ├── main.py                             # FastAPI application
│   ├── schemas.py                          # Pydantic request/response models
│   ├── services/
│   │   ├── ndvi_service.py                 # Real-time NDVI fetch for prediction
│   │   ├── weather_service.py              # Real-time weather fetch
│   │   └── soil_service.py                 # Soil pH lookup
│   └── models/                             # Saved model artifacts
│       ├── multimodal_model.pt             # Trained PyTorch model
│       ├── pinn_model.pt                   # Trained PINN model
│       ├── xgboost_baseline.pkl            # Trained XGBoost model
│       ├── district_encoder.pkl            # District label encoder
│       ├── crop_encoder.pkl                # Crop label encoder
│       └── scalers.pkl                     # Feature scalers
│
├── frontend/                               # React/Lovable UI
│   └── (existing Lovable frontend code)
│
├── results/
│   ├── figures/
│   │   ├── model_comparison.png            # R² comparison across models
│   │   ├── crop_wise_performance.png       # Per-crop analysis
│   │   ├── district_embeddings_tsne.png    # t-SNE of learned embeddings
│   │   ├── ndvi_profiles.png               # Sample NDVI curves
│   │   ├── stress_vs_yield.png             # Stress index correlation
│   │   ├── pinn_vs_standard.png            # Physics loss improvement
│   │   └── yield_trend_predictions.png     # Predicted vs actual over years
│   │
│   ├── tables/
│   │   ├── model_metrics.csv               # All model results
│   │   └── ablation_study.csv              # Feature importance analysis
│   │
│   └── logs/
│       └── training_logs/                  # TensorBoard or CSV training logs
│
├── paper/
│   ├── manuscript.docx                     # Research paper
│   ├── figures/                            # Publication-quality figures
│   └── references.bib                      # Bibliography
│
└── scripts/
    ├── run_pipeline.sh                     # End-to-end pipeline script
    ├── train_all_models.py                 # Train baseline + neural + PINN
    └── generate_results.py                 # Produce all figures and tables
```

## Key Design Decisions

1. **src/ is modular** — each step (data, models, physics, training) is self-contained
2. **notebooks/ mirror src/** — each notebook uses src/ modules, good for experimentation
3. **api/ is deployment-ready** — separate from training code
4. **physics/ is standalone** — crop growth equations can be tested independently
5. **results/ captures everything** — figures, tables, logs for paper writing
