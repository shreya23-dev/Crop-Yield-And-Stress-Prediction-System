from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel):
    crop: str = Field(..., description="Crop name", example="Rice")
    district: str = Field(..., description="District name", example="Kolhapur")
    year: int = Field(..., description="Prediction year", ge=2000, le=2030, example=2024)

class YieldPrediction(BaseModel):
    crop: str
    district: str
    year: int
    predicted_yield: float
    yield_unit: str
    yield_range: dict  # {"low": float, "high": float}

class StressInfo(BaseModel):
    overall_index: float
    level: str  # "Low", "Moderate", "High", "Severe"
    thermal_stress: float
    water_stress: float
    description: str

class NDVIProfile(BaseModel):
    months: List[str]
    values: List[float]
    peak_month: str
    health_status: str  # "Poor", "Fair", "Good", "Excellent"

class WeatherSummary(BaseModel):
    avg_temperature: float
    max_temperature: float
    total_rainfall: float
    rainfall_unit: str
    dry_weeks: int
    description: str

class ConfidenceInfo(BaseModel):
    level: str  # "Low", "Medium", "High"
    score: float
    factors: List[str]

class PredictResponse(BaseModel):
    status: str
    prediction: YieldPrediction
    stress: StressInfo
    ndvi_profile: NDVIProfile
    weather_summary: WeatherSummary
    confidence: ConfidenceInfo
    metadata: dict

class CropListResponse(BaseModel):
    crops: List[str]

class DistrictInfo(BaseModel):
    name: str
    lat: float
    lon: float
    available_crops: List[str]

class DistrictListResponse(BaseModel):
    districts: List[DistrictInfo]


class AuthRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=50, example="Rahul")
    password: str = Field(..., min_length=4, max_length=128, example="secret123")


class FarmerInfo(BaseModel):
    id: str
    name: str


class AuthResponse(BaseModel):
    status: str
    token: str
    farmer: FarmerInfo


class PredictionHistoryItem(BaseModel):
    id: str
    crop: str
    district: str
    year: int
    predicted_yield: float
    yield_unit: str
    stress_level: str
    stress_index: float
    created_at: str


class PredictionHistoryResponse(BaseModel):
    items: List[PredictionHistoryItem]
