from __future__ import annotations

from fastapi import APIRouter, Depends

from api.schemas import PredictionHistoryResponse
from api.routes.security import require_farmer
from api.services.mongo_service import get_prediction_history

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get("", response_model=PredictionHistoryResponse)
def list_history(farmer=Depends(require_farmer)):
    rows = get_prediction_history(farmer["_id"], limit=100)
    items = [
        {
            "id": str(r.get("_id")),
            "crop": r.get("crop", ""),
            "district": r.get("district", ""),
            "year": int(r.get("year", 0)),
            "predicted_yield": float(r.get("predicted_yield", 0.0)),
            "yield_unit": r.get("yield_unit", "Tonnes/Hectare"),
            "stress_level": r.get("stress_level", "Unknown"),
            "stress_index": float(r.get("stress_index", 0.0)),
            "created_at": r.get("created_at").isoformat() if r.get("created_at") else "",
        }
        for r in rows
    ]
    return {"items": items}
