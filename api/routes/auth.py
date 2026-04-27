from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException

from api.schemas import AuthRequest, AuthResponse, FarmerInfo
from api.services.mongo_service import (
    authenticate_farmer,
    create_farmer,
    create_session,
    get_farmer_by_token,
)

router = APIRouter(prefix="/api/auth", tags=["Auth"])
router_compat = APIRouter(prefix="/auth", tags=["Auth"], include_in_schema=False)


@router.post("/register", response_model=AuthResponse)
def register_farmer(payload: AuthRequest):
    try:
        farmer = create_farmer(payload.name, payload.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    token = create_session(farmer["_id"])
    return {
        "status": "success",
        "token": token,
        "farmer": FarmerInfo(id=str(farmer["_id"]), name=farmer["name"]),
    }
@router.post("/login", response_model=AuthResponse)
def login_farmer(payload: AuthRequest):
    farmer = authenticate_farmer(payload.name, payload.password)
    if not farmer:
        raise HTTPException(status_code=401, detail="Invalid name or password")

    token = create_session(farmer["_id"])
    return {
        "status": "success",
        "token": token,
        "farmer": FarmerInfo(id=str(farmer["_id"]), name=farmer["name"]),
    }


@router.get("/me", response_model=FarmerInfo)
def me(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1].strip()
    farmer = get_farmer_by_token(token)
    if not farmer:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return FarmerInfo(id=str(farmer["_id"]), name=farmer["name"])


@router_compat.post("/register", response_model=AuthResponse)
def register_farmer_compat(payload: AuthRequest):
    return register_farmer(payload)


@router_compat.post("/login", response_model=AuthResponse)
def login_farmer_compat(payload: AuthRequest):
    return login_farmer(payload)


@router_compat.get("/me", response_model=FarmerInfo)
def me_compat(authorization: str = Header(default="")):
    return me(authorization)
