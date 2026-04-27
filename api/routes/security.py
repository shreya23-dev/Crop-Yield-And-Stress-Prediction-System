from __future__ import annotations

from fastapi import Header, HTTPException

from api.services.mongo_service import get_farmer_by_token


def require_farmer(authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1].strip()
    farmer = get_farmer_by_token(token)
    if not farmer:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return farmer
