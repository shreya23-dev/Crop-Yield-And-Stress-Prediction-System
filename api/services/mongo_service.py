from __future__ import annotations

import hashlib
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, MongoClient


_MONGO_CLIENT: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    global _MONGO_CLIENT
    if _MONGO_CLIENT is None:
        uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        _MONGO_CLIENT = MongoClient(uri)
    return _MONGO_CLIENT


def _get_db():
    db_name = os.getenv("MONGO_DB_NAME", "agropinn")
    return _get_client()[db_name]


def get_mongo_db_name() -> str:
    return os.getenv("MONGO_DB_NAME", "agropinn")


def init_mongo_indexes() -> None:
    db = _get_db()
    db.farmers.create_index([("name", ASCENDING)], unique=True)
    db.sessions.create_index([("token", ASCENDING)], unique=True)
    db.sessions.create_index([("created_at", DESCENDING)])
    db.prediction_history.create_index([("farmer_id", ASCENDING), ("created_at", DESCENDING)])


def ensure_database_initialized() -> Dict[str, Any]:
    """Ensure DB exists by pinging server, creating indexes, and touching a meta doc."""
    client = _get_client()
    client.admin.command("ping")

    init_mongo_indexes()

    db = _get_db()
    db["_meta"].update_one(
        {"_id": "bootstrap"},
        {
            "$set": {
                "service": "agropinn",
                "updated_at": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
        },
        upsert=True,
    )

    return {
        "db_name": db.name,
        "collections": sorted(db.list_collection_names()),
    }


def _hash_password(password: str, salt: str) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return digest.hex()


def create_farmer(name: str, password: str) -> Dict[str, Any]:
    db = _get_db()
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Name cannot be empty")

    existing = db.farmers.find_one({"name": clean_name})
    if existing:
        raise ValueError("Farmer already registered")

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    doc = {
        "name": clean_name,
        "password_hash": password_hash,
        "salt": salt,
        "created_at": datetime.now(timezone.utc),
    }
    inserted = db.farmers.insert_one(doc)
    doc["_id"] = inserted.inserted_id
    return doc


def authenticate_farmer(name: str, password: str) -> Optional[Dict[str, Any]]:
    db = _get_db()
    farmer = db.farmers.find_one({"name": name.strip()})
    if not farmer:
        return None

    expected = _hash_password(password, farmer.get("salt", ""))
    if expected != farmer.get("password_hash"):
        return None
    return farmer


def create_session(farmer_id: ObjectId) -> str:
    db = _get_db()
    token = secrets.token_urlsafe(32)
    db.sessions.insert_one({
        "token": token,
        "farmer_id": farmer_id,
        "created_at": datetime.now(timezone.utc),
    })
    return token


def get_farmer_by_token(token: str) -> Optional[Dict[str, Any]]:
    db = _get_db()
    sess = db.sessions.find_one({"token": token})
    if not sess:
        return None
    return db.farmers.find_one({"_id": sess["farmer_id"]})


def save_prediction_history(
    farmer_id: ObjectId,
    crop: str,
    district: str,
    year: int,
    predicted_yield: float,
    yield_unit: str,
    stress_level: str,
    stress_index: float,
) -> None:
    db = _get_db()
    db.prediction_history.insert_one({
        "farmer_id": farmer_id,
        "crop": crop,
        "district": district,
        "year": int(year),
        "predicted_yield": float(predicted_yield),
        "yield_unit": yield_unit,
        "stress_level": stress_level,
        "stress_index": float(stress_index),
        "created_at": datetime.now(timezone.utc),
    })


def get_prediction_history(farmer_id: ObjectId, limit: int = 50) -> list[Dict[str, Any]]:
    db = _get_db()
    cursor = db.prediction_history.find({"farmer_id": farmer_id}).sort("created_at", DESCENDING).limit(limit)
    return list(cursor)


if __name__ == "__main__":
    try:
        info = ensure_database_initialized()
        print(f"Mongo DB initialized: {info['db_name']}")
        print("Collections:", ", ".join(info["collections"]))
    except Exception as e:
        print("Mongo initialization failed:", str(e))
        raise
