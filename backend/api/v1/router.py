# backend/api/v1/router.py
from __future__ import annotations

from fastapi import APIRouter
from api.v1.auth import router as auth_router
from api.v1.patient import router as patient_router
from api.v1.doctor import router as doctor_router

api_v1_router = APIRouter()

api_v1_router.include_router(auth_router)
api_v1_router.include_router(patient_router)
api_v1_router.include_router(doctor_router)
