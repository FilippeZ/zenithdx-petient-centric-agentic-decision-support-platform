# backend/database/models.py
from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr

class UserModel(BaseModel):
    id: Optional[int] = None
    full_name: str
    username: str
    email: EmailStr
    hashed_password: str
    user_type: str  # 'doctor' or 'patient'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class PatientRecordModel(BaseModel):
    id: Optional[int] = None
    user_id: int
    image_path: str
    symptoms: str
    submission_date: Optional[datetime] = None
    status: str = "Pending"
    diagnosis: Optional[str] = None
    treatment: Optional[str] = None
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    filepath_1: Optional[str] = None
    filepath_2: Optional[str] = None
    filepath_3: Optional[str] = None

class DiagnosticReportModel(BaseModel):
    id: Optional[int] = None
    patient_id: int
    doctor_id: Optional[int] = None
    user_query: str
    diagnosis_markdown: str
    xai_report_markdown: Optional[str] = None
    classification_results: Optional[dict] = None
    status: str = "Pending"  # Pending, Approved, Rejected, Edited
    doctor_notes: Optional[str] = None
    created_at: Optional[datetime] = None
