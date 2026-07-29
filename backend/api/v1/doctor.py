# backend/api/v1/doctor.py
from __future__ import annotations

import os
import json
import base64
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Body, Response
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor

from database.connection import get_db_connection
from api.v1.auth import get_current_user
from api.v1.patient import json_safe

router = APIRouter(tags=["Doctor"])

def _doctor_only(user: dict):
    if user.get("user_type") != "doctor" and user.get("role") != "doctor":
        raise HTTPException(403, "Doctor credentials required")

class StatusPayload(BaseModel):
    status: str
    reason: Optional[str] = None

class SavePayload(BaseModel):
    diagnosis: Optional[str] = None
    symptoms: Optional[str] = None
    doctor_message: Optional[str] = None

class EditPayload(BaseModel):
    diagnosis: str

class DoctorMessageRequest(BaseModel):
    doctor_message: str

@router.get("/doctor/reports")
def doctor_reports(curr: dict = Depends(get_current_user)):
    _doctor_only(curr)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT p.id        AS report_id,
                          u.full_name AS patient_name,
                          p.image_path, p.symptoms,
                          p.submission_date, p.status,
                          p.diagnosis
                   FROM patients p
                   JOIN users u ON p.user_id=u.id
                   ORDER BY p.submission_date DESC"""
            )
            rows = cur.fetchall()
            return {"data": [dict(r) for r in rows]}
    finally:
        conn.close()

@router.get("/doctor/reports/{report_id}")
def doctor_report(report_id: str, curr: dict = Depends(get_current_user)):
    _doctor_only(curr)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT p.id         AS report_id,
                          u.full_name  AS patient_name,
                          u.id         AS patient_id,
                          p.image_path,
                          p.symptoms,
                          p.submission_date,
                          p.status,
                          p.diagnosis,
                          p.xai_structured,
                          p.doctor_message,
                          p.rating,
                          p.original_xray,
                          p.gradcam_overlay,
                          p.captum_image,
                          p.classification_results
                   FROM patients p
                   JOIN users u ON p.user_id=u.id
                   WHERE p.id::text=%s""",
                (str(report_id),)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Report not found")
            row = dict(row)
    finally:
        conn.close()

    xai_data = {}
    if row.get("xai_structured"):
        try:
            xai_data = json_safe(json.loads(row["xai_structured"]))
        except Exception:
            xai_data = {}

    patient_overview = {
        "report_id": str(row["report_id"]),
        "patient_name": row["patient_name"],
        "patient_id": str(row["patient_id"]),
        "submission_date": str(row["submission_date"]) if row["submission_date"] else None,
        "symptoms": row["symptoms"],
        "status": row["status"],
        "doctor_message": row.get("doctor_message"),
        "rating": row.get("rating"),
        "image_path": row["image_path"],
    }
    diagnosis_report = row["diagnosis"] or ""

    return {
        "patient_overview": patient_overview,
        "diagnosis": diagnosis_report,
        "diagnosis_report": diagnosis_report,
        "original_xray": row.get("original_xray") or xai_data.get("original_xray"),
        "gradcam_overlay": row.get("gradcam_overlay") or xai_data.get("gradcam_overlay"),
        "captum_image": row.get("captum_image") or xai_data.get("captum_image"),
        "mask_image": xai_data.get("mask_image"),
        "classification_results": row.get("classification_results") or xai_data.get("classification_results"),
        **{k: v for k, v in (xai_data or {}).items()
            if k not in {
                "original_xray", "gradcam_overlay", "captum_image",
                "mask_image", "classification_results"
            }
        },
        "xai_structured": row.get("xai_structured"),
    }

@router.put("/doctor/reports/{report_id}")
def save_report(
    report_id: str,
    payload: SavePayload = Body(...),
    curr: dict = Depends(get_current_user)
):
    _doctor_only(curr)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE patients
                   SET diagnosis = COALESCE(%s, diagnosis),
                       symptoms = COALESCE(%s, symptoms),
                       doctor_message = COALESCE(%s, doctor_message),
                       status = 'Edited'
                   WHERE id::text = %s RETURNING id;""",
                (payload.diagnosis, payload.symptoms, payload.doctor_message, str(report_id))
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Report not found")
            conn.commit()
            return {"message": "Report updated successfully", "report_id": report_id}
    finally:
        conn.close()

@router.patch("/doctor/reports/{report_id}/status")
def update_status_route(report_id: str, payload: StatusPayload = Body(...), curr: dict = Depends(get_current_user)):
    _doctor_only(curr)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE patients SET status = %s WHERE id::text = %s RETURNING id;", (payload.status, str(report_id)))
            if not cur.fetchone():
                raise HTTPException(404, "Report not found")
            conn.commit()
            return {"message": f"Report status set to {payload.status}", "report_id": report_id}
    finally:
        conn.close()

@router.patch("/doctor/reports/{report_id}/approve")
def approve_report_route(report_id: str, curr: dict = Depends(get_current_user)):
    _doctor_only(curr)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE patients SET status = 'Approved' WHERE id::text = %s RETURNING id;", (str(report_id),))
            if not cur.fetchone():
                raise HTTPException(404, "Report not found")
            conn.commit()
            return {"message": "Report approved", "report_id": report_id}
    finally:
        conn.close()

@router.patch("/doctor/reports/{report_id}/reject")
def reject_report_route(report_id: str, payload: Optional[StatusPayload] = Body(None), curr: dict = Depends(get_current_user)):
    _doctor_only(curr)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE patients SET status = 'Rejected' WHERE id::text = %s RETURNING id;", (str(report_id),))
            if not cur.fetchone():
                raise HTTPException(404, "Report not found")
            conn.commit()
            return {"message": "Report rejected", "report_id": report_id}
    finally:
        conn.close()
