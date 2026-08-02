# backend/api/v1/doctor.py
from __future__ import annotations

import os
import io
import json
import base64
import sys
import re
import html
import traceback
import qrcode
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Body, Response
from pydantic import BaseModel

from config import settings
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

def path_to_base64_data_uri(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    path_str = str(path)
    if path_str.startswith("data:image"):
        return path_str
    
    target_path = path_str
    if path_str.startswith("/outputs/"):
        target_path = str(settings.OUTPUT_DIR / path_str.replace("/outputs/", ""))
    elif path_str.startswith("/uploads/"):
        target_path = str(settings.UPLOAD_DIR / path_str.replace("/uploads/", ""))
        
    if os.path.exists(target_path) and os.path.isfile(target_path):
        try:
            with open(target_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(target_path)[1].lstrip(".").lower() or "png"
                return f"data:image/{ext};base64,{encoded}"
        except Exception:
            pass
    return path_str

def format_web_path(path: Optional[str]) -> Optional[str]:
    """Converts local filesystem paths into web-servable relative URLs (/outputs/...) or data URIs."""
    if not path:
        return None
    path_str = str(path).replace("\\", "/")
    if path_str.startswith("data:image") or path_str.startswith("http://") or path_str.startswith("https://"):
        return path_str
    
    if "outputs/" in path_str:
        parts = path_str.split("outputs/")
        return f"/outputs/{parts[-1]}"
    elif "uploads/" in path_str:
        parts = path_str.split("uploads/")
        return f"/uploads/{parts[-1]}"
        
    if path_str.startswith("/outputs") or path_str.startswith("/uploads"):
        return path_str

    b64 = path_to_base64_data_uri(path)
    if b64 and b64.startswith("data:image"):
        return b64

    return path_str

def clean_markdown_for_reportlab(text: str) -> str:
    """Converts markdown tags into valid ReportLab XML (<b>, <i>) ensuring proper closing tags and XML escaping."""
    if not text:
        return ""
    # Strip headers #
    clean_text = re.sub(r"^#+\s*", "", str(text)).strip()
    # Escape XML entities
    clean_text = html.escape(clean_text)
    # Convert **bold** to <b>bold</b> safely
    parts = clean_text.split("**")
    res = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            res.append(f"<b>{part}</b>")
        else:
            res.append(part)
    return "".join(res)


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
def get_doctor_report(report_id: str, curr: dict = Depends(get_current_user)):
    _doctor_only(curr)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT p.id::text AS report_id,
                          u.full_name  AS patient_name,
                          p.user_id::text AS patient_id,
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
        "image_path": format_web_path(row["image_path"]),
    }
    diagnosis_report = row["diagnosis"] or ""

    orig_img = format_web_path(row.get("original_xray") or xai_data.get("original_xray"))
    grad_img = format_web_path(row.get("gradcam_overlay") or xai_data.get("gradcam_overlay"))
    grad_seg = format_web_path(xai_data.get("gradcam_segmented"))
    capt_img = format_web_path(row.get("captum_image") or xai_data.get("captum_image"))
    mask_img = format_web_path(xai_data.get("mask_image"))

    return {
        "patient_overview": patient_overview,
        "diagnosis": diagnosis_report,
        "diagnosis_report": diagnosis_report,
        "original_xray": orig_img,
        "gradcam_overlay": grad_img,
        "gradcam_segmented": grad_seg,
        "captum_image": capt_img,
        "mask_image": mask_img,
        "classification_results": row.get("classification_results") or xai_data.get("classification_results"),
        **{k: v for k, v in (xai_data or {}).items()
            if k not in {
                "original_xray", "gradcam_overlay", "gradcam_segmented", "captum_image",
                "mask_image", "classification_results"
            }
        },
        "xai_structured": row.get("xai_structured"),
    }

from pipelines.pdf_generator import generate_pdf_report_bytes

@router.get("/doctor/reports/{report_id}/pdf")
def export_doctor_report_pdf(report_id: str, curr: dict = Depends(get_current_user)):
    """Generates and downloads a structured PDF medical report for the doctor."""
    _doctor_only(curr)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT p.id::text AS report_id,
                          u.full_name  AS patient_name,
                          p.user_id::text AS patient_id,
                          p.symptoms,
                          p.submission_date,
                          p.status,
                          p.diagnosis,
                          p.doctor_message,
                          p.original_xray,
                          p.gradcam_overlay,
                          p.captum_image,
                          p.xai_structured,
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

    # Enrich row with history_text and gradcam_segmented from xai_structured JSON blob
    xai_data = {}
    if row.get("xai_structured"):
        try:
            xai_data = json.loads(row["xai_structured"]) if isinstance(row["xai_structured"], str) else row["xai_structured"]
        except Exception:
            pass
    # History text — only inject if it was actually retrieved by the agent
    if xai_data.get("history_retrieved") and xai_data.get("history_text"):
        row["history_text"] = xai_data["history_text"]
    else:
        row["history_text"] = None
    # Segmented Grad-CAM overlay from xai blob
    if not row.get("gradcam_segmented"):
        row["gradcam_segmented"] = xai_data.get("gradcam_segmented")

    try:
        pdf_bytes = generate_pdf_report_bytes(row, is_patient_view=False)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=ZenithDx_Doctor_Report_{str(report_id)[:8]}.pdf"}
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[PDF Error] {e}\n{tb}", file=sys.stderr)
        raise HTTPException(500, f"PDF generation error: {e}")

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
