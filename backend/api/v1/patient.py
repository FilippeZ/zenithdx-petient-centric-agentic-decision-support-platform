# backend/api/v1/patient.py
from __future__ import annotations

import os
import sys
import io
import json
import shutil
import traceback
import qrcode
import datetime as dt
from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Form, File, UploadFile, Response
from fastapi.concurrency import run_in_threadpool

from config import settings
from database.connection import get_db_connection
from api.v1.auth import get_current_user
from ai_agent_runner import run_agent

router = APIRouter(tags=["Patient"])

def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe(x) for x in obj]
    elif isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)

def pick_diagnosis(agent_out: dict) -> str:
    if isinstance(agent_out, dict):
        if agent_out.get("diagnosis"):
            return agent_out["diagnosis"]
        ao = agent_out.get("agent_outcome")
        if isinstance(ao, dict) and ao.get("diagnosis"):
            return ao["diagnosis"]
    return "[No diagnosis generated]"

def pick_xai_report(agent_out: dict) -> str:
    if isinstance(agent_out, dict):
        if agent_out.get("xai_report"):
            return agent_out["xai_report"]
        ao = agent_out.get("agent_outcome")
        if isinstance(ao, dict) and ao.get("xai_report"):
            return ao["xai_report"]
    return "No XAI report available."

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

def save_xai_images_to_timestamp_folder(user_id: int, image_dict: dict):
    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = settings.OUTPUT_DIR / str(user_id) / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)
    updated = {}
    for key, path in image_dict.items():
        if path and os.path.exists(path):
            fname = os.path.basename(path)
            dest = out_dir / fname
            try:
                if str(path) != str(dest):
                    shutil.copy2(path, dest)
                rel_path = dest.relative_to(settings.OUTPUT_DIR).as_posix()
                updated[key] = f"/outputs/{rel_path}"
            except Exception as e:
                print(f"Error copying {path} to {dest}: {e}")
                updated[key] = format_web_path(path)
        elif path and str(path).startswith(("data:image", "/outputs", "/uploads", "http")):
            updated[key] = path
        else:
            updated[key] = format_web_path(path)
    return updated, str(out_dir)

@router.post("/upload")
async def upload_and_process_report(
    symptoms: str = Form(...),
    file: Optional[UploadFile] = File(None),
    patient_id: Optional[str] = Form(None),
    current: dict = Depends(get_current_user)
):
    if current.get("user_type") != "patient" and current.get("role") != "patient":
        raise HTTPException(403, "Only patients can upload")

    user_id = current.get("id") or current.get("user_id")
    file_path = None
    if file is not None:
        user_dir = settings.OUTPUT_DIR / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        file_path = str(user_dir / file.filename)
        try:
            contents = await file.read()
            with open(file_path, "wb") as bf:
                bf.write(contents)
        except Exception as e:
            raise HTTPException(500, f"Failed to save uploaded file: {e}")

    def agent_job():
        return run_agent(
            user_query=symptoms,
            image_path=file_path,
            patient_id=patient_id or None
        )

    try:
        agent_out = await run_in_threadpool(agent_job)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[patient] Agent failure: {e}\n{tb}")
        raise HTTPException(500, f"AI agent failed: {e}")

    _out = agent_out.get("agent_outcome", agent_out) or {}
    xai_img_keys = [
        "original_xray", "gradcam_overlay", "captum_image"
    ] + [k for k in _out if k.startswith("captum_")]
    xai_img_paths = {k: _out.get(k) for k in xai_img_keys if _out.get(k)}

    if xai_img_paths:
        updated_img_paths, xai_subfolder = save_xai_images_to_timestamp_folder(user_id, xai_img_paths)
    else:
        updated_img_paths, xai_subfolder = {}, None

    for k, v in updated_img_paths.items():
        _out[k] = v

    xai_structured = json.dumps(json_safe(_out), ensure_ascii=False)
    diagnosis_md = pick_diagnosis(agent_out)
    xai_report_md = pick_xai_report(agent_out)

    original_xray = format_web_path(updated_img_paths.get("original_xray"))
    gradcam_overlay = format_web_path(updated_img_paths.get("gradcam_overlay"))
    captum_image = format_web_path(updated_img_paths.get("captum_image"))
    classification_results = agent_out.get("classification_results") or (_out.get("classification_results") if isinstance(_out, dict) else []) or []

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO patients
                   (user_id, image_path, symptoms, diagnosis, xai_report, xai_structured,
                    original_xray, gradcam_overlay, captum_image, classification_results,
                    submission_date, status)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),'Pending')
                   RETURNING id""",
                (
                    user_id, file_path or "", symptoms, diagnosis_md, xai_report_md, xai_structured,
                    original_xray, gradcam_overlay, captum_image, json.dumps(classification_results)
                )
            )
            row = cur.fetchone()
            new_id = row["id"] if isinstance(row, dict) else (row[0] if isinstance(row, (tuple, list)) else row)
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(400, f"DB insert failed: {e}")
    finally:
        conn.close()

    return {
        "message": "Report uploaded",
        "report_id": new_id,
        "diagnosis_preview": diagnosis_md[:100],
        "xai_report": xai_report_md,
        "original_xray": original_xray,
        "gradcam_overlay": gradcam_overlay,
        "captum_image": captum_image,
        "classification_results": classification_results,
        "xai_structured": xai_structured[:300],
        "xai_images_subfolder": xai_subfolder
    }

@router.get("/patient/reports")
def patient_reports(curr: dict = Depends(get_current_user)):
    role = curr.get("user_type") or curr.get("role") or ""
    if role != "patient":
        raise HTTPException(403, "Forbidden")
    user_id = curr.get("id") or curr.get("user_id")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    p.id::text AS report_id,
                    image_path,
                    symptoms,
                    submission_date,
                    status,
                    diagnosis,
                    doctor_message,
                    gradcam_overlay,
                    original_xray
                FROM patients p
                WHERE user_id = %s
                ORDER BY submission_date DESC
                """,
                (user_id,)
            )
            rows = cur.fetchall()
            data = [dict(r) for r in rows]
            for d in data:
                d["original_xray"] = format_web_path(d.get("original_xray"))
                d["gradcam_overlay"] = format_web_path(d.get("gradcam_overlay"))
            return {"data": data, "total": len(data)}
    except Exception as e:
        raise HTTPException(500, f"DB error: {e}")
    finally:
        conn.close()

class FeedbackPayload(BaseModel):
    rating: int

@router.get("/patient/reports/{report_id}")
def get_patient_report_detail(report_id: str, curr: dict = Depends(get_current_user)):
    user_id = curr.get("id") or curr.get("user_id")
    role = (curr.get("user_type") or curr.get("role") or "").lower()
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if role == "doctor":
                query = """SELECT p.id::text AS report_id, p.id, u.full_name AS patient_name,
                                  p.user_id::text AS patient_id, p.image_path, p.symptoms,
                                  p.submission_date, p.status, p.diagnosis, p.xai_structured,
                                  p.doctor_message, p.rating, p.original_xray, p.gradcam_overlay,
                                  p.captum_image, p.classification_results
                           FROM patients p JOIN users u ON p.user_id=u.id
                           WHERE p.id::text=%s"""
                cur.execute(query, (str(report_id),))
            else:
                query = """SELECT p.id::text AS report_id, p.id, u.full_name AS patient_name,
                                  p.user_id::text AS patient_id, p.image_path, p.symptoms,
                                  p.submission_date, p.status, p.diagnosis, p.xai_structured,
                                  p.doctor_message, p.rating, p.original_xray, p.gradcam_overlay,
                                  p.captum_image, p.classification_results
                           FROM patients p JOIN users u ON p.user_id=u.id
                           WHERE p.id::text=%s AND p.user_id=%s"""
                cur.execute(query, (str(report_id), user_id))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Report not found")
            row = dict(row)
    finally:
        conn.close()

    row["original_xray"] = format_web_path(row.get("original_xray"))
    row["gradcam_overlay"] = format_web_path(row.get("gradcam_overlay"))
    row["captum_image"] = format_web_path(row.get("captum_image"))
    row["image_path"] = format_web_path(row.get("image_path"))

    return row

@router.post("/patient/reports/{report_id}/feedback")
def submit_patient_feedback(report_id: str, payload: FeedbackPayload, curr: dict = Depends(get_current_user)):
    user_id = curr.get("id") or curr.get("user_id")
    role = (curr.get("user_type") or curr.get("role") or "").lower()
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if role == "doctor":
                cur.execute(
                    "UPDATE patients SET rating = %s WHERE id::text = %s RETURNING id;",
                    (payload.rating, str(report_id))
                )
            else:
                cur.execute(
                    "UPDATE patients SET rating = %s WHERE id::text = %s AND user_id = %s RETURNING id;",
                    (payload.rating, str(report_id), user_id)
                )
            if not cur.fetchone():
                raise HTTPException(404, "Report not found or permission denied")
            conn.commit()
            return {"message": "Feedback recorded", "rating": payload.rating}
    finally:
        conn.close()

from pipelines.pdf_generator import generate_pdf_report_bytes

@router.get("/patient/reports/{report_id}/pdf")
def export_patient_report_pdf(report_id: str, curr: dict = Depends(get_current_user)):
    user_id = curr.get("id") or curr.get("user_id")
    role = (curr.get("user_type") or curr.get("role") or "").lower()
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if role == "doctor":
                query = """SELECT p.id::text AS report_id, u.full_name AS patient_name,
                                  p.user_id::text AS patient_id, p.symptoms, p.submission_date,
                                  p.status, p.diagnosis, p.doctor_message, p.original_xray,
                                  p.gradcam_overlay, p.captum_image, p.xai_structured,
                                  p.classification_results
                           FROM patients p JOIN users u ON p.user_id=u.id
                           WHERE p.id::text=%s"""
                cur.execute(query, (str(report_id),))
            else:
                query = """SELECT p.id::text AS report_id, u.full_name AS patient_name,
                                  p.user_id::text AS patient_id, p.symptoms, p.submission_date,
                                  p.status, p.diagnosis, p.doctor_message, p.original_xray,
                                  p.gradcam_overlay, p.captum_image, p.xai_structured,
                                  p.classification_results
                           FROM patients p JOIN users u ON p.user_id=u.id
                           WHERE p.id::text=%s AND p.user_id=%s"""
                cur.execute(query, (str(report_id), user_id))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Report not found")
            row = dict(row)
    finally:
        conn.close()

    xai_data = {}
    if row.get("xai_structured"):
        try:
            xai_data = json.loads(row["xai_structured"]) if isinstance(row["xai_structured"], str) else row["xai_structured"]
        except Exception:
            pass
    if xai_data.get("history_retrieved") and xai_data.get("history_text"):
        row["history_text"] = xai_data["history_text"]
    else:
        row["history_text"] = None
    if not row.get("gradcam_segmented"):
        row["gradcam_segmented"] = xai_data.get("gradcam_segmented")

    try:
        pdf_bytes = generate_pdf_report_bytes(row, is_patient_view=True)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=ZenithDx_Patient_Report_{str(report_id)[:8]}.pdf"}
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[PDF Error] {e}\n{tb}", file=sys.stderr)
        raise HTTPException(500, f"PDF generation error: {e}")
