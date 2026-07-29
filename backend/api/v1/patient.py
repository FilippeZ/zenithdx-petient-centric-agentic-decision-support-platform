# backend/api/v1/patient.py
from __future__ import annotations

import os
import json
import shutil
import traceback
import datetime as dt
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Form, File, UploadFile
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
                shutil.copy2(path, dest)
                updated[key] = str(dest)
            except Exception as e:
                print(f"Error copying {path} to {dest}: {e}")
                updated[key] = path
        else:
            updated[key] = path
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

    original_xray = updated_img_paths.get("original_xray")
    gradcam_overlay = updated_img_paths.get("gradcam_overlay")
    captum_image = updated_img_paths.get("captum_image")
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
            return {"data": data, "total": len(data)}
    except Exception as e:
        raise HTTPException(500, f"DB error: {e}")
    finally:
        conn.close()
