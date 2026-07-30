# backend/api/v1/patient.py
from __future__ import annotations

import os
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
    role = curr.get("user_type") or curr.get("role") or ""
    if role != "patient":
        raise HTTPException(403, "Forbidden")
    user_id = curr.get("id") or curr.get("user_id")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT p.id::text AS report_id,
                          p.id,
                          u.full_name AS patient_name,
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
                   WHERE p.id::text=%s AND p.user_id=%s""",
                (str(report_id), user_id)
            )
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
    role = curr.get("user_type") or curr.get("role") or ""
    if role != "patient":
        raise HTTPException(403, "Forbidden")
    user_id = curr.get("id") or curr.get("user_id")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
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

@router.get("/patient/reports/{report_id}/pdf")
def export_patient_report_pdf(report_id: str, curr: dict = Depends(get_current_user)):
    user_id = curr.get("id") or curr.get("user_id")
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
                   WHERE p.id::text=%s AND p.user_id=%s""",
                (str(report_id), user_id)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Report not found")
            row = dict(row)
    finally:
        conn.close()

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        story = []

        wallpaper_path = str(settings.BASE_DIR / "assets" / "wallpaper.jpg")
        logo_path = str(settings.BASE_DIR / "assets" / "logo.png")

        def draw_background(canvas, doc_obj):
            canvas.saveState()
            if os.path.exists(wallpaper_path):
                try:
                    canvas.drawImage(wallpaper_path, 0, 0, width=612, height=792, preserveAspectRatio=False)
                except Exception:
                    pass
            canvas.restoreState()

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#0f172a'), leading=22, spaceAfter=4)
        subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#2563eb'), leading=12)
        heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#1e1b4b'), leading=16, spaceBefore=10, spaceAfter=4)
        bold_body_style = ParagraphStyle('BoldBodyStyle', parent=styles['Normal'], fontSize=9.5, textColor=colors.HexColor('#0f172a'), leading=14)
        body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#334155'), leading=13)

        brand_flow = []
        if os.path.exists(logo_path):
            try:
                brand_flow.append(RLImage(logo_path, width=100, height=32))
            except Exception:
                pass
        brand_flow.append(Paragraph("<b>ZenithDx</b>", title_style))
        brand_flow.append(Paragraph("Patient Diagnostic Clinical Report", subtitle_style))

        date_str = str(row['submission_date'])[:10] if row.get('submission_date') else "Today"
        right_flow = [
            Paragraph(f"<b>Report ID:</b> #{str(row['report_id'])[:8]}", body_style),
            Paragraph(f"<b>Date:</b> {date_str}", body_style),
            Paragraph(f"<b>Status:</b> {row['status']}", body_style)
        ]

        header_table = Table([[brand_flow, right_flow]], colWidths=[360, 180])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (1,0), (1,0), 'RIGHT'),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 10))

        meta_data = [
            [Paragraph("<b>Patient Name</b>", bold_body_style), Paragraph(str(row['patient_name']), body_style),
             Paragraph("<b>Patient ID</b>", bold_body_style), Paragraph(str(row['patient_id']), body_style)],
            [Paragraph("<b>Reported Symptoms</b>", bold_body_style), Paragraph(str(row['symptoms'] or "—"), body_style),
             Paragraph("<b>Current Status</b>", bold_body_style), Paragraph(str(row['status']), body_style)]
        ]
        meta_table = Table(meta_data, colWidths=[110, 160, 90, 180])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 10))

        story.append(Paragraph("<b>Structured Multi-Modal Diagnosis Report</b>", heading_style))
        diag_raw = row['diagnosis'] or "No diagnosis report generated."
        diag_lines = diag_raw.split("\n")
        for line in diag_lines:
            line_str = line.strip()
            if not line_str:
                story.append(Spacer(1, 3))
            elif line_str.startswith("#"):
                clean_h = line_str.replace("#", "").strip()
                story.append(Paragraph(f"<b>{clean_h}</b>", heading_style))
            elif line_str.startswith("- ") or line_str.startswith("* ") or line_str.startswith("1.") or line_str.startswith("2."):
                formatted_l = line_str.replace("**", "<b>").replace("**", "</b>")
                story.append(Paragraph(f"• {formatted_l}", body_style))
            else:
                formatted_l = line_str.replace("**", "<b>").replace("**", "</b>")
                story.append(Paragraph(formatted_l, body_style))
        story.append(Spacer(1, 10))

        if row.get('doctor_message'):
            story.append(Paragraph("<b>Doctor's Message to Patient</b>", heading_style))
            story.append(Paragraph(str(row['doctor_message']), body_style))
            story.append(Spacer(1, 10))

        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=3, border=0)
        qr.add_data(f"ZenithDx Patient Report #{row['report_id']}")
        qr.make(fit=True)
        qr_buf = io.BytesIO()
        qr.make_image(fill_color="black", back_color="white").save(qr_buf, "PNG")
        qr_buf.seek(0)
        
        story.append(Spacer(1, 10))
        qr_img = RLImage(qr_buf, width=50, height=50)
        footer_table = Table([[qr_img, Paragraph("<i>Confidential Medical Report — ZenithDx Clinical AI Suite</i>", body_style)]], colWidths=[60, 480])
        footer_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(footer_table)

        doc.build(story, onFirstPage=draw_background, onLaterPages=draw_background)
        pdf_data = buffer.getvalue()
        buffer.close()

        return Response(
            content=pdf_data,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=ZenithDx_Report_{str(report_id)[:8]}.pdf"}
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[PDF Error] {e}\n{tb}", file=sys.stderr)
        raise HTTPException(500, f"PDF generation error: {e}")
