# backend/api/v1/doctor.py
from __future__ import annotations

import os
import io
import json
import base64
import sys
import re
import html
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

@router.get("/doctor/reports/{report_id}/pdf")
def export_doctor_report_pdf(report_id: str, curr: dict = Depends(get_current_user)):
    """Generates and downloads a structured PDF medical report with background image, logo, bold headings, and XAI maps."""
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

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        story = []

        # Background Canvas Callback Function
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

        # Styles
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#0f172a'), leading=22, spaceAfter=4)
        subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#2563eb'), leading=12)
        heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=12, textColor=colors.HexColor('#1e1b4b'), leading=16, spaceBefore=10, spaceAfter=4)
        bold_body_style = ParagraphStyle('BoldBodyStyle', parent=styles['Normal'], fontSize=9.5, textColor=colors.HexColor('#0f172a'), leading=14)
        body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#334155'), leading=13)

        # Header Table with Logo & Title
        header_data = []
        brand_flow = []
        if os.path.exists(logo_path):
            try:
                brand_flow.append(RLImage(logo_path, width=100, height=32))
            except Exception:
                pass
        brand_flow.append(Paragraph("<b>ZenithDx</b>", title_style))
        brand_flow.append(Paragraph("Patient-Centric Agentic Decision Support Platform", subtitle_style))

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

        # Patient Metadata Table Card
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

        # Structured Diagnosis Report
        story.append(Paragraph("<b>Structured Multi-Modal Diagnosis Report</b>", heading_style))
        diag_raw = row['diagnosis'] or "No diagnosis report generated."
        diag_lines = diag_raw.split("\n")
        for line in diag_lines:
            line_str = line.strip()
            if not line_str:
                story.append(Spacer(1, 3))
            elif line_str.startswith("#"):
                clean_h = clean_markdown_for_reportlab(line_str)
                story.append(Paragraph(f"<b>{clean_h}</b>", heading_style))
            elif line_str.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.")):
                clean_l = clean_markdown_for_reportlab(line_str)
                story.append(Paragraph(f"• {clean_l}", body_style))
            else:
                clean_l = clean_markdown_for_reportlab(line_str)
                story.append(Paragraph(clean_l, body_style))
        story.append(Spacer(1, 10))

        # Attending Physician Notes
        if row.get('doctor_message'):
            story.append(Paragraph("<b>Attending Physician Notes</b>", heading_style))
            story.append(Paragraph(str(row['doctor_message']), body_style))
            story.append(Spacer(1, 10))

        # Multi-Label Pathology Scores Table
        cls_raw = row.get("classification_results")
        if cls_raw:
            try:
                cls_list = json.loads(cls_raw) if isinstance(cls_raw, str) else cls_raw
                if isinstance(cls_list, list) and len(cls_list) > 0:
                    story.append(Paragraph("<b>Multi-Label ResNet-50 Pathology Prediction Scores</b>", heading_style))
                    table_rows = [[Paragraph("<b>Pathology</b>", bold_body_style), Paragraph("<b>Probability</b>", bold_body_style)]]
                    for item in cls_list:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            lbl, prob = item[0], float(item[1])
                            table_rows.append([Paragraph(str(lbl), body_style), Paragraph(f"<b>{int(prob*100)}%</b>", body_style)])
                    cls_table = Table(table_rows, colWidths=[270, 270])
                    cls_table.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eff6ff')),
                        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#bfdbfe')),
                        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dbeafe')),
                        ('TOPPADDING', (0,0), (-1,-1), 4),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                    ]))
                    story.append(cls_table)
                    story.append(Spacer(1, 10))
            except Exception:
                pass

        # Diagnostic Images Gallery (Embedded in PDF)
        xai_data = {}
        if row.get("xai_structured"):
            try:
                xai_data = json.loads(row["xai_structured"])
            except Exception:
                pass

        img_paths = [
            ("Original Chest X-ray", row.get("original_xray") or xai_data.get("original_xray")),
            ("Grad-CAM Overlay", row.get("gradcam_overlay") or xai_data.get("gradcam_overlay")),
            ("Segmented Grad-CAM (S²A-UNet ROI)", xai_data.get("gradcam_segmented")),
            ("Captum Text Attribution Plot", row.get("captum_image") or xai_data.get("captum_image"))
        ]

        valid_imgs = []
        for lbl, ipath in img_paths:
            if ipath:
                # Convert relative web path to local file path if needed
                local_p = str(ipath)
                if local_p.startswith("/outputs/"):
                    local_p = str(settings.OUTPUT_DIR / local_p.replace("/outputs/", ""))
                if os.path.exists(local_p) and os.path.isfile(local_p):
                    try:
                        valid_imgs.append((lbl, RLImage(local_p, width=240, height=180)))
                    except Exception:
                        pass

        if valid_imgs:
            story.append(PageBreak())
            story.append(Paragraph("<b>Diagnostic Radiographs & XAI Explainability Maps</b>", heading_style))
            story.append(Spacer(1, 8))
            img_cells = []
            for i in range(0, len(valid_imgs), 2):
                row_cells = []
                for j in range(2):
                    if i + j < len(valid_imgs):
                        lbl, rl_img = valid_imgs[i + j]
                        cell_flow = [Paragraph(f"<b>{lbl}</b>", bold_body_style), rl_img]
                        row_cells.append(cell_flow)
                    else:
                        row_cells.append("")
                img_cells.append(row_cells)

            gallery_table = Table(img_cells, colWidths=[270, 270])
            gallery_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ]))
            story.append(gallery_table)

        # Footer QR Code
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=3, border=0)
        qr.add_data(f"ZenithDx Report #{row['report_id']}")
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
