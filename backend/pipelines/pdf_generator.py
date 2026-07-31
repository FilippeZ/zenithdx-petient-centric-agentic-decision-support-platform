# backend/pipelines/pdf_generator.py
from __future__ import annotations

import io
import json
import os
import re
import sys
import datetime as dt
from typing import Dict, Any, List, Optional
import qrcode
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from config import settings


def clean_markdown_for_reportlab(text: str) -> str:
    """Converts standard markdown bold/italic tags to ReportLab XML tags."""
    if not text:
        return ""
    # Strip heading characters like ###
    text = re.sub(r"^#+\s*", "", text)
    # Convert **bold** to <b>bold</b>
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    # Convert *italic* or _italic_ to <i>italic</i>
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    text = re.sub(r"_(.*?)_", r"<i>\1</i>", text)
    return text.strip()


def resolve_local_image_path(raw_path: Optional[str]) -> Optional[str]:
    """Resolves web URI / relative path to a valid local filesystem path."""
    if not raw_path:
        return None
    path_str = str(raw_path)
    if path_str.startswith("data:image"):
        return None
    if path_str.startswith("/outputs/"):
        target = str(settings.OUTPUT_DIR / path_str.replace("/outputs/", ""))
    elif path_str.startswith("/uploads/"):
        target = str(settings.UPLOAD_DIR / path_str.replace("/uploads/", ""))
    else:
        target = path_str

    if os.path.exists(target) and os.path.isfile(target):
        return target
    return None


def generate_pdf_report_bytes(row: Dict[str, Any]) -> bytes:
    """
    Generates high-precision PDF report bytes for patient/doctor download.
    - Correct Logo Aspect Ratio (no compression)
    - Prominent Large QR Code at Top-Right of Header
    - All Diagnostic Radiographs & XAI Explainability Heatmaps
    - Clean ReportLab Typography & Formatting
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
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

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#0f172a'),
        leading=22,
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#2563eb'),
        leading=12,
        fontName="Helvetica-Bold"
    )
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#1e1b4b'),
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
        fontName="Helvetica-Bold"
    )
    bold_body = ParagraphStyle('BoldBody', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#0f172a'), leading=13, fontName="Helvetica-Bold")
    body_style = ParagraphStyle('BodyText', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#334155'), leading=13)

    # 1. ZenithDx Logo with Exact Preserved Aspect Ratio
    brand_flow = []
    if os.path.exists(logo_path):
        try:
            with PILImage.open(logo_path) as im:
                orig_w, orig_h = im.size
            aspect = orig_w / float(max(1, orig_h))
            target_h = 42
            target_w = min(220, int(target_h * aspect))
            brand_flow.append(RLImage(logo_path, width=target_w, height=target_h))
            brand_flow.append(Spacer(1, 4))
        except Exception as e:
            print(f"[PDF] Logo load note: {e}")

    brand_flow.append(Paragraph("<b>ZenithDx</b>", title_style))
    brand_flow.append(Paragraph("Clinical AI Decision Support Platform", subtitle_style))

    # 2. Large Top-Right QR Code Header
    report_id_str = str(row.get("report_id", "00000000"))
    patient_id_str = str(row.get("patient_id") or row.get("user_id") or "10000032")
    patient_name_str = str(row.get("patient_name") or "Patient Consultation")
    date_str = str(row.get("submission_date") or "")[:10] or dt.date.today().isoformat()

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=4, border=1)
    qr.add_data(f"ZenithDx Medical Report #{report_id_str[:8]}\nPatient: {patient_name_str} ({patient_id_str})\nDate: {date_str}\nStatus: {row.get('status', 'Pending')}")
    qr.make(fit=True)
    qr_buf = io.BytesIO()
    qr.make_image(fill_color="#0f172a", back_color="white").save(qr_buf, "PNG")
    qr_buf.seek(0)
    
    qr_img_header = RLImage(qr_buf, width=75, height=75)

    right_flow = [
        qr_img_header,
        Spacer(1, 4),
        Paragraph(f"<b>Report ID:</b> #{report_id_str[:8]}", body_style),
        Paragraph(f"<b>Date:</b> {date_str}", body_style),
        Paragraph(f"<b>Status:</b> <font color='#047857'><b>{row.get('status', 'Pending')}</b></font>", body_style)
    ]

    header_table = Table([[brand_flow, right_flow]], colWidths=[340, 200])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # 3. Patient Overview Metadata Card
    meta_data = [
        [Paragraph("<b>Patient Name</b>", bold_body), Paragraph(patient_name_str, body_style),
         Paragraph("<b>Patient ID</b>", bold_body), Paragraph(patient_id_str, body_style)],
        [Paragraph("<b>Clinical Symptoms</b>", bold_body), Paragraph(str(row.get('symptoms') or "Not recorded"), body_style),
         Paragraph("<b>Submission Date</b>", bold_body), Paragraph(date_str, body_style)]
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
    story.append(Spacer(1, 12))

    # 4. Structured Clinical Diagnostic Report Text
    story.append(Paragraph("<b>Structured Diagnostic Assessment & Conclusion</b>", section_title_style))
    diag_raw = row.get('diagnosis') or "No clinical report text available."
    diag_lines = diag_raw.split("\n")
    for line in diag_lines:
        line_str = line.strip()
        if not line_str:
            story.append(Spacer(1, 3))
        elif line_str.startswith("#"):
            clean_h = clean_markdown_for_reportlab(line_str)
            story.append(Paragraph(f"<b>{clean_h}</b>", section_title_style))
        elif line_str.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.")):
            clean_l = clean_markdown_for_reportlab(line_str)
            story.append(Paragraph(f"• {clean_l}", body_style))
        else:
            clean_l = clean_markdown_for_reportlab(line_str)
            story.append(Paragraph(clean_l, body_style))
    story.append(Spacer(1, 10))

    # Doctor Message
    if row.get('doctor_message'):
        story.append(Paragraph("<b>Clinician Message to Patient</b>", section_title_style))
        story.append(Paragraph(clean_markdown_for_reportlab(str(row['doctor_message'])), body_style))
        story.append(Spacer(1, 10))

    # 5. Multi-Label ResNet-50 Pathology Prediction Table
    cls_raw = row.get("classification_results")
    if cls_raw:
        try:
            cls_list = json.loads(cls_raw) if isinstance(cls_raw, str) else cls_raw
            if isinstance(cls_list, list) and len(cls_list) > 0:
                story.append(Paragraph("<b>Multi-Label ResNet-50 Pathology Prediction Scores</b>", section_title_style))
                table_rows = [[Paragraph("<b>Pathology Finding</b>", bold_body), Paragraph("<b>Probability Score</b>", bold_body)]]
                for item in sorted(cls_list, key=lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) >= 2 else 0, reverse=True):
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        lbl, prob = item[0], float(item[1])
                        pct = int(prob * 100)
                        color_code = "#059669" if pct >= 70 else ("#d97706" if pct >= 40 else "#dc2626")
                        table_rows.append([
                            Paragraph(str(lbl), body_style),
                            Paragraph(f"<font color='{color_code}'><b>{pct}%</b></font>", body_style)
                        ])
                cls_table = Table(table_rows, colWidths=[270, 270])
                cls_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eff6ff')),
                    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#bfdbfe')),
                    ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#dbeafe')),
                    ('TOPPADDING', (0,0), (-1,-1), 5),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ]))
                story.append(cls_table)
                story.append(Spacer(1, 12))
        except Exception:
            pass

    # 6. ALL Diagnostic Images Gallery (Original X-ray, Grad-CAM, S²A-UNet, Captum)
    xai_data = {}
    if row.get("xai_structured"):
        try:
            xai_data = json.loads(row["xai_structured"]) if isinstance(row["xai_structured"], str) else row["xai_structured"]
        except Exception:
            pass

    candidate_images = [
        ("Original Chest Radiograph", row.get("original_xray") or xai_data.get("original_xray")),
        ("Grad-CAM Pathology Overlay", row.get("gradcam_overlay") or xai_data.get("gradcam_overlay")),
        ("Segmented Grad-CAM (S²A-UNet ROI)", xai_data.get("gradcam_segmented")),
        ("Captum Sequence Attribution", row.get("captum_image") or xai_data.get("captum_image")),
        ("Captum Token Importance Map", xai_data.get("captum_tok_path") or xai_data.get("captum_query_tok"))
    ]

    valid_imgs = []
    for lbl, rpath in candidate_images:
        local_p = resolve_local_image_path(rpath)
        if local_p:
            try:
                valid_imgs.append((lbl, RLImage(local_p, width=250, height=180)))
            except Exception as e:
                print(f"[PDF] Image load error ({lbl}): {e}")

    if valid_imgs:
        story.append(PageBreak())
        story.append(Paragraph("<b>Diagnostic Radiographs & Explainable AI (XAI) Heatmaps</b>", section_title_style))
        story.append(Spacer(1, 8))
        img_cells = []
        for i in range(0, len(valid_imgs), 2):
            row_cells = []
            for j in range(2):
                if i + j < len(valid_imgs):
                    lbl, rl_img = valid_imgs[i + j]
                    cell_flow = [Paragraph(f"<b>{lbl}</b>", bold_body), Spacer(1, 4), rl_img]
                    row_cells.append(cell_flow)
                else:
                    row_cells.append("")
            img_cells.append(row_cells)

        gallery_table = Table(img_cells, colWidths=[270, 270])
        gallery_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ]))
        story.append(gallery_table)

    # Footer note
    story.append(Spacer(1, 14))
    story.append(Paragraph("<i>Confidential Diagnostic Medical Report — Generated by ZenithDx Multi-Modal Agentic AI Suite</i>", body_style))

    doc.build(story, onFirstPage=draw_background, onLaterPages=draw_background)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
