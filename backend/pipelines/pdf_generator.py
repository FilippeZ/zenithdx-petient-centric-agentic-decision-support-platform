# backend/pipelines/pdf_generator.py
from __future__ import annotations

import io
import json
import os
import re
import sys
import datetime as dt
import html as html_lib
from typing import Dict, Any, List, Optional
import qrcode
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from config import settings

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _escape(text: str) -> str:
    """Escape XML/HTML entities for ReportLab Paragraph."""
    return html_lib.escape(str(text or ""))


def clean_markdown_for_reportlab(text: str) -> str:
    """Converts standard markdown bold/italic tags to ReportLab XML tags, with XML escaping."""
    if not text:
        return ""
    # Strip heading characters like ###
    text = re.sub(r"^#+\s*", "", str(text)).strip()
    # Escape XML entities first
    text = html_lib.escape(text)
    # Convert **bold** to <b>bold</b>
    parts = text.split("**")
    result = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            result.append(f"<b>{part}</b>")
        else:
            result.append(part)
    text = "".join(result)
    # Convert *italic* to <i>italic</i>
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    # Convert _italic_ to <i>italic</i>
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


def _make_rl_image_aspect(path: str, max_w: float = 250, max_h: float = 180) -> Optional[RLImage]:
    """Safely creates a ReportLab Image preserving exact aspect ratio within max_w and max_h."""
    try:
        with PILImage.open(path) as im:
            w, h = im.size
        if w <= 0 or h <= 0:
            return None
        aspect = w / float(h)
        target_w = max_w
        target_h = target_w / aspect
        if target_h > max_h:
            target_h = max_h
            target_w = target_h * aspect
        return RLImage(path, width=target_w, height=target_h)
    except Exception as e:
        print(f"[PDF] Image load error for {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Main PDF builder
# ─────────────────────────────────────────────────────────────

def generate_pdf_report_bytes(row: Dict[str, Any], is_patient_view: bool = False) -> bytes:
    """
    Generates a high-quality PDF medical report for both doctor and patient.

    Layout:
      ┌───────────────────────────────┬──────────────────────┐
      │  [Logo] ZenithDx              │  Metadata | [QR Code]│
      │  Clinical AI Platform         │  #ID  Date  Status   │
      └───────────────────────────────┴──────────────────────┘
      ┌──────────────────────────────────────────────────────┐
      │  Patient Metadata Card                               │
      └──────────────────────────────────────────────────────┘
      ┌──────────────────────────────────────────────────────┐
      │  Structured Diagnostic Assessment                    │
      │  (full text, markdown → ReportLab)                   │
      └──────────────────────────────────────────────────────┘
      [Optional: Doctor Message]
      [Optional: Patient History — only if history_text present]
      [Classification Table — only if has predictions]
      [Page Break]
      [XAI Image Gallery — original, grad-cam, segmented, captum]
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

    # ── Asset paths ─────────────────────────────────────────
    wallpaper_path = str(settings.BASE_DIR / "assets" / "wallpaper.jpg")
    logo_path = str(settings.BASE_DIR / "assets" / "logo.png")

    def draw_background(canvas, doc_obj):
        canvas.saveState()
        if os.path.exists(wallpaper_path):
            try:
                canvas.drawImage(wallpaper_path, 0, 0, width=612, height=792,
                                 preserveAspectRatio=False)
                # Soft white overlay to ensure high contrast and zero background text clutter
                canvas.setFillColor(colors.HexColor('#ffffff'))
                canvas.setLineWidth(0)
                canvas.setFillAlpha(0.85)
                canvas.rect(0, 0, 612, 792, fill=True, stroke=False)
            except Exception:
                pass
        canvas.restoreState()

    # ── Typography Styles ────────────────────────────────────
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'],
        fontSize=20, textColor=colors.HexColor('#0f172a'),
        leading=22, spaceAfter=2, fontName="Helvetica-Bold"
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#2563eb'),
        leading=12, fontName="Helvetica-Bold"
    )
    section_title_style = ParagraphStyle(
        'SectionTitle', parent=styles['Heading2'],
        fontSize=12, textColor=colors.HexColor('#1e1b4b'),
        leading=16, spaceBefore=10, spaceAfter=6,
        fontName="Helvetica-Bold"
    )
    bold_body = ParagraphStyle(
        'BoldBody', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#0f172a'),
        leading=13, fontName="Helvetica-Bold"
    )
    body_style = ParagraphStyle(
        'BodyText', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#334155'), leading=13
    )
    history_style = ParagraphStyle(
        'HistoryText', parent=styles['Normal'],
        fontSize=8.5, textColor=colors.HexColor('#374151'),
        leading=12, leftIndent=8
    )

    # ── Section 1: Header — Logo + Brand (left) | QR Code + Meta (right) ───
    report_id_str = str(row.get("report_id", "00000000"))
    patient_id_str = str(row.get("patient_id") or row.get("user_id") or "N/A")
    patient_name_str = str(row.get("patient_name") or "Patient Consultation")
    date_str = str(row.get("submission_date") or "")[:10] or dt.date.today().isoformat()
    status_str = str(row.get("status", "Pending"))

    # Left brand column with un-squished logo next to title
    brand_title_flow = [
        Paragraph("<b>ZenithDx</b>", title_style),
        Paragraph("Clinical AI Decision Support Platform", subtitle_style),
        Spacer(1, 2),
        Paragraph(
            "<i>Patient Medical Report</i>" if is_patient_view else "<i>Clinician Diagnostic Report</i>",
            body_style
        )
    ]

    logo_img = None
    if os.path.exists(logo_path):
        logo_img = _make_rl_image_aspect(logo_path, max_w=52, max_h=52)

    if logo_img:
        brand_table = Table([[logo_img, brand_title_flow]], colWidths=[58, 280])
        brand_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        left_header = brand_table
    else:
        left_header = brand_title_flow

    # Right QR column prominently at top-right
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=5, border=1
    )
    qr.add_data(
        f"ZenithDx Medical Report #{report_id_str[:8]}\n"
        f"Patient: {patient_name_str} (ID: {patient_id_str})\n"
        f"Date: {date_str}\n"
        f"Status: {status_str}"
    )
    qr.make(fit=True)
    qr_buf = io.BytesIO()
    qr.make_image(fill_color="#0f172a", back_color="white").save(qr_buf, "PNG")
    qr_buf.seek(0)
    qr_img = RLImage(qr_buf, width=72, height=72)

    status_color = "#047857" if status_str.lower() == "approved" else ("#b91c1c" if status_str.lower() == "rejected" else "#b45309")

    qr_text = [
        Paragraph(f"<b>Report ID:</b> #{report_id_str[:8]}", body_style),
        Paragraph(f"<b>Date:</b> {_escape(date_str)}", body_style),
        Paragraph(f"<b>Status:</b> <font color='{status_color}'><b>{_escape(status_str)}</b></font>", body_style),
    ]

    qr_table = Table([[qr_text, qr_img]], colWidths=[120, 75])
    qr_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))

    header_table = Table([[left_header, qr_table]], colWidths=[340, 200])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'),
                            spaceAfter=8, spaceBefore=6))

    # ── Section 2: Patient Metadata Card ────────────────────
    meta_data = [
        [
            Paragraph("<b>Patient Name</b>", bold_body),
            Paragraph(_escape(patient_name_str), body_style),
            Paragraph("<b>Patient ID</b>", bold_body),
            Paragraph(_escape(patient_id_str), body_style),
        ],
        [
            Paragraph("<b>Clinical Symptoms</b>", bold_body),
            Paragraph(_escape(str(row.get('symptoms') or "Not recorded")), body_style),
            Paragraph("<b>Submission Date</b>", bold_body),
            Paragraph(_escape(date_str), body_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[110, 165, 90, 175])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ── Section 3: Patient Notice (if Patient View) ─────────────
    if is_patient_view:
        notice_text = (
            "<b>Patient Guidance & Summary:</b> This medical report provides an AI-assisted "
            "diagnostic assessment reviewed and validated by your physician. "
            "Please review the clinical recommendations below with your healthcare provider."
        )
        notice_table = Table([[Paragraph(notice_text, body_style)]], colWidths=[540])
        notice_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0fdf4')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#86efac')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(notice_table)
        story.append(Spacer(1, 10))

    # ── Section 4: Full Structured Diagnostic Text ───────────
    story.append(Paragraph(
        "<b>Structured Diagnostic Assessment &amp; Conclusion</b>",
        section_title_style
    ))
    diag_raw = row.get('diagnosis') or "No clinical report text available."
    for line in diag_raw.split("\n"):
        line_str = line.strip()
        if not line_str:
            story.append(Spacer(1, 3))
        elif line_str.startswith("#"):
            story.append(Paragraph(f"<b>{clean_markdown_for_reportlab(line_str)}</b>", section_title_style))
        elif line_str.startswith(("- ", "* ", "• ")):
            story.append(Paragraph(f"• {clean_markdown_for_reportlab(line_str[2:])}", body_style))
        elif re.match(r"^\d+\.", line_str):
            story.append(Paragraph(f"• {clean_markdown_for_reportlab(line_str)}", body_style))
        else:
            story.append(Paragraph(clean_markdown_for_reportlab(line_str), body_style))

    story.append(Spacer(1, 10))

    # ── Section 5: Doctor Message (if present) ───────────────
    if row.get('doctor_message'):
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceAfter=6))
        story.append(Paragraph("<b>Clinician Message to Patient</b>", section_title_style))
        story.append(Paragraph(clean_markdown_for_reportlab(str(row['doctor_message'])), body_style))
        story.append(Spacer(1, 10))

    # ── Section 6: Patient History (ONLY if history_text present) ─
    history_raw = row.get("history_text")
    if history_raw:
        if isinstance(history_raw, list):
            history_entries = [t for t in history_raw if t and str(t).strip()]
        else:
            history_entries = [str(history_raw).strip()] if str(history_raw).strip() else []

        if history_entries:
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceAfter=6))
            story.append(Paragraph(
                "<b>Longitudinal EHR Patient History (HGT Graph Analysis)</b>",
                section_title_style
            ))
            story.append(Paragraph(
                "The following historical visits were retrieved from the MIMIC-IV clinical graph "
                "and used to enrich the diagnosis via Heterogeneous Graph Transformer (HGT) embeddings:",
                body_style
            ))
            story.append(Spacer(1, 6))
            for i, visit_note in enumerate(history_entries[:5], 1):
                story.append(Paragraph(f"<b>Historical Visit #{i}:</b>", bold_body))
                for vline in str(visit_note).split("\n"):
                    vl = vline.strip()
                    if vl:
                        story.append(Paragraph(f"• {_escape(vl)}", history_style))
                story.append(Spacer(1, 4))
            story.append(Spacer(1, 6))

    # ── Section 7: Multi-Label Classification Table ──────────
    cls_raw = row.get("classification_results")
    if cls_raw:
        try:
            cls_list = json.loads(cls_raw) if isinstance(cls_raw, str) else cls_raw
            if isinstance(cls_list, list) and len(cls_list) > 0:
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceAfter=6))
                story.append(Paragraph(
                    "<b>Multi-Label ResNet-50 Pathology Prediction Scores</b>",
                    section_title_style
                ))
                table_rows = [[
                    Paragraph("<b>Pathology Finding</b>", bold_body),
                    Paragraph("<b>Probability Score</b>", bold_body)
                ]]
                for item in sorted(
                    cls_list,
                    key=lambda x: x[1] if isinstance(x, (list, tuple)) and len(x) >= 2 else 0,
                    reverse=True
                ):
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        lbl, prob = item[0], float(item[1])
                        pct = int(prob * 100)
                        color_code = "#059669" if pct >= 70 else ("#d97706" if pct >= 40 else "#dc2626")
                        table_rows.append([
                            Paragraph(_escape(str(lbl)), body_style),
                            Paragraph(
                                f"<font color='{color_code}'><b>{pct}%</b></font>",
                                body_style
                            )
                        ])
                cls_table = Table(table_rows, colWidths=[270, 270])
                cls_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eff6ff')),
                    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#bfdbfe')),
                    ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dbeafe')),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ]))
                story.append(cls_table)
                story.append(Spacer(1, 12))
        except Exception:
            pass

    # ── Section 8: XAI Image Gallery ─────────────────────────
    xai_data = {}
    if row.get("xai_structured"):
        try:
            xai_data = (
                json.loads(row["xai_structured"])
                if isinstance(row["xai_structured"], str)
                else row["xai_structured"]
            )
        except Exception:
            pass

    candidate_images = [
        ("Original Chest Radiograph", row.get("original_xray") or xai_data.get("original_xray")),
        ("Grad-CAM Pathology Overlay", row.get("gradcam_overlay") or xai_data.get("gradcam_overlay")),
        ("Segmented Grad-CAM (S²A-UNet ROI)", xai_data.get("gradcam_segmented") or row.get("gradcam_segmented")),
    ]

    valid_imgs = []
    for lbl, rpath in candidate_images:
        local_p = resolve_local_image_path(rpath)
        if local_p:
            rl_img = _make_rl_image_aspect(local_p, max_w=250, max_h=180)
            if rl_img:
                valid_imgs.append((lbl, rl_img))

    if valid_imgs:
        story.append(PageBreak())
        story.append(Paragraph(
            "<b>Diagnostic Radiographs &amp; Explainable AI (XAI) Heatmaps</b>",
            section_title_style
        ))
        story.append(Spacer(1, 8))
        img_cells = []
        for i in range(0, len(valid_imgs), 2):
            row_cells = []
            for j in range(2):
                if i + j < len(valid_imgs):
                    lbl, rl_img = valid_imgs[i + j]
                    cell_flow = [
                        Paragraph(f"<b>{_escape(lbl)}</b>", bold_body),
                        Spacer(1, 4),
                        rl_img
                    ]
                    row_cells.append(cell_flow)
                else:
                    row_cells.append("")
            img_cells.append(row_cells)

        gallery_table = Table(img_cells, colWidths=[270, 270])
        gallery_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ]))
        story.append(gallery_table)

    # ── Footer ────────────────────────────────────────────────
    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceAfter=4))
    story.append(Paragraph(
        "<i>Confidential Diagnostic Medical Report — Generated by ZenithDx "
        "Multi-Modal Agentic AI Suite. For physician &amp; patient records only.</i>",
        body_style
    ))

    doc.build(story, onFirstPage=draw_background, onLaterPages=draw_background)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
