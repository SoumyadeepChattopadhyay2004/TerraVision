"""
Builds a professional PDF report (segmentation map, statistics table,
vegetation index map, landscape metrics) for a single classification, or a
before/after change-detection summary. This is the "hand this to a client
or a professor" deliverable.

Requires reportlab (pip install reportlab). Falls back gracefully with a
clear error if it isn't installed, rather than crashing the whole app.
"""
from __future__ import annotations
import io
from datetime import datetime
from typing import Optional

from PIL import Image

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
                                     Table, TableStyle, PageBreak)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def _pil_to_rlimage(img: Image.Image, max_width_cm: float = 16) -> "RLImage":
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    aspect = img.height / img.width
    w = max_width_cm * cm
    return RLImage(buf, width=w, height=w * aspect)


def build_classification_report(
    original: Image.Image,
    overlay: Image.Image,
    veg_map: Image.Image,
    stats: dict,
    landscape_metrics: dict,
    engine: str,
    gsd_m: Optional[float] = None,
) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed — run `pip install reportlab` to enable PDF export.")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=20)
    story = []

    story.append(Paragraph("Terra — Land-Cover Classification Report", title_style))
    story.append(Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · Engine: {engine}", styles["Normal"]))
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("Input & Segmentation", styles["Heading2"]))
    img_table = Table([[_pil_to_rlimage(original, 8), _pil_to_rlimage(overlay, 8)]])
    img_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(img_table)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Land-Cover Statistics", styles["Heading2"]))
    header = ["Class", "% cover", "Pixels"] + (["Area (km²)"] if gsd_m else [])
    rows = [header]
    for name, s in stats.items():
        row = [name, f"{s['percent']}%", f"{s['pixels']:,}"]
        if gsd_m:
            row.append(f"{s.get('km2', '—')}")
        rows.append(row)
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2420")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9d0bd")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef1e8")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Vegetation Health Index (proxy-NDVI)", styles["Heading2"]))
    story.append(_pil_to_rlimage(veg_map, 10))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Landscape Fragmentation Metrics", styles["Heading2"]))
    lm_rows = [["Class", "Patch count", "Mean patch size (px)", "Edge density"]]
    for name, m in landscape_metrics.items():
        lm_rows.append([name, str(m["patch_count"]), str(m["mean_patch_size_px"]), str(m["edge_density"])])
    lm_table = Table(lm_rows, hAlign="LEFT")
    lm_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2420")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9d0bd")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(lm_table)

    doc.build(story)
    return buf.getvalue()


def build_change_report(
    before_overlay: Image.Image,
    after_overlay: Image.Image,
    change_result: dict,
) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed — run `pip install reportlab` to enable PDF export.")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    story = [Paragraph("Terra — Land-Cover Change Detection Report", styles["Title"]),
             Paragraph(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · Engine: {change_result.get('engine_used', '—')}", styles["Normal"]),
             Spacer(1, 0.5 * cm)]

    img_table = Table([[_pil_to_rlimage(before_overlay, 8), _pil_to_rlimage(after_overlay, 8)]])
    story.append(img_table)
    story.append(Spacer(1, 0.4 * cm))

    summary = change_result.get("summary", {})
    story.append(Paragraph(
        f"<b>Urban expansion rate:</b> {summary.get('urban_expansion_rate_percent', '—')}% "
        f"&nbsp;&nbsp; <b>Vegetation loss rate:</b> {summary.get('vegetation_loss_rate_percent', '—')}%",
        styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    rows = [["Class", "Before %", "After %", "Δ %"]]
    for name, c in change_result["per_class_change"].items():
        rows.append([name, f"{c['before_percent']}", f"{c['after_percent']}", f"{c['delta_percent']:+}"])
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2420")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9d0bd")),
    ]))
    story.append(t)

    doc.build(story)
    return buf.getvalue()
