"""
Flask backend for the Terra Satellite Land-Cover Classification web app.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000

Endpoints:
    GET  /                        -> web UI
    POST /api/classify            -> segmentation + stats + vegetation index +
                                      confidence map + landscape metrics (tiled
                                      inference for large images in CNN mode)
    POST /api/change-detection    -> before/after change stats + transition matrix
    POST /api/export-pdf          -> PDF report for the last single-image classification
    POST /api/export-change-pdf   -> PDF report for a change-detection run
    GET  /api/history             -> recent analyses (persisted in history.db)
    DELETE /api/history           -> clear history
"""
from __future__ import annotations
import base64
import io

from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image

from model.inference import (
    LandCoverClassifier, stats_from_mask, colorize_mask, change_detection,
    vegetation_index_map, confidence_map, landscape_metrics,
)
from model.unet import CLASS_NAMES, CLASS_COLORS
from model import history as history_db
from model import report as report_gen

app = Flask(__name__)
classifier = LandCoverClassifier()  # auto-detects trained checkpoint vs heuristic fallback

MAX_IMAGE_DIM = 1024  # cap resolution server-side to keep inference fast
THUMB_SIZE = 96


def _load_image_from_request(field_name: str) -> Image.Image:
    if field_name not in request.files:
        raise ValueError(f"Missing file field: {field_name}")
    file = request.files[field_name]
    img = Image.open(file.stream).convert("RGB")
    img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.BILINEAR)
    return img


def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _thumbnail_base64(img: Image.Image) -> str:
    thumb = img.copy()
    thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.BILINEAR)
    return _image_to_base64(thumb)


def _float_or_none(val):
    try:
        return float(val) if val not in (None, "") else None
    except ValueError:
        return None


@app.route("/")
def index():
    return render_template(
        "index.html",
        class_names=CLASS_NAMES,
        # pass as an index-ordered list, not the raw {int: tuple} dict — a Python
        # dict with int keys serializes to a JSON *object* via tojson (keys become
        # strings), which breaks any JS array method (.map/.forEach) called on it
        # even though bracket indexing like COLORS[i] still happens to work.
        class_colors=[list(CLASS_COLORS[i]) for i in range(len(CLASS_NAMES))],
        mode=classifier.mode,
        has_cnn=classifier.model is not None,
    )


@app.route("/api/classify", methods=["POST"])
def api_classify():
    try:
        img = _load_image_from_request("image")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    gsd = _float_or_none(request.form.get("gsd"))
    engine = request.form.get("engine")  # "cnn" | "heuristic" | None (auto)

    mask = classifier.classify(img, engine=engine)
    stats = stats_from_mask(mask, gsd_m=gsd)
    overlay = colorize_mask(mask)
    veg_map = vegetation_index_map(img)
    conf_map = confidence_map(mask, classifier, img)
    metrics = landscape_metrics(mask)

    engine_used = engine if engine in ("cnn", "heuristic") else classifier.mode
    try:
        history_db.log_analysis("classify", engine_used, _thumbnail_base64(overlay), stats)
    except Exception as e:  # never let history logging break the actual response
        app.logger.warning(f"history log failed: {e}")

    return jsonify({
        "engine_used": engine_used,
        "cnn_available": classifier.model is not None,
        "width": img.width,
        "height": img.height,
        "stats": stats,
        "landscape_metrics": metrics,
        "overlay_png_base64": _image_to_base64(overlay),
        "vegetation_index_base64": _image_to_base64(veg_map),
        "confidence_map_base64": _image_to_base64(conf_map),
        "legend": {name: list(CLASS_COLORS[i]) for i, name in enumerate(CLASS_NAMES)},
    })


@app.route("/api/change-detection", methods=["POST"])
def api_change_detection():
    try:
        img_before = _load_image_from_request("image_before")
        img_after = _load_image_from_request("image_after")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    gsd = _float_or_none(request.form.get("gsd"))
    engine = request.form.get("engine")

    common_size = (min(img_before.width, img_after.width), min(img_before.height, img_after.height))
    img_before = img_before.resize(common_size, Image.BILINEAR)
    img_after = img_after.resize(common_size, Image.BILINEAR)

    mask_before = classifier.classify(img_before, engine=engine)
    mask_after = classifier.classify(img_after, engine=engine)

    result = change_detection(mask_before, mask_after, gsd_m=gsd)
    engine_used = engine if engine in ("cnn", "heuristic") else classifier.mode
    result["engine_used"] = engine_used
    result["overlay_before_base64"] = _image_to_base64(colorize_mask(mask_before))
    result["overlay_after_base64"] = _image_to_base64(colorize_mask(mask_after))
    result["landscape_metrics_before"] = landscape_metrics(mask_before)
    result["landscape_metrics_after"] = landscape_metrics(mask_after)

    try:
        thumb_b64 = _thumbnail_base64(colorize_mask(mask_after))
        history_db.log_analysis("change", engine_used, thumb_b64, result["summary"])
    except Exception as e:
        app.logger.warning(f"history log failed: {e}")

    return jsonify(result)


@app.route("/api/export-pdf", methods=["POST"])
def api_export_pdf():
    try:
        img = _load_image_from_request("image")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    gsd = _float_or_none(request.form.get("gsd"))
    engine = request.form.get("engine")

    mask = classifier.classify(img, engine=engine)
    stats = stats_from_mask(mask, gsd_m=gsd)
    overlay = colorize_mask(mask)
    veg_map = vegetation_index_map(img)
    metrics = landscape_metrics(mask)
    engine_used = engine if engine in ("cnn", "heuristic") else classifier.mode

    try:
        pdf_bytes = report_gen.build_classification_report(img, overlay, veg_map, stats, metrics, engine_used, gsd_m=gsd)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf",
                      as_attachment=True, download_name="terra_landcover_report.pdf")


@app.route("/api/export-change-pdf", methods=["POST"])
def api_export_change_pdf():
    try:
        img_before = _load_image_from_request("image_before")
        img_after = _load_image_from_request("image_after")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    gsd = _float_or_none(request.form.get("gsd"))
    engine = request.form.get("engine")

    common_size = (min(img_before.width, img_after.width), min(img_before.height, img_after.height))
    img_before = img_before.resize(common_size, Image.BILINEAR)
    img_after = img_after.resize(common_size, Image.BILINEAR)

    mask_before = classifier.classify(img_before, engine=engine)
    mask_after = classifier.classify(img_after, engine=engine)
    result = change_detection(mask_before, mask_after, gsd_m=gsd)
    result["engine_used"] = engine if engine in ("cnn", "heuristic") else classifier.mode

    try:
        pdf_bytes = report_gen.build_change_report(colorize_mask(mask_before), colorize_mask(mask_after), result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf",
                      as_attachment=True, download_name="terra_change_report.pdf")


@app.route("/api/history", methods=["GET"])
def api_get_history():
    limit = request.args.get("limit", 20, type=int)
    return jsonify(history_db.get_history(limit=limit))


@app.route("/api/history", methods=["DELETE"])
def api_clear_history():
    history_db.clear_history()
    return jsonify({"ok": True})


if __name__ == "__main__":
    print(f"Classifier running in '{classifier.mode}' mode "
          f"({'trained CNN (tiled inference for large images)' if classifier.mode == 'cnn' else 'heuristic fallback — train a model to improve accuracy'})")
    app.run(debug=True, host="0.0.0.0", port=5000)
