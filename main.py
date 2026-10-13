import os

from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.exceptions import RequestEntityTooLarge

from inference import Classifier, check_input, heatmap_overlay, open_image, thumbnail, to_data_url

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))
SAMPLES_DIR = os.path.join(BASE_DIR, "static", "samples")
MAX_UPLOAD_MB = 10
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Refuse "decompression bomb" images (tiny files that expand to huge bitmaps).
Image.MAX_IMAGE_PIXELS = 25_000_000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

classifier = Classifier(MODEL_DIR)


def list_samples():
    """Example scans shipped in static/samples, e.g. 'glioma_1.jpg'."""
    if not os.path.isdir(SAMPLES_DIR):
        return []
    samples = []
    for name in sorted(os.listdir(SAMPLES_DIR)):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            label = name.rsplit(".", 1)[0].split("_")[0].lower()
            samples.append({
                "file": name,
                "label": classifier.display_names[classifier.labels.index(label)]
                if label in classifier.labels else "Sample",
            })
    return samples


def error(message, status, **extra):
    return jsonify({"status": "error", "message": message, **extra}), status


@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return error(f"The file is too large. The maximum upload size is {MAX_UPLOAD_MB} MB.", 413)


@app.route("/")
def index():
    return render_template("index.html", meta=classifier.meta, samples=list_samples(),
                           max_upload_mb=MAX_UPLOAD_MB)


@app.route("/healthz")
def healthz():
    return {"status": "ok", "model": classifier.meta["name"]}


@app.route("/api/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None or not file.filename:
        return error("No file was uploaded. Please choose an MRI image.", 400)

    # Images are processed in memory and never written to disk.
    try:
        img = open_image(file.read())
    except ValueError as exc:
        return error(str(exc), 400)

    rejections, warnings = check_input(img)
    if rejections:
        return jsonify({
            "status": "rejected",
            "message": "This image does not look like a brain MRI slice, so no prediction was made.",
            "reasons": rejections,
        }), 422

    result = classifier.predict(img)
    view = thumbnail(img, 512)
    heatmap = heatmap_overlay(view, result.pop("cam"))

    if result["confident"]:
        status, message = "ok", None
    else:
        status = "inconclusive"
        message = (f"The model is not confident (top score {result['confidence'] * 100:.1f}%, "
                   f"below the {classifier.threshold * 100:.0f}% threshold). "
                   "Treat this result as inconclusive.")
    return jsonify({
        "status": status,
        "message": message,
        "warnings": warnings,
        "prediction": result,
        "heatmap": to_data_url(heatmap, "JPEG"),
        "model": {"name": classifier.meta["name"], "description": classifier.meta.get("description", "")},
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
