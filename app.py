# app.py
import os
import base64
import requests
from io import BytesIO
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from PIL import Image
import numpy as np
import cv2

# InsightFace
import insightface
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
TRAIN_DIR = "Training_images"
LOG_FILE = "Log_Data.csv"

INSIGHT_MODEL = "buffalo_l"   # or "antelope"
COSINE_THRESHOLD = 0.50       # tune as needed

os.makedirs(TRAIN_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Name,LoginTime,Status\n")

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ---------------- InsightFace Initialization ----------------
try:
    insight_app = FaceAnalysis(name=INSIGHT_MODEL)
    try:
        insight_app.prepare(ctx_id=0, det_size=(640, 640))  # try GPU
    except:
        insight_app.prepare(ctx_id=-1, det_size=(640, 640)) # CPU
    print("InsightFace loaded successfully.")
except Exception as e:
    insight_app = None
    print(f"[ERROR] Failed to load InsightFace: {e}")


# ---------------- Helper Functions ----------------
def b64_to_pil_image(data_url_or_b64):
    if data_url_or_b64.startswith("data:"):
        header, b64 = data_url_or_b64.split(",", 1)
    else:
        b64 = data_url_or_b64

    img_bytes = base64.b64decode(b64)
    return Image.open(BytesIO(img_bytes)).convert("RGB")


def pil_to_bgr_np(pil_img):
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_first_normed_embedding_from_pil(pil_img):
    """Extract ArcFace embedding safely — NO boolean check on arrays."""
    if insight_app is None:
        raise RuntimeError("InsightFace not initialized")

    img_bgr = pil_to_bgr_np(pil_img)
    faces = insight_app.get(img_bgr)

    if not faces:
        return None

    # pick largest face
    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    face = faces_sorted[0]

    # SAFE: do NOT use "or" on NumPy arrays
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)

    if emb is None:
        return None

    emb = np.asarray(emb, dtype=np.float32)

    # ensure l2 normalization
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b))


def log_attendance_local(name):
    now = datetime.now()
    with open(LOG_FILE, "a") as f:
        f.write(f"{name},{now.strftime('%d-%m-%Y %H:%M:%S')},Present\n")


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_image", methods=["POST"])
def process_image():
    try:
        data = request.get_json(force=True)

        captured_data = data.get("captured_image")
        reference = data.get("reference_image")
        user_id = data.get("user_id")
        emp_id = data.get("emp_id")

        if not captured_data:
            return jsonify({"status": "error", "error": "No captured_image provided"})

        # ---------------- PROCESS CAPTURED IMAGE ----------------
        try:
            captured_pil = b64_to_pil_image(captured_data)
        except Exception as e:
            return jsonify({"status": "error", "error": f"Invalid captured_image: {e}"})

        captured_emb = get_first_normed_embedding_from_pil(captured_pil)
        if captured_emb is None:
            return jsonify({"status": "unknown", "error": "No face found in captured image"})

        # ---------------- PROCESS REFERENCE IMAGE ----------------
        if not reference:
            return jsonify({"status": "error", "error": "No reference_image provided"})

        # If URL, download
        if reference.startswith("http"):
            try:
                resp = requests.get(reference, timeout=10)
                resp.raise_for_status()
                ref_pil = Image.open(BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                return jsonify({"status": "error", "error": f"Could not download reference image: {e}"})
        else:
            try:
                ref_pil = b64_to_pil_image(reference)
            except Exception as e:
                return jsonify({"status": "error", "error": f"Invalid reference_image: {e}"})

        ref_emb = get_first_normed_embedding_from_pil(ref_pil)
        if ref_emb is None:
            return jsonify({"status": "error", "error": "No face found in reference image"})

        # ---------------- COMPARE USING COSINE SIMILARITY ----------------
        similarity = cosine_similarity(ref_emb, captured_emb)

        if similarity >= COSINE_THRESHOLD:
            # Match found
            name = data.get("name") or f"USER_{user_id or emp_id}"

            log_attendance_local(name)

            # Convert reference image back to base64
            buf = BytesIO()
            ref_pil.save(buf, format="JPEG", quality=90)
            b64img = base64.b64encode(buf.getvalue()).decode("utf-8")
            matched_datauri = f"data:image/jpeg;base64,{b64img}"

            # ---------------- CALL ATTENDANCE API ----------------
            attendance_response = None
            if emp_id:
                try:
                    attendance_url = f"https://acsdev.in/fynryx_backend/api/qr/attendance/{emp_id}"

                    now = datetime.now()
                    payload = {
                        "user_id": str(user_id or emp_id),
                        "attendance_date": now.strftime("%d-%m-%Y"),
                        "check_in_time": now.strftime("%H:%M"),
                        "check_out_time": "",
                        "status": "present",
                        "create_audit_id": "1",
                    }

                    headers = {"Content-Type": "application/json"}
                    res = requests.post(attendance_url, json=payload, headers=headers, timeout=10)
                    attendance_response = res.json()
                except Exception as e:
                    attendance_response = {"error": f"API call failed: {e}"}

            return jsonify({
                "status": "success",
                "name": name,
                "matched_image": matched_datauri,
                "similarity": similarity,
                "attendance_response": attendance_response,
            })

        else:
            return jsonify({
                "status": "unknown",
                "similarity": similarity,
                "message": "Face did not match reference."
            })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})


# ---------------- BOOT ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
