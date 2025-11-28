# app.py
import os
import base64
import requests
from io import BytesIO
from datetime import datetime, timezone, timedelta

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from PIL import Image
import numpy as np
import cv2
import msal

# InsightFace
import insightface
from insightface.app import FaceAnalysis


# ---------------- CONFIG ----------------
TRAIN_DIR = "Training_images"
LOG_FILE = "Log_Data.csv"

INSIGHT_MODEL = "buffalo_l"
COSINE_THRESHOLD = 0.50

os.makedirs(TRAIN_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Name,LoginTime,Status\n")

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ---------------- EMAIL CONFIG (MICROSOFT GRAPH) ----------------
IST = timezone(timedelta(hours=5, minutes=30))

CLIENT_ID     = os.getenv("CLIENT_ID", "377ae826-cdfd-41ab-bfa5-8b510aa5e668")
TENANT_ID     = os.getenv("TENANT_ID", "11f40179-cc91-4f79-bc13-d468ae3c3faf")
CLIENT_SECRET = os.getenv("CLIENT_SECRET", "Mff8Q~Jvf_m~FxmSojsoCd6Ef6QLoRYsb8h3TbwJ")
HR_UPN        = os.getenv("HR_UPN", "hr@algorithmicaconsulting.com")


def get_graph_token():
    app_msal = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{TENANT_ID}",
        client_credential=CLIENT_SECRET,
    )

    token = app_msal.acquire_token_for_client(
        scopes=["https://graph.microsoft.com/.default"]
    )

    if "access_token" not in token:
        raise RuntimeError(f"Token error: {token}")

    return token["access_token"]


def send_email(to_email, subject, html_body):
    try:
        token = get_graph_token()

        msg = {
            "message": {
                "subject": subject,
                "body": {
                    "contentType": "HTML",
                    "content": html_body
                },
                "toRecipients": [
                    {"emailAddress": {"address": to_email}}
                ],
            },
            "saveToSentItems": True,
        }

        requests.post(
            f"https://graph.microsoft.com/v1.0/users/{HR_UPN}/sendMail",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json=msg,
            timeout=20
        ).raise_for_status()

        print(f"[EMAIL SENT ✅] {to_email}")

    except Exception as e:
        print(f"[EMAIL ERROR ❌] {e}")


# ---------------- InsightFace Initialization ----------------
try:
    insight_app = FaceAnalysis(name=INSIGHT_MODEL)
    try:
        insight_app.prepare(ctx_id=0, det_size=(640, 640))  # GPU
    except:
        insight_app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU
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
    if insight_app is None:
        raise RuntimeError("InsightFace not initialized")

    img_bgr = pil_to_bgr_np(pil_img)
    faces = insight_app.get(img_bgr)

    if not faces:
        return None

    faces_sorted = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    face = faces_sorted[0]

    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)

    if emb is None:
        return None

    emb = np.asarray(emb, dtype=np.float32)
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


# ---------------- FACE MATCH + ATTENDANCE ----------------
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

        # --- Captured Image ---
        captured_pil = b64_to_pil_image(captured_data)
        captured_emb = get_first_normed_embedding_from_pil(captured_pil)
        if captured_emb is None:
            return jsonify({"status": "unknown", "error": "No face found in captured image"})

        # --- Reference Image ---
        if reference.startswith("http"):
            resp = requests.get(reference, timeout=10)
            resp.raise_for_status()
            ref_pil = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            ref_pil = b64_to_pil_image(reference)

        ref_emb = get_first_normed_embedding_from_pil(ref_pil)
        if ref_emb is None:
            return jsonify({"status": "error", "error": "No face found in reference image"})

        # --- Compare ---
        similarity = cosine_similarity(ref_emb, captured_emb)

        if similarity >= COSINE_THRESHOLD:
            name = data.get("name") or f"USER_{user_id or emp_id}"
            log_attendance_local(name)

            buf = BytesIO()
            ref_pil.save(buf, format="JPEG", quality=90)
            b64img = base64.b64encode(buf.getvalue()).decode("utf-8")
            matched_datauri = f"data:image/jpeg;base64,{b64img}"

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
                    res = requests.post(attendance_url, json=payload, timeout=10)
                    attendance_response = res.json()
                except Exception as e:
                    attendance_response = {"error": str(e)}

            return jsonify({
                "status": "success",
                "name": name,
                "matched_image": matched_datauri,
                "similarity": similarity,
                "attendance_response": attendance_response,
            })

        return jsonify({
            "status": "unknown",
            "similarity": similarity,
            "message": "Face did not match reference."
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})


# ---------------- EMAIL API (FIXED 404 ISSUE) ----------------
@app.route("/api/send_attendance_email", methods=["POST", "OPTIONS"])
def api_send_attendance_email():
    if request.method == "OPTIONS":
        return ("", 200)

    data = request.get_json(force=True) or {}
    to_email = data.get("email")
    name = data.get("name", "Employee")

    if not to_email:
        return jsonify({"status": "error", "message": "email required"}), 400

    ts = datetime.now(IST).strftime("%d %b %Y, %I:%M %p")
    subject = f"Attendance Marked - {ts}"

    html_body = f"""
    <html>
      <body style="font-family: system-ui; background:#f3f4f6; padding:24px;">
        <div style="max-width:480px; margin:0 auto;">
          <div style="background:linear-gradient(135deg,#2563eb,#7c3aed); padding:2px; border-radius:16px;">
            <div style="background:#ffffff; border-radius:14px; padding:24px;">
              <h2>Attendance Marked ✅</h2>
              <p>Hello <b>{name}</b>, your attendance was marked successfully.</p>
              <div style="background:#f9fafb; padding:16px; border-radius:12px;">
                <p><b>Time:</b> {ts} (IST)</p>
              </div>
              <p style="color:#6b7280; font-size:12px;">Automated Attendance System</p>
            </div>
          </div>
        </div>
      </body>
    </html>
    """

    send_email(to_email, subject, html_body)

    return jsonify({"status": "success", "message": "Email sent"}), 200


# ---------------- BOOT ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
