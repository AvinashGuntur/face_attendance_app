from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from datetime import datetime
import face_recognition
from PIL import Image
import base64
from flask_cors import CORS
from io import BytesIO
import requests

# ---------------- CONFIG ----------------
TRAIN_DIR = "Training_images"  # still available but we will use reference_image from frontend
LOG_FILE = "Log_Data.csv"

os.makedirs(TRAIN_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Name,LoginTime,Status\n")

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

def b64_to_pil_image(data_url_or_b64):
    """Accepts either a data URL (data:image/..;base64,...) or raw base64 string"""
    if data_url_or_b64.startswith('data:'):
        header, b64 = data_url_or_b64.split(',', 1)
    else:
        b64 = data_url_or_b64
    img_bytes = base64.b64decode(b64)
    return Image.open(BytesIO(img_bytes)).convert('RGB')

def pil_to_rgb_np(pil_img):
    return np.array(pil_img)  # RGB

def find_first_encoding_from_pil(pil_img):
    np_img = pil_to_rgb_np(pil_img)
    encs = face_recognition.face_encodings(np_img)
    if encs:
        return encs[0]
    return None

def log_attendance_local(name):
    now = datetime.now()
    with open(LOG_FILE, "a") as f:
        f.write(f"{name},{now.strftime('%d-%m-%Y %H:%M:%S')},Present\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process_image", methods=["POST"])
def process_image():
    """
    Expects JSON:
    {
      "captured_image": "data:image/png;base64,...",
      "reference_image": "data:image/jpeg;base64,..." OR full URL OR data URL,
      "user_id": "3",
      "emp_id": "3"
    }
    """
    try:
        data = request.get_json(force=True)
        captured_data = data.get('captured_image')
        reference = data.get('reference_image')
        user_id = data.get('user_id')
        emp_id = data.get('emp_id')

        if not captured_data:
            return jsonify({"status": "error", "error": "No captured_image provided"})

        # Convert captured image to PIL
        try:
            captured_pil = b64_to_pil_image(captured_data)
        except Exception as e:
            return jsonify({"status": "error", "error": f"Invalid captured_image: {str(e)}"})

        captured_enc = find_first_encoding_from_pil(captured_pil)
        if captured_enc is None:
            return jsonify({"status": "unknown", "error": "No face found in captured image"})

        # Obtain reference image PIL
        ref_pil = None
        if reference:
            # if reference is a URL (starts with http), download it
            if isinstance(reference, str) and reference.startswith('http'):
                try:
                    r = requests.get(reference, timeout=10)
                    r.raise_for_status()
                    ref_pil = Image.open(BytesIO(r.content)).convert('RGB')
                except Exception as e:
                    return jsonify({"status": "error", "error": f"Could not download reference image: {str(e)}"})
            else:
                # assume data URL or base64
                try:
                    ref_pil = b64_to_pil_image(reference)
                except Exception as e:
                    return jsonify({"status": "error", "error": f"Invalid reference_image: {str(e)}"})
        else:
            # no reference provided
            return jsonify({"status": "error", "error": "No reference_image provided"})

        ref_enc = find_first_encoding_from_pil(ref_pil)
        if ref_enc is None:
            return jsonify({"status": "error", "error": "No face found in reference image"})

        # Compare faces
        distances = face_recognition.face_distance([ref_enc], captured_enc)
        distance = float(distances[0]) if len(distances) > 0 else None

        # threshold: 0.50 is common; you can tune it
        threshold = 0.50

        if distance is not None and distance <= threshold:
            # matched
            name = data.get('name') or f"USER_{user_id or emp_id or 'unknown'}"

            # Log locally
            log_attendance_local(name)

            # Prepare matched image (reference) as data URI to send back
            # encode ref_pil to jpeg base64
            buf = BytesIO()
            ref_pil.save(buf, format='JPEG', quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            matched_datauri = f"data:image/jpeg;base64,{b64}"

            attendance_response = None
            # Call external attendance API (use emp_id if present; else user_id)
            if emp_id:
                try:
                    attendance_url = f"https://acsdev.in/fynryx_backend/api/qr/attendance/{emp_id}"
                    now = datetime.now()
                    payload = {
                        "user_id": str(user_id or emp_id),
                        "attendance_date": now.strftime('%d-%m-%Y'),
                        "check_in_time": now.strftime('%H:%M'),
                        "check_out_time": "",
                        "status": "present",
                        "create_audit_id": "1"
                    }
                    headers = {"Content-Type": "application/json"}
                    at_resp = requests.post(attendance_url, json=payload, headers=headers, timeout=10)
                    try:
                        attendance_response = at_resp.json()
                    except:
                        attendance_response = {"status_code": at_resp.status_code, "text": at_resp.text}
                except Exception as e:
                    attendance_response = {"error": f"Attendance API call failed: {str(e)}"}

            return jsonify({
                "status": "success",
                "name": name,
                "matched_image": matched_datauri,
                "distance": distance,
                "attendance_response": attendance_response
            })
        else:
            # not matched
            return jsonify({"status": "unknown", "distance": distance, "message": "Face did not match reference."})

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
