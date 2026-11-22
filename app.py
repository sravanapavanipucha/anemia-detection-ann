
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
import json
from werkzeug.security import generate_password_hash, check_password_hash
from process_image import extract_values_from_image, get_missing_required_fields
from medical_ai import (
    has_anemia_by_hb,
    get_severity,
    get_likely_anemia_type,
    get_common_anemia_type,
    get_precautions,
    get_risks,
    get_doctor_advice,
    generate_ai_summary,
    get_no_anemia_tips,
)

app = Flask(__name__)
app.secret_key = "your_secret_key_here_123"   # change this to any strong secret key

USERS_FILE = "users.json"

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)
        
def load_users():
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf", "webp"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Simple login credentials
# VALID_USERNAME = "admin"
# VALID_PASSWORD = "1234"


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def to_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def build_anemia_response(values, gender="female"):
    hb = to_float(values.get("hemoglobin"))
    mcv = to_float(values.get("mcv"))

    if hb is None:
        return {
            "success": False,
            "error": "Hemoglobin value is required for prediction.",
            "values": values
        }, 400

    anemia_status = has_anemia_by_hb(hb, gender)

    if anemia_status is None:
        return {
            "success": False,
            "error": "Could not determine anemia status.",
            "values": values
        }, 400

    # ---------------------------
    # NO ANEMIA CASE
    # ---------------------------
    if not anemia_status:
        return {
            "success": True,
            "prediction": "No Anemia",
            "summary": generate_ai_summary(
                prediction="No Anemia Detected",
                severity=None,
                anemia_type=None,
                values=values
            ),
            "health_tips": get_no_anemia_tips(values),
            "values": values
        }, 200

    # ---------------------------
    # ANEMIA CASE
    # ---------------------------
    severity = get_severity(hb, gender)
    morph_type = get_likely_anemia_type(mcv)
    common_type = get_common_anemia_type(morph_type, hb, mcv)

    precautions = get_precautions(common_type, severity)
    risks = get_risks(severity)
    doctor_advice = get_doctor_advice(severity)

    summary = generate_ai_summary(
        prediction="Anemia Detected",
        severity=severity,
        anemia_type=morph_type,
        values=values
    )

    return {
        "success": True,
        "prediction": "Anemia Detected",
        "severity": severity,
        "type": morph_type,
        "common_type": common_type,
        "summary": summary,
        "precautions": precautions,
        "risks": risks,
        "doctor_advice": doctor_advice,
        "values": values
    }, 200


def is_logged_in():
    return session.get("user_logged_in") is True


# ---------------------------
# LOGIN ROUTES
# ---------------------------

# @app.route("/login", methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         username = request.form.get("username", "").strip()
#         password = request.form.get("password", "").strip()

#         if username == VALID_USERNAME and password == VALID_PASSWORD:
#             session["user_logged_in"] = True
#             session["username"] = username
#             return redirect(url_for("index"))
#         else:
#             return render_template("login.html", error="Invalid username or password")

#     return render_template("login.html", error=None)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not full_name or not username or not password or not confirm_password:
            return render_template("signup.html", error="Please fill all fields.")

        if password != confirm_password:
            return render_template("signup.html", error="Passwords do not match.")

        users = load_users()

        if username in users:
            return render_template("signup.html", error="Username already exists.")

        users[username] = {
            "full_name": full_name,
            "password": generate_password_hash(password)
        }

        save_users(users)
        return redirect(url_for("login"))

    return render_template("signup.html", error=None)
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()

        users = load_users()

        if username not in users:
            return render_template("login.html", error="User not found.")

        stored_password = users[username]["password"]

        if not check_password_hash(stored_password, password):
            return render_template("login.html", error="Invalid password.")

        session["user_logged_in"] = True
        session["username"] = username
        return redirect(url_for("index"))

    return render_template("login.html", error=None)



@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------
# MAIN ROUTES
# ---------------------------

@app.route("/", methods=["GET"])
def index():
    if not is_logged_in():
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/predict_manual", methods=["POST"])
def predict_manual():
    if not is_logged_in():
        return jsonify({
            "success": False,
            "error": "Please login first."
        }), 401

    try:
        data = request.get_json()

        gender = str(data.get("gender", "female")).strip().lower()

        values = {
            "hemoglobin": data.get("hemoglobin"),
            "rbc": data.get("rbc"),
            "pcv": data.get("pcv"),
            "mcv": data.get("mcv"),
            "mch": data.get("mch"),
            "mchc": data.get("mchc"),
            "rdw": data.get("rdw"),
            "wbc": data.get("wbc"),
            "platelets": data.get("platelets"),
            "gender": gender
        }

        result, status_code = build_anemia_response(values, gender)
        return jsonify(result), status_code

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Manual prediction failed: {str(e)}"
        }), 500


@app.route("/predict_image", methods=["POST"])
def predict_image():
    if not is_logged_in():
        return jsonify({
            "success": False,
            "error": "Please login first."
        }), 401

    try:
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded."
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Unsupported file type. Please upload an image or PDF."
            }), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        extracted = extract_values_from_image(file_path)

        if not extracted:
            return jsonify({
                "success": False,
                "error": "I couldn’t extract usable values from the uploaded file. Please try a clearer report or enter the values manually."
            }), 400

        if isinstance(extracted, dict) and extracted.get("success") is False:
            return jsonify(extracted), 400

        values = extracted
        missing_fields = get_missing_required_fields(values)

        if "hemoglobin" in missing_fields:
            return jsonify({
                "success": False,
                "error": "I could read part of the report, but I couldn’t confidently extract hemoglobin. Please enter the values manually or upload a clearer report.",
                "values": values,
                "missing_fields": missing_fields
            }), 400

        gender = str(values.get("gender", "female")).strip().lower()

        result, status_code = build_anemia_response(values, gender)
        result["missing_fields"] = missing_fields
        return jsonify(result), status_code

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Image/PDF prediction failed: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)