import os
import io
import json
import re
from typing import Dict, Any, List, Optional

import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv

from google import genai
from google.genai import types


# =========================================================
# ENV / CONFIG
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
REQUIRED_FIELDS = ["hemoglobin"]

SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".pdf"]


def _get_gemini_client():
    """
    Create Gemini client only when needed.
    This avoids import-time crashes if the API key is missing.
    """
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not found. Please add it to your .env file like:\n"
            "GEMINI_API_KEY=your_actual_key_here"
        )
    return genai.Client(api_key=GEMINI_API_KEY)


# =========================================================
# PUBLIC FUNCTIONS USED BY app.py
# =========================================================

def get_missing_required_fields(values: Dict[str, Any]) -> List[str]:
    missing = []
    for field in REQUIRED_FIELDS:
        if values.get(field) is None or values.get(field) == "":
            missing.append(field)
    return missing


def extract_values_from_image(file_path: str) -> Dict[str, Any]:
    """
    Compatibility function for app.py.
    Returns only normalized values dict.
    Raises ValueError if extraction fails.
    """
    result = extract_cbc_from_file(file_path)

    if not result.get("success"):
        raise ValueError(result.get("error", "Extraction failed."))

    return result.get("values", {})


def extract_cbc_from_file(file_path: str) -> Dict[str, Any]:
    """
    Main extraction entry point used by app.py.
    Supports:
    - Images: jpg, jpeg, png, webp, bmp
    - PDF: converted locally into page images, then sent to Gemini

    Success rule:
    - If hemoglobin is extracted -> success=True
    - Otherwise -> success=False and manual confirmation is requested
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return {
            "success": False,
            "error": "Unsupported file type. Please upload an image or PDF.",
            "values": {},
            "missing_fields": REQUIRED_FIELDS,
            "raw_text_preview": ""
        }

    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": "Uploaded file was not found on the server.",
            "values": {},
            "missing_fields": REQUIRED_FIELDS,
            "raw_text_preview": ""
        }

    try:
        if ext == ".pdf":
            gemini_result = _extract_from_pdf(file_path)
        else:
            gemini_result = _extract_from_image(file_path)

        print("\n===== GEMINI RAW RESULT =====")
        print(gemini_result)

        values = _normalize_values(gemini_result.get("values", {}))
        missing_fields = get_missing_required_fields(values)
        non_null_count = sum(1 for v in values.values() if v is not None)

        if values.get("hemoglobin") is not None:
            return {
                "success": True,
                "source": gemini_result.get("source", "gemini"),
                "values": values,
                "missing_fields": missing_fields,
                "raw_text_preview": gemini_result.get("raw_text_preview", "")
            }

        if non_null_count >= 1 or gemini_result.get("raw_text_preview", "").strip():
            return {
                "success": False,
                "error": "Some report text was detected, but hemoglobin could not be extracted confidently. Please confirm manually.",
                "values": values,
                "missing_fields": missing_fields,
                "raw_text_preview": gemini_result.get("raw_text_preview", "")
            }

        return {
            "success": False,
            "error": "The uploaded report could not be confidently identified as a CBC / hematology lab report.",
            "values": values,
            "missing_fields": missing_fields if missing_fields else REQUIRED_FIELDS,
            "raw_text_preview": gemini_result.get("raw_text_preview", "")
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Gemini extraction failed: {str(e)}",
            "values": {},
            "missing_fields": REQUIRED_FIELDS,
            "raw_text_preview": ""
        }


def build_prediction_from_values(values: Dict[str, Any]) -> Dict[str, Any]:
    hb = _safe_float(values.get("hemoglobin"))
    mcv = _safe_float(values.get("mcv"))

    if hb is None:
        raise ValueError("Hemoglobin is required for prediction.")

    gender = str(values.get("gender") or "").strip().lower()
    threshold = 13.0 if gender == "male" else 12.0
    anemia_detected = hb < threshold

    if not anemia_detected:
        return {
            "prediction": "No Anemia Detected",
            "has_anemia": False,
            "summary": (
                "The available CBC values do not currently suggest anemia. "
                "Maintain a balanced diet, hydration, and routine health monitoring."
            ),
            "precautions": [
                "Continue iron-rich balanced nutrition.",
                "Maintain adequate hydration.",
                "Repeat CBC if symptoms such as fatigue or dizziness persist."
            ],
            "doctor_advice": (
                "Clinical consultation is recommended only if symptoms are present "
                "or if the patient has prior hematologic concerns."
            ),
            "risks": "No anemia-related risk is indicated from the current CBC pattern."
        }

    severity = get_severity(hb)
    anemia_type = get_likely_anemia_type(mcv)
    common_type = get_common_anemia_type(anemia_type)

    return {
        "prediction": "Anemia Detected",
        "has_anemia": True,
        "severity": severity,
        "anemia_type": anemia_type,
        "common_type": common_type,
        "precautions": get_precautions(anemia_type, severity),
        "risks": get_risks(severity),
        "doctor_advice": get_doctor_advice(severity),
        "summary": (
            f"The CBC pattern suggests {severity.lower()} anemia. "
            f"Based on MCV, the likely classification is {anemia_type.lower()}. "
            f"Common likely type: {common_type}."
        )
    }


# =========================================================
# GEMINI EXTRACTION
# =========================================================

def _extract_from_image(image_path: str) -> Dict[str, Any]:
    client = _get_gemini_client()
    mime_type = _guess_mime_type(image_path)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )

    prompt = _cbc_prompt()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    text = getattr(response, "text", "") or ""
    data = _parse_json_response(text)

    return {
        "source": "gemini_image",
        "values": data,
        "raw_text_preview": text[:2000]
    }


def _extract_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Convert up to first 3 pages of PDF into images locally,
    then send all page images in one Gemini request.
    """
    client = _get_gemini_client()
    page_images = _pdf_to_jpeg_bytes(pdf_path, max_pages=3)

    if not page_images:
        return {
            "source": "gemini_pdf",
            "values": {},
            "raw_text_preview": ""
        }

    parts = []
    for img_bytes in page_images:
        parts.append(
            types.Part.from_bytes(
                data=img_bytes,
                mime_type="image/jpeg"
            )
        )

    prompt = _cbc_prompt()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=parts + [prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    text = getattr(response, "text", "") or ""
    data = _parse_json_response(text)

    return {
        "source": "gemini_pdf",
        "values": data,
        "raw_text_preview": text[:2000]
    }


def _cbc_prompt() -> str:
    return """
You are reading a medical CBC / hematology lab report.

Extract ONLY these fields and return ONLY valid JSON:
{
  "hemoglobin": number or null,
  "rbc": number or null,
  "pcv": number or null,
  "mcv": number or null,
  "mch": number or null,
  "mchc": number or null,
  "rdw": number or null,
  "wbc": number or null,
  "platelets": number or null,
  "gender": "male" or "female" or null,
  "age": number or null
}

Rules:
1. Extract ONLY the actual patient result values.
2. Do NOT extract reference range values.
3. Do NOT guess any field.
4. If a value is unclear, missing, or ambiguous, return null.
5. Recognize synonyms:
   - hemoglobin / haemoglobin / hb / hgb
   - rbc / total rbc count
   - pcv / hct / hematocrit / haematocrit / packed cell volume
   - mcv
   - mch
   - mchc
   - rdw / rdw-cv / rbc distribution width
   - wbc / tlc / total leucocyte count / white blood cell count
   - platelets / platelet count
6. Prefer RESULT columns over UNIT or REFERENCE RANGE.
7. Ignore hospital names, report IDs, dates, doctor names, and reference intervals.
8. Output ONLY raw JSON. No markdown. No explanation.
""".strip()


def _parse_json_response(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {}


# =========================================================
# PDF / IMAGE HELPERS
# =========================================================

def _pdf_to_jpeg_bytes(pdf_path: str, max_pages: int = 3) -> List[bytes]:
    doc = fitz.open(pdf_path)
    output = []

    try:
        total = min(len(doc), max_pages)
        for page_index in range(total):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            img = _resize_if_needed(img, max_side=1800)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            output.append(buf.getvalue())
    finally:
        doc.close()

    return output


def _resize_if_needed(image: Image.Image, max_side: int = 1800) -> Image.Image:
    w, h = image.size
    longest = max(w, h)

    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size)


def _guess_mime_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp"
    }.get(ext, "image/jpeg")


# =========================================================
# NORMALIZATION
# =========================================================

def _normalize_values(values: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hemoglobin": _safe_float(values.get("hemoglobin")),
        "rbc": _safe_float(values.get("rbc")),
        "pcv": _safe_float(values.get("pcv")),
        "mcv": _safe_float(values.get("mcv")),
        "mch": _safe_float(values.get("mch")),
        "mchc": _safe_float(values.get("mchc")),
        "rdw": _safe_float(values.get("rdw")),
        "wbc": _safe_float(values.get("wbc")),
        "platelets": _safe_float(values.get("platelets")),
        "age": _safe_float(values.get("age")),
        "gender": _normalize_gender(values.get("gender")),
    }


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _normalize_gender(gender: Optional[str]) -> Optional[str]:
    if not gender:
        return None
    g = str(gender).strip().lower()
    if g in ["male", "m"]:
        return "male"
    if g in ["female", "f"]:
        return "female"
    return None


# =========================================================
# CLINICAL HELPERS
# =========================================================

def get_severity(hb: float) -> str:
    if hb < 7:
        return "Severe"
    if hb < 10:
        return "Moderate"
    return "Mild"


def get_likely_anemia_type(mcv: Optional[float]) -> str:
    if mcv is None:
        return "Unable to classify"
    if mcv < 80:
        return "Microcytic"
    if mcv > 100:
        return "Macrocytic"
    return "Normocytic"


def get_common_anemia_type(anemia_type: str) -> str:
    if anemia_type == "Microcytic":
        return "Iron Deficiency Anemia"
    if anemia_type == "Normocytic":
        return "Anemia of Chronic Disease"
    if anemia_type == "Macrocytic":
        return "Vitamin Deficiency Anemia"
    return "Unable to determine"


def get_precautions(anemia_type: str, severity: str) -> List[str]:
    base = [
        "Follow a balanced diet with iron, folate, and vitamin B12 sources.",
        "Avoid self-medicating without physician guidance.",
        "Repeat CBC and follow-up testing if advised."
    ]

    if anemia_type == "Microcytic":
        base.append("Discuss iron studies with a clinician.")
    elif anemia_type == "Macrocytic":
        base.append("Discuss vitamin B12 and folate evaluation with a clinician.")
    elif anemia_type == "Normocytic":
        base.append("Further medical evaluation may be needed to identify the exact cause.")
    else:
        base.append("Some classification details are unavailable, so clinical review is recommended.")

    if severity == "Severe":
        base.append("Seek urgent medical consultation.")

    return base


def get_risks(severity: str) -> List[str]:
    if severity == "Severe":
        return [
            "Marked fatigue",
            "Shortness of breath",
            "Reduced oxygen-carrying capacity",
            "Potential need for urgent medical care"
        ]
    if severity == "Moderate":
        return [
            "Fatigue",
            "Weakness",
            "Reduced exercise tolerance"
        ]
    return [
        "Mild tiredness",
        "Possible reduced stamina"
    ]


def get_doctor_advice(severity: str) -> str:
    if severity == "Severe":
        return "Immediate clinical evaluation is strongly recommended."
    if severity == "Moderate":
        return "A doctor consultation is recommended for cause evaluation and treatment planning."
    return "Routine doctor consultation is advisable, especially if symptoms are present."