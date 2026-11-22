from typing import Dict, Any, List, Optional


# =========================================================
# BASIC HELPERS
# =========================================================

def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _clean_gender(gender: Any) -> Optional[str]:
    if gender is None:
        return None
    g = str(gender).strip().lower()
    if g in ["male", "m"]:
        return "male"
    if g in ["female", "f"]:
        return "female"
    return None


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    final = []
    for item in items:
        item = str(item).strip()
        if item and item not in seen:
            final.append(item)
            seen.add(item)
    return final


# =========================================================
# INPUT VALIDATION
# =========================================================

def validate_inputs(values: Dict[str, Any]) -> List[str]:
    """
    Optional validation helper for CBC inputs.
    Keeps your project professional and avoids unrealistic values.
    """
    errors = []

    def val(key: str) -> Optional[float]:
        return _safe_float(values.get(key))

    age = val("Age")
    hb = val("Hb")
    rbc = val("RBC")
    wbc = val("WBC")
    platelets = val("PLATELETS")
    lymp = val("LYMP")
    mono = val("MONO")
    hct = val("HCT")
    mcv = val("MCV")
    mch = val("MCH")
    mchc = val("MCHC")
    rdw = val("RDW")
    pdw = val("PDW")
    mpv = val("MPV")
    pct = val("PCT")

    if age is not None and not (0 <= age <= 120):
        errors.append("Age seems invalid.")

    if hb is not None and not (3 <= hb <= 22):
        errors.append("Hemoglobin value seems invalid.")

    if rbc is not None and not (1 <= rbc <= 8):
        errors.append("RBC value seems invalid.")

    if wbc is not None and not (1 <= wbc <= 50000):
        errors.append("WBC value seems invalid.")

    if platelets is not None and not (1 <= platelets <= 1000000):
        errors.append("Platelets value seems invalid.")

    if lymp is not None and not (0 <= lymp <= 100):
        errors.append("LYMP value seems invalid.")

    if mono is not None and not (0 <= mono <= 100):
        errors.append("MONO value seems invalid.")

    if hct is not None and not (10 <= hct <= 70):
        errors.append("HCT value seems invalid.")

    if mcv is not None and not (50 <= mcv <= 130):
        errors.append("MCV value seems invalid.")

    if mch is not None and not (10 <= mch <= 40):
        errors.append("MCH value seems invalid.")

    if mchc is not None and not (20 <= mchc <= 40):
        errors.append("MCHC value seems invalid.")

    if rdw is not None and not (8 <= rdw <= 30):
        errors.append("RDW value seems invalid.")

    if pdw is not None and not (1 <= pdw <= 30):
        errors.append("PDW value seems invalid.")

    if mpv is not None and not (1 <= mpv <= 30):
        errors.append("MPV value seems invalid.")

    if pct is not None and not (0.001 <= pct <= 5):
        errors.append("PCT value seems invalid.")

    return errors


# =========================================================
# CORE CLINICAL LOGIC
# =========================================================

def has_anemia_by_hb(hb: Any, gender: Any = "female") -> Optional[bool]:
    hb_value = _safe_float(hb)
    if hb_value is None:
        return None

    g = _clean_gender(gender) or "female"
    threshold = 13.0 if g == "male" else 12.0
    return hb_value < threshold


def get_severity(hb: Any, gender: Any = None) -> Optional[str]:
    """
    Severity based on hemoglobin.
    Returns:
    Severe / Moderate / Mild / Normal / None
    """
    hb_value = _safe_float(hb)
    if hb_value is None:
        return None

    if hb_value < 7:
        return "Severe"
    elif hb_value < 10:
        return "Moderate"
    elif hb_value < 12:
        return "Mild"
    return "Normal"


def get_likely_anemia_type(mcv: Any) -> str:
    """
    Morphological classification based on MCV.
    """
    mcv_value = _safe_float(mcv)
    if mcv_value is None:
        return "Unable to classify"

    if mcv_value < 80:
        return "Microcytic"
    elif 80 <= mcv_value <= 100:
        return "Normocytic"
    elif mcv_value > 100:
        return "Macrocytic"
    return "Unable to classify"


def get_common_anemia_type(morph_type: Any, hemoglobin: Any = None, mcv: Any = None) -> str:
    """
    Maps morphological class to a commonly associated anemia explanation.
    This is NOT a final diagnosis.
    """
    if not morph_type:
        return "Unable to determine"

    mt = str(morph_type).strip().lower()

    if mt == "microcytic":
        return "Iron Deficiency Anemia"
    elif mt == "macrocytic":
        return "Vitamin Deficiency Anemia (Vitamin B12 / Folate Deficiency)"
    elif mt == "normocytic":
        return "Anemia of Chronic Disease"
    return "Unable to determine"


# =========================================================
# PATTERN EXPLANATION
# =========================================================

def explain_pattern(
    hb: Any = None,
    mcv: Any = None,
    mch: Any = None,
    mchc: Any = None,
    rdw: Any = None
) -> List[str]:
    reasons = []

    hb_value = _safe_float(hb)
    mcv_value = _safe_float(mcv)
    mch_value = _safe_float(mch)
    mchc_value = _safe_float(mchc)
    rdw_value = _safe_float(rdw)

    if hb_value is not None:
        if hb_value < 12:
            reasons.append("hemoglobin is below the expected normal range")
        else:
            reasons.append("hemoglobin is within a normal range")

    if mcv_value is not None:
        if mcv_value < 80:
            reasons.append("MCV is low, suggesting a microcytic red cell pattern")
        elif mcv_value > 100:
            reasons.append("MCV is high, suggesting a macrocytic red cell pattern")
        else:
            reasons.append("MCV is within the normal range")

    if mch_value is not None and mch_value < 27:
        reasons.append("MCH is low, indicating reduced hemoglobin content per red cell")

    if mchc_value is not None and mchc_value < 32:
        reasons.append("MCHC is low, suggesting hypochromia")

    if rdw_value is not None and rdw_value > 14.5:
        reasons.append("RDW is elevated, showing variation in red blood cell size")

    return reasons


# =========================================================
# NO ANEMIA GUIDANCE
# =========================================================

def get_no_anemia_tips(values: Optional[Dict[str, Any]] = None) -> List[str]:
    tips = [
        "Your available values do not currently suggest anemia.",
        "Continue a balanced diet that includes iron, folate, vitamin B12, and protein.",
        "Stay hydrated, sleep well, and avoid skipping meals regularly.",
        "Routine health monitoring is still a good habit, especially if symptoms continue."
    ]
    return tips


# =========================================================
# DIET + MAINTENANCE + RISKS + DOCTOR ADVICE
# =========================================================

def get_precautions(anemia_type: Any, severity: Any) -> List[str]:
    """
    Main precautions function used by app.py.
    Signature matches the helper style you already prepared:
    get_precautions(anemia_type, severity)
    """
    anemia_type = str(anemia_type or "").strip()
    severity = str(severity or "").strip()

    precautions: List[str] = []

    # ---------- TYPE-BASED DIET / GUIDANCE ----------
    if "Iron Deficiency" in anemia_type or anemia_type == "Microcytic":
        precautions.extend([
            "Include iron-rich foods such as spinach, legumes, dates, jaggery, beans, and lean meats.",
            "Take vitamin C rich foods such as lemon, orange, or amla to improve iron absorption.",
            "Avoid tea or coffee immediately after meals because they can reduce iron absorption."
        ])

    elif "Vitamin Deficiency" in anemia_type or anemia_type == "Macrocytic":
        precautions.extend([
            "Include vitamin B12 and folate rich foods such as eggs, dairy, leafy vegetables, and fortified foods.",
            "Do not ignore symptoms such as weakness, fatigue, tingling, or poor concentration.",
            "Balanced nutrition is important because macrocytic patterns may relate to vitamin deficiency."
        ])

    elif "Normocytic" in anemia_type or "Chronic Disease" in anemia_type:
        precautions.extend([
            "Monitor fatigue, weakness, and breathlessness carefully.",
            "A broader medical evaluation may be needed because this pattern can be linked to underlying health conditions.",
            "Do not depend only on diet changes if symptoms are continuing."
        ])

    else:
        precautions.extend([
            "Maintain a balanced and nutritious diet.",
            "Monitor symptoms regularly and avoid ignoring persistent fatigue or weakness."
        ])

    # ---------- SEVERITY-BASED MAINTENANCE ----------
    if severity == "Mild":
        precautions.extend([
            "Maintain hydration and proper nutrition every day.",
            "Avoid skipping meals and monitor your energy levels over time.",
            "A repeat blood test after a reasonable interval may help track improvement."
        ])

    elif severity == "Moderate":
        precautions.extend([
            "Avoid heavy physical strain and take adequate rest.",
            "Do not delay follow-up if weakness, dizziness, or breathlessness are increasing.",
            "A structured diet and proper medical review are recommended."
        ])

    elif severity == "Severe":
        precautions.extend([
            "Immediate doctor consultation is strongly recommended.",
            "Avoid exertion, prioritize rest, and seek urgent medical assessment if symptoms are significant.",
            "Diet alone is usually not enough for severe anemia and medical treatment may be required."
        ])

    return _dedupe(precautions)


def get_risks(severity: Any) -> List[str]:
    severity = str(severity or "").strip()

    if severity == "Severe":
        return [
            "Severe fatigue and marked weakness may affect normal daily activities.",
            "Breathlessness, dizziness, palpitations, or fainting may occur.",
            "If left untreated, severe anemia can increase the risk of serious health complications."
        ]

    elif severity == "Moderate":
        return [
            "Persistent tiredness and weakness may reduce routine functioning.",
            "Dizziness, low stamina, and reduced concentration may become more noticeable.",
            "Without follow-up, the condition may progress or symptoms may worsen."
        ]

    elif severity == "Mild":
        return [
            "Mild fatigue and reduced stamina may still affect daily routine.",
            "If ignored, early anemia can gradually progress."
        ]

    return []


def get_doctor_advice(severity: Any) -> str:
    severity = str(severity or "").strip()

    if severity == "Severe":
        return (
            "Immediate medical consultation is strongly recommended. "
            "This level should not be ignored, especially if there is dizziness, weakness, shortness of breath, or fainting."
        )

    elif severity == "Moderate":
        return (
            "Please consult a doctor soon for proper evaluation and treatment guidance. "
            "A follow-up review in the near term is advisable rather than waiting too long."
        )

    elif severity == "Mild":
        return (
            "A doctor consultation is advisable if symptoms persist or if repeat testing does not improve. "
            "Mild anemia often needs monitoring, diet correction, and sometimes further evaluation."
        )

    return (
        "No urgent doctor consultation is indicated from the anemia perspective, "
        "but clinical review is appropriate if symptoms are present."
    )


# =========================================================
# AI-STYLE SUMMARY GENERATION
# =========================================================

def generate_ai_summary(
    prediction: Any,
    severity: Optional[Any],
    anemia_type: Optional[Any],
    values: Dict[str, Any]
) -> str:
    """
    Main summary builder for your platform.
    This gives a more human-like explanation instead of robotic text.
    Signature matches your uploaded helper module style. :contentReference[oaicite:1]{index=1}
    """
    prediction_text = str(prediction or "").strip().lower()
    severity_text = str(severity or "").strip()
    anemia_type_text = str(anemia_type or "").strip()

    hb = values.get("hemoglobin")
    mcv = values.get("mcv")
    mch = values.get("mch")
    mchc = values.get("mchc")
    rdw = values.get("rdw")

    pattern_points = explain_pattern(hb, mcv, mch, mchc, rdw)

    # ---------- NO ANEMIA ----------
    if "no anemia" in prediction_text:
        if pattern_points:
            return (
                "The available CBC values do not currently indicate anemia. "
                f"From the observed pattern, {', '.join(pattern_points)}. "
                "At this time, the result appears reassuring. Maintain a balanced diet, good hydration, and regular health monitoring."
            )

        return (
            "The available CBC values do not currently indicate anemia. "
            "At this time, the result appears reassuring. Continue healthy food habits, hydration, and routine health monitoring."
        )

    # ---------- ANEMIA PRESENT ----------
    detail_parts: List[str] = []

    if severity_text:
        if severity_text == "Mild":
            detail_parts.append(
                "The result suggests a mild anemia pattern. This is usually an early-stage finding that still deserves attention."
            )
        elif severity_text == "Moderate":
            detail_parts.append(
                "The result suggests moderate anemia. This level is more clinically significant and should not be ignored."
            )
        elif severity_text == "Severe":
            detail_parts.append(
                "The result suggests severe anemia. This is a serious finding and requires urgent medical attention."
            )

    if anemia_type_text:
        if anemia_type_text == "Microcytic":
            detail_parts.append(
                "The red cell pattern appears microcytic, which is commonly seen with iron deficiency-related causes."
            )
        elif anemia_type_text == "Normocytic":
            detail_parts.append(
                "The red cell pattern appears normocytic, which may be associated with underlying medical conditions and may need further evaluation."
            )
        elif anemia_type_text == "Macrocytic":
            detail_parts.append(
                "The red cell pattern appears macrocytic, which can be associated with vitamin B12 or folate deficiency."
            )
        elif anemia_type_text == "Unable to classify":
            detail_parts.append(
                "A clear morphological type could not be determined from the currently available values."
            )

    if pattern_points:
        detail_parts.append(
            f"This interpretation is supported because {', '.join(pattern_points)}."
        )

    if severity_text == "Mild":
        detail_parts.append(
            "With proper nutrition, monitoring, and medical follow-up when needed, mild anemia can often be managed effectively."
        )
    elif severity_text == "Moderate":
        detail_parts.append(
            "This condition may need structured dietary correction and timely clinical review to identify the underlying cause."
        )
    elif severity_text == "Severe":
        detail_parts.append(
            "Because of the low hemoglobin severity, medical evaluation should be prioritized rather than depending on diet alone."
        )

    detail_parts.append(
        "This output is intended for AI-assisted screening guidance and should not be treated as a final medical diagnosis."
    )

    return " ".join(detail_parts)


# =========================================================
# OPTIONAL FULL RESPONSE BUILDER
# =========================================================

def build_ai_guidance_response(values: Dict[str, Any], gender: Any = "female") -> Dict[str, Any]:
    """
    Optional helper if you want one single function later.
    Your current app.py may not need this immediately, but it is useful.
    """
    hb = _safe_float(values.get("hemoglobin"))
    mcv = _safe_float(values.get("mcv"))
    clean_gender = _clean_gender(gender or values.get("gender")) or "female"

    if hb is None:
        return {
            "success": False,
            "error": "Hemoglobin value is required for prediction.",
            "values": values
        }

    anemia_status = has_anemia_by_hb(hb, clean_gender)

    if anemia_status is None:
        return {
            "success": False,
            "error": "Unable to determine anemia status.",
            "values": values
        }

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
        }

    severity = get_severity(hb, clean_gender)
    morph_type = get_likely_anemia_type(mcv)
    common_type = get_common_anemia_type(morph_type, hb, mcv)

    return {
        "success": True,
        "prediction": "Anemia Detected",
        "severity": severity,
        "type": morph_type,
        "common_type": common_type,
        "summary": generate_ai_summary(
            prediction="Anemia Detected",
            severity=severity,
            anemia_type=morph_type,
            values=values
        ),
        "precautions": get_precautions(common_type, severity),
        "risks": get_risks(severity),
        "doctor_advice": get_doctor_advice(severity),
        "values": values
    }