import joblib
import pandas as pd
from tensorflow.keras.models import load_model

from medical_ai import (
    get_severity,
    get_likely_anemia_type,
    get_precautions,
    get_doctor_advice,
    generate_ai_summary
)

# ---------------------------
# Load saved files
# ---------------------------
model = load_model("model/anemia_dl_model.keras")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")
feature_columns = joblib.load("model/features.pkl")

# ---------------------------
# Enter a test patient case here
# Change these values and run again
# ---------------------------
patient = {
    "Gender": 0,      # female = 0, male = 1
    "Age": 22,
    "Hb": 8.4,
    "RBC": 3.6,
    "WBC": 7000,
    "PLATELETS": 250000,
    "LYMP": 32,
    "MONO": 6,
    "HCT": 29,
    "MCV": 72,
    "MCH": 23,
    "MCHC": 30,
    "RDW": 16.2,
    "PDW": 14,
    "MPV": 9.2,
    "PCT": 0.24
}

# ---------------------------
# Prediction
# ---------------------------
input_df = pd.DataFrame([patient])[feature_columns]
input_scaled = scaler.transform(input_df)

prob = model.predict(input_scaled)[0][0]
pred_class = 1 if prob > 0.5 else 0
prediction_label = label_encoder.inverse_transform([pred_class])[0]

# Confidence
confidence = prob * 100 if prediction_label == "Anemia" else (1 - prob) * 100

# Convert gender text for explanation
gender_text = "female" if patient["Gender"] == 0 else "male"

severity = get_severity(patient["Hb"], gender_text)
likely_type = get_likely_anemia_type(
    hb=patient["Hb"],
    mcv=patient["MCV"],
    mch=patient["MCH"],
    mchc=patient["MCHC"],
    rdw=patient["RDW"]
)
precautions = get_precautions(severity, likely_type)
doctor_advice = get_doctor_advice(severity)

summary = generate_ai_summary(
    prediction_label=prediction_label,
    confidence=confidence,
    gender=gender_text,
    hb=patient["Hb"],
    mcv=patient["MCV"],
    mch=patient["MCH"],
    mchc=patient["MCHC"],
    rdw=patient["RDW"],
    severity=severity,
    likely_type=likely_type,
    doctor_advice=doctor_advice,
)

# ---------------------------
# Output
# ---------------------------
print("\n========== FINAL OUTPUT ==========")
print("Prediction        :", prediction_label)
print("Confidence        :", f"{confidence:.2f}%")
print("Likely Type       :", likely_type)
print("Severity          :", severity)
print("Doctor Advice     :", doctor_advice)

print("\nPrecautions:")
for i, item in enumerate(precautions, start=1):
    print(f"{i}. {item}")

print("\nAI Summary:")
print(summary)
print("==================================")