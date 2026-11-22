import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

os.makedirs("model", exist_ok=True)

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv("dataset/anemia.csv")
df.columns = df.columns.str.strip()

print("Original shape:", df.shape)

# Remove extra unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]

# Drop existing severity if present
if "Severity" in df.columns:
    df = df.drop(columns=["Severity"])

print("Columns after cleanup:")
print(df.columns.tolist())

feature_columns = [
    "Gender", "Age", "Hb", "RBC", "WBC", "PLATELETS",
    "LYMP", "MONO", "HCT", "MCV", "MCH", "MCHC",
    "RDW", "PDW", "MPV", "PCT"
]

target_col = "Diagnosis"
required_columns = feature_columns + [target_col]

df = df[required_columns].copy()
df = df.drop_duplicates()
df = df.dropna(subset=required_columns)

print("Shape after selecting required columns:", df.shape)

# ---------------------------
# Clean features
# ---------------------------
df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
df["Gender"] = df["Gender"].replace({
    "male": 1, "m": 1, "1": 1,
    "female": 0, "f": 0, "0": 0
})

numeric_columns = [
    "Age", "Hb", "RBC", "WBC", "PLATELETS",
    "LYMP", "MONO", "HCT", "MCV", "MCH", "MCHC",
    "RDW", "PDW", "MPV", "PCT"
]

for col in numeric_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Gender"] + numeric_columns + [target_col])
df["Gender"] = df["Gender"].astype(int)

print("Shape after numeric cleaning:", df.shape)

# ---------------------------
# Binary target mapping
# ---------------------------
def map_to_binary(label: str) -> str:
    label = str(label).strip().lower()
    anemia_keywords = ["anemia", "iron deficiency", "beta-thalassemia"]
    if any(k in label for k in anemia_keywords):
        return "Anemia"
    return "Non-Anemia"

df["Target"] = df["Diagnosis"].apply(map_to_binary)

print("\nBinary target distribution:")
print(df["Target"].value_counts())

# ---------------------------
# Prepare X and y
# ---------------------------
X = df[feature_columns].copy()
y = df["Target"].copy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoded classes:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{i} -> {label}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ---------------------------
# Class weights
# ---------------------------
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = dict(zip(classes, class_weights))

print("\nClass weights:", class_weight_dict)

# ---------------------------
# Deep learning model
# ---------------------------
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.35),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.25),

    Dense(32, activation="relu"),
    Dropout(0.20),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=1
)

# ---------------------------
# Evaluation
# ---------------------------
y_prob = model.predict(X_test_scaled, verbose=0)
y_pred = (y_prob > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

# ---------------------------
# Save artifacts
# ---------------------------
model.save("model/anemia_dl_model.keras")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")
joblib.dump(feature_columns, "model/features.pkl")

print("\nSaved:")
print("- model/anemia_dl_model.keras")
print("- model/scaler.pkl")
print("- model/label_encoder.pkl")
print("- model/features.pkl")
