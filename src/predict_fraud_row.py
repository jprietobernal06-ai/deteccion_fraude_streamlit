# src/predict_fraud_row.py
import pandas as pd
import joblib
import os

CSV_PATH = "data/creditcard.csv"
MODEL_PATH = "fraud_detection_model.pkl"  # o ajusta si tu modelo tiene otro nombre dentro de models/

# 1) Cargar dataset y tomar la primera transacción que sea fraude
df = pd.read_csv(CSV_PATH)
fraud_row = df[df["Class"] == 1].iloc[0]  # primera fila marcada como fraude

# Mostrar la fila (valores)
print("=== Fila de fraude (valores) ===")
print(fraud_row.to_string())

# 2) Cargar el modelo
if not os.path.exists(MODEL_PATH):
    # si tu modelo está en models/, intenta cargarlo desde allí
    if os.path.exists(os.path.join("models", "fraud_detection_model.pkl")):
        MODEL_PATH = os.path.join("models", "fraud_detection_model.pkl")
    elif os.path.exists(os.path.join("models", "lightgbm_fraud_model.pkl")):
        MODEL_PATH = os.path.join("models", "lightgbm_fraud_model.pkl")
    else:
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH} ni en models/")

model = joblib.load(MODEL_PATH)
print(f"\n✅ Modelo cargado desde: {MODEL_PATH}")

# 3) Preparar la fila para la predicción (quitar Class)
X = fraud_row.drop(labels=["Class"]).to_frame().T

# 4) Predecir
pred = model.predict(X)[0]
prob = model.predict_proba(X)[0][1]

print("\n--- Resultado de la predicción sobre la fila real de fraude ---")
print(f"Predicción (0=no fraude, 1=fraude): {pred}")
print(f"Probabilidad de fraude: {prob:.6f}")