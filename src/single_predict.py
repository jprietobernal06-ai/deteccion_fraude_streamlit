import joblib
import numpy as np
import pandas as pd
import os

# Ruta del modelo entrenado
MODEL_PATH = os.path.join("models", "lightgbm_fraud_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ No se encontró el modelo entrenado en {MODEL_PATH}")
        return None
    model = joblib.load(MODEL_PATH)
    print("✅ Modelo cargado correctamente.")
    return model

def predict_single_transaction(model, input_data):
    # Convertir a DataFrame con las columnas adecuadas
    columns = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    df = pd.DataFrame([input_data], columns=columns)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    print("\n📊 RESULTADO DE LA PREDICCIÓN:")
    print(f" - Predicción: {'💥 FRAUDE' if prediction == 1 else '✅ NO FRAUDE'}")
    print(f" - Probabilidad de fraude: {probability * 100:.2f}%")

if __name__ == "__main__":
    model = load_model()
    if model is not None:
        print("\n🧮 Ingrese los valores de la transacción (usaremos valores ejemplo si deja vacío):")

        # Valores por defecto (una transacción legítima)
        default_values = [
            -1.359807,  # V1
            -0.072781,  # V2
            2.536347,   # V3
            1.378155,   # V4
            -0.338321,  # V5
            0.462388,   # V6
            0.239599,   # V7
            0.098698,   # V8
            0.363787,   # V9
            0.090794,   # V10
            -0.5516,    # V11
            -0.617801,  # V12
            -0.99139,   # V13
            -0.311169,  # V14
            1.468177,   # V15
            -0.4704,    # V16
            0.207971,   # V17
            0.025791,   # V18
            0.403993,   # V19
            0.251412,   # V20
            -0.018307,  # V21
            0.277838,   # V22
            -0.110474,  # V23
            0.066928,   # V24
            0.128539,   # V25
            -0.189115,  # V26
            0.133558,   # V27
            -0.021053,  # V28
            149.62,     # Amount
            0.0         # Time
        ]

        # Permitir ingresar manualmente (si presionas Enter, usa el valor por defecto)
        user_values = []
        for i, col in enumerate([f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]):
            val = input(f"{col} (default {default_values[i]}): ")
            user_values.append(float(val) if val.strip() else default_values[i])

        predict_single_transaction(model, user_values)