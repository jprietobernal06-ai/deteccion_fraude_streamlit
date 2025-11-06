import pandas as pd
import joblib
import numpy as np

# Cargar el modelo entrenado
MODEL_PATH = "fraud_detection_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Modelo cargado correctamente.")
except FileNotFoundError:
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}. Ejecuta train_model.py primero.")


# ==============================
# 🔍 FUNCIÓN DE PREDICCIÓN
# ==============================
def predict_transaction(transaction_dict):
    """
    Recibe un diccionario con los datos de una transacción
    y devuelve la predicción del modelo (fraude o no).
    """
    df = pd.DataFrame([transaction_dict])

    # ⚙️ Asegurar que las columnas estén en el orden esperado
    try:
        expected_features = model.booster_.feature_name()
    except AttributeError:
        expected_features = [str(i) for i in range(df.shape[1])]

    # Si el modelo fue entrenado con columnas numéricas, convertir nombres
    if all(col.isdigit() for col in expected_features):
        df.columns = [str(i) for i in range(len(df.columns))]

    # Realizar la predicción
    prediction = model.predict(df)
    prob = model.predict_proba(df)[0][1]

    result = "⚠️ FRAUDE DETECTADO" if prediction[0] == 1 else "✅ Transacción legítima"
    print(f"\n📊 Resultado: {result}")
    print(f"Probabilidad de fraude: {prob:.4f}")

    return prediction[0], prob


# ==============================
# 💳 EJEMPLO DE PRUEBA
# ==============================
if __name__ == "__main__":
    # Ejemplo de transacción simulada
    sample_transaction = {
        "Time": 0.0,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470400,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62
    }

    predict_transaction(sample_transaction)