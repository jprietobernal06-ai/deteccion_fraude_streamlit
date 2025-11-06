# src/run_tests.py
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Buscar modelo en rutas comunes
MODEL_CANDIDATES = [
    "fraud_detection_model.pkl",
    os.path.join("models", "fraud_detection_model.pkl"),
    os.path.join("models", "lightgbm_fraud_model.pkl"),
    os.path.join("models", "model.pkl"),
    os.path.join("models", "model_lgbm.pkl"),
]

def find_model():
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            print(f"✅ Cargando modelo desde: {p}")
            return joblib.load(p)
    raise FileNotFoundError("No se encontró el modelo en rutas esperadas. Entrena y guarda el modelo primero.")

def predict_row(model, row_df):
    """Recibe un dataframe de una fila (con columnas Time,V1..V28,Amount) y devuelve pred y prob."""
    X = row_df.copy()
    if "Class" in X.columns:
        X = X.drop(columns=["Class"])
    # Asegurarse del orden: Time, V1..V28, Amount -> modelo fue entrenado con Time+V1..V28+Amount
    # Si las columnas no están en ese orden, reordenamos si es posible
    expected = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
    if all(col in X.columns for col in expected):
        X = X[expected]
    # predict
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return int(pred), float(prob)

def main():
    model = find_model()

    # Cargar dataset
    csv_path = os.path.join("data", "creditcard.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró {csv_path}")
    df = pd.read_csv(csv_path)

    print("\n=== Caso 1: fila real FRAUDE (primera encontrada) ===")
    fraud_row = df[df["Class"]==1].iloc[0]
    fraud_df = fraud_row.to_frame().T
    pred_f, prob_f = predict_row(model, fraud_df)
    print(f"Predicción: {pred_f} (esperado 1). Probabilidad: {prob_f:.6f}")

    print("\n=== Caso 2: fila real NO FRAUDE (primera encontrada) ===")
    nonfraud_row = df[df["Class"]==0].iloc[0]
    nf_df = nonfraud_row.to_frame().T
    pred_nf, prob_nf = predict_row(model, nf_df)
    print(f"Predicción: {pred_nf} (esperado 0). Probabilidad: {prob_nf:.6f}")

    print("\n=== Caso 3: transacción inventada (extrema) ===")
    # orden: V1..V28, Amount, Time  OR Time first — vamos a crear DataFrame con columnas esperadas
    invented = {
        "Time": 120000.0,
        **{f"V{i}": v for i,v in enumerate([
            2.5,1.8,2.0,1.5,1.2,0.8,0.9,0.6,1.1,-0.8,-0.5,-0.4,0.7,1.9,2.2,-0.9,0.4,0.3,0.5,0.2,0.0,0.1,-0.2,0.0,0.2,0.1,0.05,-0.02
        ], start=1)},
        "Amount": 20000.0
    }
    inv_df = pd.DataFrame([invented])
    pred_i, prob_i = predict_row(model, inv_df)
    print(f"Predicción (inventada): {pred_i}. Probabilidad: {prob_i:.6f}")

    # === Prueba masiva: 10 fraudes reales + 10 no-fraudes reales ===
    print("\n=== Caso 4: lote pequeño (10 fraudes + 10 normales) y gráfico de probabilidades ===")
    fr = df[df["Class"]==1].sample(10, random_state=42)
    nofr = df[df["Class"]==0].sample(10, random_state=42)
    small = pd.concat([fr, nofr], ignore_index=True)
    X_small = small.drop(columns=["Class"])
    # asegurarse orden esperado
    expected = ["Time"] + [f"V{i}" for i in range(1,29)] + ["Amount"]
    if all(col in X_small.columns for col in expected):
        X_small = X_small[expected]
    probs_small = model.predict_proba(X_small)[:,1]
    preds_small = model.predict(X_small)
    # mostrar tabla
    out = small.copy()
    out["Pred"] = preds_small
    out["Prob"] = probs_small
    print(out[["Pred","Prob","Class"]])

    # guardar gráfico
    os.makedirs("reports", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(len(probs_small)), probs_small)
    ax.set_xlabel("Índice (10 fraudes + 10 normales)")
    ax.set_ylabel("Probabilidad de fraude")
    ax.set_title("Probabilidades para lote pequeño")
    plt.tight_layout()
    png_path = os.path.join("reports","probs_small.png")
    fig.savefig(png_path)
    print(f"\n✅ Gráfico guardado en: {png_path}")

if __name__ == "__main__":
    main()