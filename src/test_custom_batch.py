import joblib
import pandas as pd

def predict_custom_batch(model_path="fraud_detection_model.pkl", csv_path="data/test_custom.csv"):
    # Cargar modelo
    model = joblib.load(model_path)
    print(f"✅ Modelo cargado desde: {model_path}")

    # Cargar datos y quitar encabezados
    df = pd.read_csv(csv_path)

    # Eliminar columna 'Class' si existe (solo se usa para validación)
    if 'Class' in df.columns:
        y_true = df['Class']
        X = df.drop(columns=['Class'])
    else:
        y_true = None
        X = df

    # Quitar nombres de columnas (usar solo los valores)
    X.columns = range(X.shape[1])

    print(f"📂 Datos cargados desde: {csv_path}")
    print(f"🔢 Total de registros: {len(X)}")

    # Predicciones
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)

    # Crear DataFrame de resultados
    results = pd.DataFrame({
        "Transacción": range(1, len(preds) + 1),
        "Probabilidad de Fraude": probs,
        "Predicción": ["FRAUDE" if p == 1 else "Normal" for p in preds]
    })

    # Precisión (si hay etiquetas)
    if y_true is not None:
        acc = (preds == y_true).mean() * 100
        print(f"✅ Precisión: {acc:.2f}%")
    else:
        acc = None
        print("⚠️ No se encontró columna 'Class', solo se generaron predicciones.")

    return results, acc


# Si quieres seguir usándolo también desde la consola:
if __name__ == "__main__":
    results, acc = predict_custom_batch()
    print("\nPredicciones (primeras 10):")
    print(results.head(10))