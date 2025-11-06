import sys
import pandas as pd
import joblib
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Uso: python src/batch_predict.py <archivo_entrada.csv> <archivo_salida.csv>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"❌ Error: El archivo {input_path} no existe.")
        sys.exit(1)

    # ==============================
    # 🔹 Cargar modelo entrenado
    # ==============================
    model_path = Path("fraud_detection_model.pkl")
    if not model_path.exists():
        print(f"❌ No se encontró el modelo entrenado en {model_path}")
        sys.exit(1)

    print("✅ Modelo cargado correctamente.")
    model = joblib.load(model_path)

    # ==============================
    # 📂 Cargar datos
    # ==============================
    print(f"📂 Cargando datos desde: {input_path}")
    data = pd.read_csv(input_path)

    # Quitar solo la columna de clase si existe
    if "Class" in data.columns:
        X = data.drop(columns=["Class"])
    else:
        X = data.copy()

    # Debug opcional: ver columnas
    print(f"🧩 Columnas cargadas ({len(X.columns)}): {list(X.columns)}")

    # ==============================
    # 🚀 Realizar predicciones
    # ==============================
    print("🚀 Realizando predicciones en lote...")
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)

    # ==============================
    # 💾 Guardar resultados
    # ==============================
    results = data.copy()
    results["Prediccion"] = preds
    results["Probabilidad_Fraude"] = probs

    results.to_csv(output_path, index=False)
    print(f"✅ Predicciones guardadas en: {output_path}")

    # ==============================
    # 📊 Resumen
    # ==============================
    resumen = results["Prediccion"].value_counts(normalize=True) * 100
    print("\n📊 Distribución de predicciones (%):")
    print(resumen)

if __name__ == "__main__":
    main()