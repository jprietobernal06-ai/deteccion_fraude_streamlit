import joblib
import pandas as pd

# Rutas
MODEL_PATH = "fraud_detection_model.pkl"
DATA_PATH = "data/test_custom.csv"

# Cargar modelo y datos
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

print("✅ Modelo cargado y datos leídos correctamente.\n")

# Columnas esperadas por el modelo
expected_features = list(model.feature_name_)
actual_features = list(df.columns)

print(f"📊 Columnas esperadas por el modelo: {len(expected_features)}")
print(f"📂 Columnas presentes en el CSV: {len(actual_features)}\n")

# Comparar
missing = [col for col in expected_features if col not in actual_features]
extra = [col for col in actual_features if col not in expected_features]

if missing:
    print("❌ Faltan las siguientes columnas en el CSV:")
    for c in missing:
        print(f"   - {c}")
else:
    print("✅ No faltan columnas.")

if extra:
    print("\n⚠️ Hay columnas extra en el CSV:")
    for c in extra:
        print(f"   - {c}")
else:
    print("\n✅ No hay columnas extra.")

print("\n---")
print("🔍 Revisa si las columnas faltantes o extra corresponden a transformaciones (ej. OneHotEncoder).")