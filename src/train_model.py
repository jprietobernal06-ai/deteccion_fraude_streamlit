import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# === Rutas ===
PROCESSED_PATH = "data/processed"
MODEL_PATH = "models"

# Crear carpeta de modelos si no existe
os.makedirs(MODEL_PATH, exist_ok=True)

# === Verificación de los archivos procesados ===
train_file = os.path.join(PROCESSED_PATH, "train_data.csv")
test_file = os.path.join(PROCESSED_PATH, "test_data.csv")

if not os.path.exists(train_file) or not os.path.exists(test_file):
    raise FileNotFoundError(
        f"No se encontraron los archivos procesados en '{PROCESSED_PATH}'. Ejecuta preprocessing.py primero."
    )

# === Cargar datos ===
print("📂 Cargando datos de entrenamiento y prueba...")
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

X_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]

X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

print(f"✅ Datos cargados: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} de prueba")

# === Entrenar modelo LightGBM ===
print("\n🚀 Entrenando modelo LightGBM...")

lgbm = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42
)

lgbm.fit(X_train, y_train)

# === Evaluar modelo ===
print("\n📊 Evaluando el modelo...")
y_pred = lgbm.predict(X_test)
y_prob = lgbm.predict_proba(X_test)[:, 1]

print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred, digits=4))

print("\n--- Matriz de Confusión ---")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"\n🔥 AUC-ROC: {roc_auc:.4f}")

# === Guardar modelo ===
model_file = os.path.join(MODEL_PATH, "lightgbm_fraud_model.pkl")
joblib.dump(lgbm, model_file)

print(f"\n💾 Modelo guardado en: {model_file}")
print("✅ Entrenamiento completado correctamente.")