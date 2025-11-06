import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os

# === 1. Cargar los datasets procesados ===
train_path = "data/processed/train_data.csv"
test_path = "data/processed/test_data.csv"

print("📂 Cargando datos...")
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

X_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]

X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

# === 2. Configurar el modelo LightGBM ===
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=32,
    random_state=42,
    class_weight="balanced"  # ayuda con el desbalanceo
)

# === 3. Entrenar el modelo ===
print("🚀 Entrenando el modelo LightGBM...")
model.fit(X_train, y_train)

# === 4. Evaluar el modelo ===
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n📊 Resultados del modelo:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# === 5. Guardar el modelo entrenado ===
output_path = "fraud_detection_model.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model, f)

print(f"\n💾 Modelo guardado como '{output_path}'")