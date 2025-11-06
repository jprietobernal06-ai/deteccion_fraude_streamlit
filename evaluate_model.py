import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# === 1. Cargar el modelo entrenado ===
with open("fraud_detection_model.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Modelo cargado correctamente.\n")

# === 2. Cargar los datos de prueba ===
# Puedes cambiar esta ruta a otro dataset que quieras probar
test_data = pd.read_csv("data/creditcard.csv") 

# Tomamos las mismas columnas que se usaron para entrenar
X = test_data.drop(columns=["Class"])
y = test_data["Class"]

# === 3. Hacer predicciones ===
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

# === 4. Mostrar métricas ===
print("📊 Resultados del modelo en datos nuevos:")
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred, digits=4))
print(f"AUC-ROC: {roc_auc_score(y, y_pred_proba):.4f}")

# === 5. Ejemplo de predicción con una sola transacción ===
print("\n🔍 Prueba con una sola transacción:")
sample = X.sample(1, random_state=42)
prediction = model.predict(sample)[0]
print(f"Predicción: {'FRAUDE ⚠️' if prediction == 1 else 'Transacción normal ✅'}")
print(f"Probabilidad de fraude: {model.predict_proba(sample)[0][1]:.4f}")