# src/eda_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ruta del dataset
data_path = os.path.join(os.path.dirname(__file__), '../data/creditcard.csv')

# Cargar los datos
df = pd.read_csv(data_path)

print("✅ Dataset cargado correctamente para análisis exploratorio")
print(df.head())

# --- Distribución de clases ---
fraud_count = df['Class'].value_counts()
print("\nDistribución de clases:")
print(fraud_count)
print("\nPorcentajes:")
print(fraud_count / len(df) * 100)

# --- Histograma del monto (Amount) ---
plt.figure(figsize=(8,4))
sns.histplot(df['Amount'], bins=100, kde=True)
plt.title('Distribución de los montos de transacción')
plt.xlabel('Monto')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# --- Distribución del tiempo (Time) ---
plt.figure(figsize=(8,4))
sns.histplot(df['Time'], bins=100, kde=True, color='orange')
plt.title('Distribución del tiempo (segundos desde el inicio)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# --- Matriz de correlación ---
corr = df.corr(numeric_only=True)
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Matriz de correlación entre variables')
plt.tight_layout()
plt.show()