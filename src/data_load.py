import pandas as pd

# Ruta del dataset
data_path = "data/creditcard.csv"

# Cargar los datos
df = pd.read_csv(data_path)

# Mostrar información general
print("🔹 Dimensiones del dataset:", df.shape)
print("\n🔹 Primeras filas del dataset:")
print(df.head())

print("\n🔹 Información general:")
print(df.info())

print("\n🔹 Distribución de clases (0 = legítima, 1 = fraude):")
print(df["Class"].value_counts(normalize=True))