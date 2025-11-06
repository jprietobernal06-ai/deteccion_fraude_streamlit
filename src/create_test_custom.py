import pandas as pd
from pathlib import Path

# Paths
data_path = Path("data/creditcard.csv")
output_path = Path("data/test_custom.csv")

print(f"📂 Cargando datos desde: {data_path}")
df = pd.read_csv(data_path)

# Verificamos que la columna 'Class' exista
if "Class" not in df.columns:
    raise ValueError("❌ El archivo no contiene la columna 'Class'.")

# Seleccionamos 10 fraudes y 10 normales
fraudes = df[df["Class"] == 1].sample(10, random_state=42)
normales = df[df["Class"] == 0].sample(10, random_state=42)

# Combinamos y mezclamos
subset = pd.concat([fraudes, normales]).sample(frac=1, random_state=42).reset_index(drop=True)

# Guardamos
output_path.parent.mkdir(parents=True, exist_ok=True)
subset.to_csv(output_path, index=False)

print(f"✅ Archivo de prueba balanceado creado: {output_path}")
print(subset["Class"].value_counts())