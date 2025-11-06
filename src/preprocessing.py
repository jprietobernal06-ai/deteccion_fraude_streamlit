import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data():
    # Ruta del dataset original
    file_path = "data/creditcard.csv"
    df = pd.read_csv(file_path)
    print("✅ Dataset cargado correctamente")

    # Separar variables y etiquetas
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Escalado de variables (normalización)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Conjunto de entrenamiento: {X_train.shape}")
    print(f"Conjunto de prueba: {X_test.shape}")

    # Crear carpeta 'data/processed' si no existe
    os.makedirs("data/processed", exist_ok=True)

    # Guardar los datos procesados
    df_train = pd.DataFrame(X_train)
    df_train["Class"] = y_train.values
    df_test = pd.DataFrame(X_test)
    df_test["Class"] = y_test.values

    df_train.to_csv("data/processed/train_data.csv", index=False)
    df_test.to_csv("data/processed/test_data.csv", index=False)

    print("💾 Archivos guardados en 'data/processed/'")

if __name__ == "__main__":
    preprocess_data()