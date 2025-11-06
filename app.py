import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time
import math

# --- Configuración de la página ---
st.set_page_config(
    page_title="Detección de Fraude en Transacciones",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🌗 Selector de modo oscuro / claro ---
if "tema" not in st.session_state:
    st.session_state.tema = "Claro"

col_tema1, col_tema2 = st.columns([3, 1])
with col_tema2:
    cambiar_tema = st.button(
        "🌙 Cambiar a Modo Oscuro" if st.session_state.tema == "Claro" else "☀️ Cambiar a Modo Claro"
    )

if cambiar_tema:
    st.session_state.tema = "Oscuro" if st.session_state.tema == "Claro" else "Claro"
    st.rerun()

# --- Aplicar estilos CSS según el tema ---
if st.session_state.tema == "Oscuro":
    st.markdown("""
        <style>
        body, .stApp {background-color: #0e1117; color: white;}
        .stButton>button {background-color: #1f77b4; color: white; border-radius: 8px;}
        .stDataFrame, .stMarkdown, .stMetric, .stTextInput, .stDownloadButton {color: white;}
        div[data-testid="stMetricValue"] {color: #00ff88;}
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp {background-color: #f9f9f9; color: black;}
        .stButton>button {background-color: #007bff; color: white; border-radius: 8px;}
        div[data-testid="stMetricValue"] {color: #007bff;}
        </style>
        """, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712102.png", width=80)
    st.title("🧠 Detector de Fraude")
    st.markdown("""
    **Proyecto:** Detección de Fraude en Transacciones  
    **Autor:** JuanDa Prieto Bernal  
    **Modelo:** `fraud_detection_model.pkl`  
    **Versión:** 1.7  
    """)
    st.markdown("---")
    st.info("Sube tu archivo CSV para analizar las transacciones y detectar posibles fraudes.")

# --- Título principal ---
st.title("🚨 Detección de Fraude en Transacciones")

# --- Cargar modelo ---
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

try:
    model = load_model()
    st.sidebar.success("✅ Modelo cargado correctamente")
except Exception as e:
    st.sidebar.error(f"❌ Error al cargar modelo: {e}")
    st.stop()

# ==========================================================
# ⚙️ FUNCIÓN DE ESCALADO AUTOMÁTICO (solo para CSV si no está preprocesado)
# ==========================================================
def escalar_valores(df):
    df_scaled = df.copy()
    for col in df_scaled.columns:
        mean = df_scaled[col].mean()
        std = df_scaled[col].std()
        if std != 0:
            df_scaled[col] = (df_scaled[col] - mean) / std
        else:
            df_scaled[col] = 0
    return df_scaled

# ==========================================================
# 🧠 SECCIÓN PRINCIPAL
# ==========================================================
st.markdown("Carga un archivo CSV y observa las predicciones del modelo en tiempo real.")
st.subheader("📂 Sube el archivo CSV de transacciones")

uploaded_file = st.file_uploader("Arrastra o selecciona un archivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success(f"✅ Archivo cargado: {uploaded_file.name}")
    st.write(f"Total de registros: **{len(df)}**")

    st.session_state["df_original"] = df.copy()

    progress_text = "Calculando predicciones..."
    progress_bar = st.progress(0, text=progress_text)

    y_true = None
    if 'Class' in df.columns:
        y_true = df['Class']
        df = df.drop(columns=['Class'])

    df.columns = range(df.shape[1])

    for percent_complete in range(0, 101, 10):
        time.sleep(0.05)
        progress_bar.progress(percent_complete, text=progress_text)

    # --- Escalar antes de predecir (solo CSV) ---
    df_scaled = escalar_valores(df)

    probs = model.predict_proba(df_scaled)[:, 1]
    preds = (probs > 0.5).astype(int)
    progress_bar.empty()

    resultados = pd.DataFrame({
        'Transacción': range(1, len(preds) + 1),
        'Probabilidad de Fraude': probs,
        'Predicción': ['FRAUDE' if p == 1 else 'Normal' for p in preds]
    })

    st.session_state["resultados"] = resultados

    if y_true is not None:
        acc = (preds == y_true).mean() * 100
        st.metric(label="🎯 Precisión del modelo", value=f"{acc:.2f}%")

    st.subheader("📋 Resultados de las Predicciones")

    def color_pred(val):
        return 'color: red; font-weight: bold;' if val == 'FRAUDE' else 'color: green; font-weight: bold;'

    page_size = 1000
    total_rows = len(resultados)
    total_pages = max(1, math.ceil(total_rows / page_size))

    if "page_idx" not in st.session_state:
        st.session_state.page_idx = 1

    col1, col2, col3 = st.columns([1, 4, 1])

    with col1:
        if st.button("⬅️ Anterior") and st.session_state.page_idx > 1:
            st.session_state.page_idx -= 1

    with col3:
        if st.button("Siguiente ➡️") and st.session_state.page_idx < total_pages:
            st.session_state.page_idx += 1

    with col2:
        page_input = st.text_input(
            f"📄 Ir a página (1 - {total_pages}):",
            value=str(st.session_state.page_idx)
        )
        try:
            page = int(page_input)
            if page < 1:
                page = 1
            elif page > total_pages:
                page = total_pages
            st.session_state.page_idx = page
        except ValueError:
            pass

    page = st.session_state.page_idx
    start = (page - 1) * page_size
    end = min(start + page_size, total_rows)
    page_data = resultados.iloc[start:end].reset_index(drop=True)

    st.dataframe(page_data.style.map(color_pred, subset=['Predicción']))
    st.info(f"Mostrando transacciones {start+1} a {end} de {total_rows} totales. (Página {page}/{total_pages})")

    st.markdown("---")
    st.subheader("📈 Resumen Estadístico de las Predicciones")

    total = len(resultados)
    fraudes = (resultados["Predicción"] == "FRAUDE").sum()
    normales = total - fraudes
    porcentaje = (fraudes / total) * 100 if total > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total de Transacciones", total)
    col2.metric("🚨 Fraudes Detectados", fraudes)
    col3.metric("⚖️ % de Fraude", f"{porcentaje:.2f}%")

    promedio_fraude = resultados.loc[resultados["Predicción"] == "FRAUDE", "Probabilidad de Fraude"].mean()
    promedio_normal = resultados.loc[resultados["Predicción"] == "Normal", "Probabilidad de Fraude"].mean()

    st.write(f"🔴 Promedio probabilidad de fraude (casos FRAUDE): **{promedio_fraude:.2f}**")
    st.write(f"🟢 Promedio probabilidad de fraude (casos normales): **{promedio_normal:.2f}**")

    st.subheader("🧩 Distribución General")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([fraudes, normales], labels=['Fraude', 'Normal'], autopct='%1.1f%%', colors=['red', 'green'], startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("💾 Descargar Reporte")
    csv_download = resultados.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Descargar CSV con predicciones",
        data=csv_download,
        file_name="predicciones_fraude.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Sube un archivo CSV para comenzar el análisis.")


# ==========================================================
# 🧮 SECCIÓN: PREDICCIÓN MANUAL CON CASOS DE EJEMPLO
# ==========================================================
st.markdown("---")
st.header("🧮 Ingreso Manual de Datos para Predicción")
st.write("Puedes ingresar manualmente los valores de una transacción o probar con ejemplos predefinidos.")
st.info("El modelo espera **30 características numéricas.**")

# --- Botones de ejemplo ---
col1, col2, col3 = st.columns(3)
with col1:
    caso = st.button("🟢 Cargar Caso Normal")
with col2:
    fraude = st.button("🔴 Cargar Caso Fraude Real")
with col3:
    borde = st.button("🟡 Cargar Caso Borde")

# Casos predefinidos
ejemplo_normal = [0, -1.3598, -0.07278, 2.5363, 1.3782, -0.3383, 0.4624, 0.2396, 0.0987, 0.3638,
                  0.0908, -0.5516, -0.6178, -0.9914, -0.3112, 1.4682, -0.4704, 0.2080, 0.0258, 0.4040,
                  0.2514, -0.0183, 0.2778, -0.1105, 0.0669, 0.1285, -0.1891, 0.1336, -0.0211, 10.00]

ejemplo_fraude = [406.0, -2.312227, 1.951992, -1.609851, 3.997906, -0.522188, -1.426545, -2.537387, 1.391657, -2.770089,
                  -2.772272, 3.202033, -2.899907, -0.595222, -4.289254, 0.389724, -1.140747, -2.830056, -0.016822, 0.416956,
                  0.126911, 0.517232, -0.035049, -0.465211, 0.320198, 0.044519, 0.177840, 0.261145, -0.143276, 0.00]

ejemplo_borde = [120000.0, 2.50, 1.80, 2.00, 1.50, 1.20, 0.80, 0.90, 0.60, 1.10,
                 -0.80, -0.50, -0.40, 0.70, 1.90, 2.20, -0.90, 0.40, 0.30, 0.50,
                 0.20, 0.00, 0.10, -0.20, 0.00, 0.20, 0.10, 0.05, -0.02, 20000.00]

if "valores_manual" not in st.session_state:
    st.session_state.valores_manual = [0.0] * 30

# Actualizar valores y forzar recarga
if caso:
    st.session_state.valores_manual = ejemplo_normal
    st.rerun()
elif fraude:
    st.session_state.valores_manual = ejemplo_fraude
    st.rerun()
elif borde:
    st.session_state.valores_manual = ejemplo_borde
    st.rerun()

st.subheader("✍️ Introduce los valores de entrada")
cols = st.columns(4)
valores = []
for i in range(30):
    with cols[i % 4]:
        val = st.number_input(f"X{i}", value=float(st.session_state.valores_manual[i]), format="%.4f")
        valores.append(val)

if st.button("🔍 Predecir"):
    df_manual = pd.DataFrame([valores])
    
    # 🚫 No escalar aquí, el modelo ya fue entrenado con datos procesados
    prob = model.predict_proba(df_manual)[0, 1]
    pred = "FRAUDE" if prob > 0.5 else "Normal"

    st.markdown(f"### Resultado: **{'🚨 FRAUDE' if pred == 'FRAUDE' else '✅ Normal'}**")
    st.metric("Probabilidad de Fraude", f"{prob*100:.2f}%")

    if "historial_manual" not in st.session_state:
        st.session_state["historial_manual"] = []

    st.session_state["historial_manual"].append({
        "Fecha/Hora": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Predicción": pred,
        "Probabilidad (%)": round(prob*100, 2)
    })

if "historial_manual" in st.session_state and len(st.session_state["historial_manual"]) > 0:
    st.markdown("---")
    st.subheader("📜 Historial de Predicciones Manuales")
    hist_df = pd.DataFrame(st.session_state["historial_manual"])
    st.dataframe(hist_df, use_container_width=True)

    csv_hist = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Descargar historial en CSV",
        data=csv_hist,
        file_name="historial_predicciones_manual.csv",
        mime="text/csv"
    )

# --- Pie de página ---
st.markdown("---")
st.caption("🧠 Proyecto académico de detección de fraude en transacciones | © 2025 JuanDa Prieto Bernal")