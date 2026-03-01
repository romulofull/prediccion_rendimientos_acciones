import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ---------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------

st.set_page_config(
    page_title="Sistema de Proyección Financiera",
    layout="wide",
)

# ---------------------------------------------------
# ESTILO ULTRA COMPACTO – HEDGE FUND STYLE
# ---------------------------------------------------

st.markdown("""
<style>
.stDivider {
    margin-top: 0px !important;
    margin-bottom: 0px !important;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}            
/* ===== FONDO GENERAL ===== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0b1220, #111827);
}

/* ===== CONTENEDOR PRINCIPAL ULTRA COMPACTO ===== */
.block-container {
    max-width: 1300px;
    margin: 10px auto;
    padding: 25px 40px 35px 40px;
    background: linear-gradient(145deg, #ffffff, #f8fafc);
    border-radius: 18px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.35);
}

/* ===== OCULTAR ELEMENTOS STREAMLIT ===== */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ===== TÍTULO ===== */
h1 {
    text-align: center;
    font-size: 48px !important;
    margin-top: -10px !important;
    margin-bottom: 0px !important;
    color: #1e293b;
    font-weight: 700;
}

/* ===== SUBTÍTULO ===== */
h3 {
    margin-top: 0px !important;
    margin-bottom: 5px !important;
    font-size: 40px !important;
    font-weight: 600;
}
            
      


/* ===== INPUTS COMPACTOS ===== */
div[data-baseweb="input"] {
    height: 38px;
}

label {
    font-size: 26px !important;
    font-weight: 600 !important;
}
div[data-testid="stNumberInput"] p {
    font-size: 17px !important;
    font-weight: 700 !important;
}
span {
    font-size: 18px !important;
}

/* ===== BOTÓN ===== */
div.stButton > button {
    width: 100%;
    height: 48px;
    font-size: 16px;
    font-weight: 700;
    border-radius: 8px;
    background-color: #0f172a;
    color: white;
    border: none;
    transition: 0.2s ease;
    margin-bottom: 5px !important;
}

div.stButton > button:hover {
    background-color: #1e293b;
}

/* ===== RENDIMIENTO ===== */
.rendimiento-boton {
    margin-top: 10px;
    padding: 14px 30px;
    font-size: 30px;
    border-radius: 30px;
    background: linear-gradient(135deg, #111827, #1f2937);
    color: white;
    text-align: center;
    font-weight: 700;
}

/* ===== ANALYSIS BOX ===== */
.analysis-box {
    margin-top:10px;
    padding:14px;
    border-radius:10px;
    background: #0f172a;
    color:white;
    text-align:center;
    font-size:15px;
}

/* ===== ESCENARIOS ===== */
.escenario-alcista {
    margin-top: 15px;
    padding: 16px;
    border-radius: 10px;
    background: linear-gradient(135deg, #065f46, #047857);
    color: white;
    font-size: 26px;
    font-weight: 700;
    text-align: center;
}

.escenario-bajista {
    margin-top: 15px;
    padding: 16px;
    border-radius: 10px;
    background: linear-gradient(135deg, #7f1d1d, #b91c1c);
    color: white;
    font-size: 26px;
    font-weight: 700;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TÍTULO
# ---------------------------------------------------

st.markdown("<h1>📊 Modelo de Predicción de Rendimientos de Acciones </h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:left; font-size:25px; margin-top:-5px;'>Modelo con Factores Macroeconómicos</div>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------
# INPUTS
# ---------------------------------------------------

st.markdown("<h3>📥 Simulación de Escenario</h3>", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    ret_precio_apertura = st.number_input("Δ% Precio Apertura", 0.0, step=0.01)
    ret_volumen = st.number_input("Δ Volumen", 0.0, step=0.01)

with col2:
    ret_precio_maximo = st.number_input("Δ% Precio Máximo", 0.0, step=0.01)
    sp500 = st.number_input("Δ% S&P 500", 0.0, step=0.01)

with col3:
    ret_precio_minimo = st.number_input("Δ% Precio Mínimo", 0.0, step=0.01)
    ret_petroleo_usd = st.number_input("Δ% Petróleo", 0.0, step=0.01)

with col4:
    d_tasa_tesoro_10y = st.number_input("Δ Tasa 10Y", 0.0, step=0.01)
    ret_cobre_usd = st.number_input("Δ% Cobre", 0.0, step=0.01)

with col5:
    d_tasa_tesoro_3m = st.number_input("Δ Tasa 3M", 0.0, step=0.01)
    ret_usd_yuan = st.number_input("Δ% USD/YUAN", 0.0, step=0.01)

# Transformación volumen
if ret_volumen <= -1:
    st.warning("El cambio en volumen no puede ser ≤ -100%.")
    log_volumen = np.nan
else:
    log_volumen = np.log1p(ret_volumen)

input_data = pd.DataFrame({
    'ret_precio_apertura': [ret_precio_apertura],
    'ret_precio_maximo': [ret_precio_maximo],
    'ret_precio_minimo': [ret_precio_minimo],
    'ret_volumen': [log_volumen],
    'sp500': [sp500],
    'ret_petroleo_usd': [ret_petroleo_usd],
    'd_tasa_tesoro_10y': [d_tasa_tesoro_10y],
    'ret_cobre_usd': [ret_cobre_usd],
    'd_tasa_tesoro_3m': [d_tasa_tesoro_3m],
    'ret_usd_yuan': [ret_usd_yuan]
})

# ---------------------------------------------------
# CARGAR MODELO
# ---------------------------------------------------

try:
    data = joblib.load("modelo_rendimientos_catboost.pkl")
    model = data["model"]
except:
    st.error("❌ Modelo no encontrado")
    model = None

# ---------------------------------------------------
# PREDICCIÓN
# ---------------------------------------------------

if model is not None and not input_data['ret_volumen'].isna().any():

    input_data = input_data[model.feature_names_]

    if st.button("🚀 Ejecutar Predicción"):

        prediccion = model.predict(input_data)
        retorno = prediccion[0] * 100

        st.divider()
        st.markdown("<div style='margin-top:0px; margin-bottom:5px; font-size:28px; font-weight:600;'>📌 Resultado del Modelo</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:center; font-size: 50px;">
            <div class="rendimiento-boton">
                Rendimiento Esperado: {retorno:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure(go.Indicator( 
            mode="gauge+number", 
            value=retorno, 
            title={'text': "Rendimiento Diario Esperado (%)"}, 
            gauge={ 'axis': {'range': [-5, 5]}, 
            'bar': {'color': "#22c55e" if retorno > 0 else "#ef4444"}, 
            'steps': [ 
                {'range': [-5, 0], 'color': "#3f1d1d"}, 
                {'range': [0, 5], 'color': "#0f3d2e"} ], } )) 
        fig.update_layout(template="plotly_dark", height=220, margin=dict(l=10, r=10, t=30, b=10)) 
        st.plotly_chart(fig, use_container_width=True)

    

        if retorno > 0:
            st.markdown("""
            <div class="escenario-alcista">
                📈 Recomendación de Asignación: POSICIÓN LARGA RECOMENDADA
            </div>
            """, unsafe_allow_html=True)
        elif retorno < 0:
            st.markdown("""
            <div class="escenario-bajista">
                📉 MODO PRESERVACIÓN DE CAPITAL: EVITAR ENTRADA
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Señal de Mercado Neutral")