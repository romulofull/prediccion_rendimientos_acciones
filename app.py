import streamlit as st
import pandas as pd
import numpy as np
import joblib  


st.set_page_config(
    page_title="Predicción de Rendimiento de Acciones",
    layout="wide"
)
st.markdown("""
<style>
/* Number inputs más compactos */
div[data-testid="stNumberInput"] {
    max-width: 370px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Tipografía general */
html, body, [class*="css"]  {
    font-size: 17px;
}

h1 {
    font-size: 60px !important;
}
h2 {
    font-size: 50px !important;
}
h3 {
    font-size: 32px !important;
}

section[data-testid="stSidebar"] {
    width: 360px !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 70px;
    display: flex;
    flex-direction: column;
    align-items: center;
}
section[data-testid="stSidebar"] input {
    max-width: 260px;
    border-radius: 8px;
    padding: 6px;
}


table {
    font-size: 55px;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    "<h1 style='text-align:center;'>Modelo de Retornos de Precios de Acciones</h1>",
    unsafe_allow_html=True
)


st.markdown("""
<div style="margin-top:30px;">
  <h2>📌 Descripción del proyecto</h2>
  <ul>
    <li><b>Objetivo</b>: Proyectar el rendimiento diario del precio de cierre de una acción.</li>
    <li><b>Metodología</b>: Enfoque multifactorial que integra información de precios, volumen, commodities y variables macroeconómicas.</li>
  </ul>
</div>
<div style="margin-top:30px;">
<h2>🧠 Variables utilizadas por el modelo</h2>
<ul>
  <li>📈 <b>Ret_precio_apertura</b>: Rendimiento porcentual diario del precio de apertura.</li>
  <li>📊 <b>Ret_precio_maximo</b>: Rendimiento porcentual del precio máximo del día anterior.</li>
  <li>📉 <b>Ret_precio_minimo</b>: Rendimiento porcentual del precio mínimo del día anterior.</li>
  <li>🔄 <b>Ret_volumen</b>: Variación logarítmica diaria del volumen transado (rezago de un día).</li>
  <li>🌍 <b>Sp500</b>: Rendimiento porcentual diario del índice S&P 500 (rezago de un día).</li>
  <li>🛢️ <b>Ret_petroleo_usd</b>: Rendimiento porcentual diario del precio del petróleo (rezago de un día).</li>
  <li>🏦 <b>D_tasa_tesoro_10y</b>: Variación diaria de la tasa del Tesoro de EE. UU. a 10 años (rezago de un día).</li>
  <li>🔩 <b>Ret_cobre_usd</b>: Rendimiento porcentual diario del precio del cobre (rezago de un día).</li>
  <li>🏦 <b>D_tasa_tesoro_3m</b>: Variación diaria de la tasa del Tesoro de EE. UU. a 3 meses (rezago de un día).</li>
  <li>🌏 <b>Ret_usd_yuan</b>: Rendimiento diario del tipo de cambio USD/CHINA. expectativas sobre comercio global y cadenas de suministro (rezago de un día).</li>
 </div>
</ul>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:20px;">
  <h2>📥 Simulación de escenario de mercado</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    ret_precio_apertura = st.number_input("Δ% Precio Apertura (hoy)", 0.0, step=0.01)
    ret_precio_maximo   = st.number_input("Δ% Precio Máximo (ayer)", 0.0, step=0.01)
    ret_precio_minimo   = st.number_input("Δ% Precio Mínimo (ayer)", 0.0, step=0.01)

with col2:
    ret_volumen = st.number_input("Δ Volumen (ayer)", 0.0, step=0.01)
    sp500       = st.number_input("Δ% S&P 500 (ayer)", 0.0, step=0.01)
    ret_petroleo_usd = st.number_input("Δ% Petróleo (ayer)", 0.0, step=0.01)

with col3:
    d_tasa_tesoro_10y = st.number_input("Δ Tasa Tesoro 10Y (ayer)", 0.0, step=0.01)
    ret_cobre_usd     = st.number_input("Δ% Cobre (ayer)", 0.0, step=0.01)

with col4:
    d_tasa_tesoro_3m = st.number_input("Δ Tasa Tesoro 3M (ayer)", 0.0, step=0.01)
    ret_usd_yuan     = st.number_input("Δ% USD/Yuan (ayer)", 0.0, step=0.01)

if ret_volumen <= -1:
    st.warning("El cambio porcentual de volumen no puede ser menor o igual a -100%.")
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


tabla_escenario = pd.DataFrame({
    "Δ% Precio Apertura (hoy)": [ret_precio_apertura],
    "Δ% Precio Máximo (ayer)": [ret_precio_maximo],
    "Δ% Precio Mínimo (ayer)": [ret_precio_minimo],
    "Δ Log Volumen (ayer)": [log_volumen],  
    "Δ% S&P 500 (ayer)": [sp500],
    "Δ% Petróleo (ayer)": [ret_petroleo_usd],
    "Δ Tasa Tesoro 10Y (ayer)": [d_tasa_tesoro_10y],
    "Δ% Cobre (ayer)": [ret_cobre_usd],
    "Δ Tasa Tesoro 3M (ayer)": [d_tasa_tesoro_3m],
    "Δ% USD/Yuan (ayer)": [ret_usd_yuan]
}).round(4)  

st.markdown(
    "<h2 style='margin-top:30px;'>📋 Escenario de mercado utilizado por el modelo</h2>",
    unsafe_allow_html=True
)
st.dataframe(tabla_escenario, use_container_width=True, hide_index=True)
try:
    data = joblib.load("modelo_rendimientos_catboost.pkl")
    model = data["model"]
    st.markdown("""
    <div style="padding:16px; border-radius:10px; background-color:#e8f5e9;
    color:#1b5e20; font-size:18px; font-weight:600;">
    ✅ Modelo cargado 
    </div>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("No se encontró el archivo del modelo.")
    model = None
    
resultado = st.empty()
if model is not None and not input_data['ret_volumen'].isna().any():
    input_data = input_data[model.feature_names_]
    resultado_placeholder = st.empty()
    st.markdown(""" <style> div.stButton > button { width: 100%; height: 3.2em; font-size: 18px; font-weight: 600; border-radius: 10px; background-color: green; color: white; border: none; } div.stButton > button:hover { background-color: #198754; } </style> """, unsafe_allow_html=True)

    if st.button("📊 Ejecutar predicción"):
        prediccion = model.predict(input_data)

        resultado_placeholder.metric(
            label="📌 Rendimiento esperado de la acción",
            value=f"{prediccion[0] * 100:.2f} %"
        )
