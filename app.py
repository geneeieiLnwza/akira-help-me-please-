"""
FarmTwin v2 — Full Dashboard (Streamlit)
Multi-tab interactive Digital Twin Agriculture Simulator
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
sys.path.insert(0, '.')

from farmtwin.simulation import simulate, run_all_scenarios, predict_future, PREDEFINED_SCENARIOS
from farmtwin.decision import recommend_fertilizer, recommend_crop, assess_risk

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(page_title="FarmTwin v2", page_icon="🌱", layout="wide")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e; color: white; border-radius: 8px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] { background-color: #16813d; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px; border-radius: 12px; text-align: center;
        border: 1px solid #0f3460;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Models ──────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    rf = joblib.load('models/random_forest.pkl')
    lr = joblib.load('models/linear_regression.pkl')
    ann = joblib.load('models/neural_network.pkl')
    stacking = joblib.load('models/stacking_meta.pkl')
    encoder = joblib.load('models/encoder.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return rf, lr, ann, stacking, encoder, scaler

try:
    rf_model, lr_model, ann_model, stacking_meta, encoder, scaler = load_all_models()
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
    st.info("กรุณารัน `python3 farmtwin/model_layer.py` ก่อนครับ")
    st.stop()

# Use RF as default model for simulation
model = rf_model


# ─── Sidebar: Environment Inputs ─────────────────────────────────
st.sidebar.title("🌍 สภาพแวดล้อม")
crop = st.sidebar.selectbox("🌾 ชนิดพืช", ['Rice', 'Wheat', 'Maize', 'Soybean'])
season = st.sidebar.selectbox("📅 ฤดูกาล", ['Kharif', 'Rabi', 'Zaid'])
location = st.sidebar.selectbox("📍 พื้นที่", ['Region_North', 'Region_South', 'Region_East', 'Region_West', 'Region_Central'])
soil_type = st.sidebar.selectbox("🪨 ประเภทดิน", ['Clay', 'Loam', 'Sandy', 'Silt'])

st.sidebar.divider()
st.sidebar.subheader("🌦️ สภาพอากาศ")
temp = st.sidebar.slider("อุณหภูมิ (°C)", 10.0, 45.0, 27.0, 0.5)
rainfall = st.sidebar.slider("ปริมาณฝน (mm)", 0.0, 2000.0, 800.0, 10.0)
humidity = st.sidebar.slider("ความชื้น (%)", 0.0, 100.0, 70.0, 1.0)
soil_moisture = st.sidebar.slider("ความชื้นในดิน (%)", 0.0, 100.0, 40.0, 1.0)

st.sidebar.divider()
st.sidebar.subheader("🧪 การจัดการฟาร์ม")
irrigation = st.sidebar.slider("ชลประทาน (mm)", 0.0, 1000.0, 300.0, 10.0)
n_fert = st.sidebar.slider("ปุ๋ย N (kg/ha)", 0.0, 300.0, 120.0, 5.0)
p_fert = st.sidebar.slider("ปุ๋ย P (kg/ha)", 0.0, 150.0, 40.0, 5.0)
k_fert = st.sidebar.slider("ปุ๋ย K (kg/ha)", 0.0, 150.0, 40.0, 5.0)

# Base parameters dict
base_params = {
    'Crop_Type': crop, 'Season': season, 'Location': location, 'Soil_Type': soil_type,
    'Temperature_C': temp, 'Rainfall_mm': rainfall, 'Humidity_pct': humidity,
    'Soil_Moisture_pct': soil_moisture, 'Irrigation_mm': irrigation,
    'N_Fertilizer': n_fert, 'P_Fertilizer': p_fert, 'K_Fertilizer': k_fert, 'Year': 2024
}


# ─── Header ───────────────────────────────────────────────────────
st.title("🌱 FarmTwin v2: Digital Twin Agriculture Simulator")
st.caption("AI-powered simulation system for smart farming decisions — อิงงานวิจัย 10 Paper")


# ─── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Yield Prediction",
    "🔄 What-If Simulation",
    "📈 Scenario Analysis",
    "🔮 Future Prediction",
    "💡 Decision Support",
    "🏆 Model Comparison"
])


# ═══════════ TAB 1: YIELD PREDICTION ═════════════════════════════
with tab1:
    st.header("📊 การพยากรณ์ผลผลิต")

    baseline, predicted, _ = simulate(model, encoder, scaler, base_params)

    col1, col2, col3 = st.columns(3)
    col1.metric("🌾 ผลผลิตที่คาดการณ์", f"{predicted:,.0f} kg/ha")
    col2.metric("🌡️ อุณหภูมิ", f"{temp}°C")
    col3.metric("💧 น้ำรวม", f"{rainfall + irrigation:,.0f} mm")

    st.divider()
    st.subheader("📋 สรุปสภาพแวดล้อม")
    info_df = pd.DataFrame([base_params]).T
    info_df.columns = ['ค่า']
    st.dataframe(info_df, use_container_width=True)


# ═══════════ TAB 2: WHAT-IF SIMULATION ═══════════════════════════
with tab2:
    st.header("🔄 จำลองสถานการณ์ What-If")
    st.info("ปรับค่าด้านล่างเพื่อดูว่าผลผลิตจะเปลี่ยนไปอย่างไร ถ้าเปลี่ยนปัจจัยบางอย่าง")

    c1, c2 = st.columns(2)
    with c1:
        rain_change = st.slider("เปลี่ยนฝน (%)", -80, 80, 0, 5, key='wif_rain')
        irr_change = st.slider("เปลี่ยนชลประทาน (%)", -80, 80, 0, 5, key='wif_irr')
    with c2:
        n_change = st.slider("เปลี่ยนปุ๋ย N (%)", -80, 80, 0, 5, key='wif_n')
        temp_change = st.slider("เปลี่ยนอุณหภูมิ (°C)", -5.0, 5.0, 0.0, 0.5, key='wif_temp')

    changes = {}
    if rain_change != 0: changes['Rainfall_mm'] = f'{rain_change}%'
    if irr_change != 0: changes['Irrigation_mm'] = f'{irr_change}%'
    if n_change != 0: changes['N_Fertilizer'] = f'{n_change}%'
    if temp_change != 0: changes['Temperature_C'] = temp_change

    if changes:
        base_y, sim_y, diff = simulate(model, encoder, scaler, base_params, changes)
        pct = (diff / (base_y + 1)) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("เดิม (Baseline)", f"{base_y:,.0f} kg/ha")
        c2.metric("หลังเปลี่ยน (Simulated)", f"{sim_y:,.0f} kg/ha", f"{diff:+,.0f}")
        c3.metric("เปลี่ยนไป", f"{pct:+.1f}%", delta_color="normal")

        # Bar chart
        chart_data = pd.DataFrame({'สถานการณ์': ['เดิม', 'หลังเปลี่ยน'], 'ผลผลิต (kg/ha)': [base_y, sim_y]})
        st.bar_chart(chart_data.set_index('สถานการณ์'))
    else:
        st.warning("กรุณาปรับค่าอย่างน้อย 1 ตัวเพื่อดูผลจำลอง")


# ═══════════ TAB 3: SCENARIO ANALYSIS ════════════════════════════
with tab3:
    st.header("📈 การวิเคราะห์สถานการณ์ (Scenario Analysis)")

    results_df = run_all_scenarios(model, encoder, scaler, base_params)
    results_df.columns = ['สถานการณ์', 'ผลผลิตเดิม', 'ผลผลิตจำลอง', 'ส่วนต่าง', 'เปลี่ยน (%)']
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # Chart
    chart = results_df[['สถานการณ์', 'ผลผลิตเดิม', 'ผลผลิตจำลอง']].set_index('สถานการณ์')
    st.bar_chart(chart)


# ═══════════ TAB 4: FUTURE PREDICTION ════════════════════════════
with tab4:
    st.header("🔮 พยากรณ์อนาคต (Time Simulation)")
    years = st.slider("จำนวนปีที่ต้องการพยากรณ์", 1, 10, 5, key='future_years')

    future_df = predict_future(model, encoder, scaler, base_params, years)
    future_df.columns = ['ปี', 'ผลผลิตคาดการณ์ (kg/ha)', 'อุณหภูมิเพิ่ม', 'ฝนเปลี่ยน']

    st.dataframe(future_df, use_container_width=True, hide_index=True)
    st.line_chart(future_df.set_index('ปี')['ผลผลิตคาดการณ์ (kg/ha)'])


# ═══════════ TAB 5: DECISION SUPPORT ═════════════════════════════
with tab5:
    st.header("💡 ระบบช่วยตัดสินใจ (Decision Support)")

    d1, d2 = st.columns(2)

    with d1:
        st.subheader("🧪 แนะนำปุ๋ย N ที่เหมาะสม")
        fert_rec = recommend_fertilizer(model, encoder, scaler, base_params)
        st.success(fert_rec['advice'])
        st.metric("ปุ๋ย N ที่แนะนำ", f"{fert_rec['optimal_N']} kg/ha")
        st.metric("ผลผลิตที่คาดหวัง", f"{fert_rec['expected_yield']:,.0f} kg/ha")

        # N response curve
        curve = fert_rec['curve_data']
        st.line_chart(curve.set_index('N_Fertilizer')['Predicted_Yield'])

    with d2:
        st.subheader("🌾 แนะนำชนิดพืช")
        crop_rec = recommend_crop(model, encoder, scaler, base_params)
        st.success(crop_rec['advice'])
        st.dataframe(crop_rec['comparison'], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("⚠️ ประเมินความเสี่ยง")
    risk = assess_risk(model, encoder, scaler, base_params)
    r1, r2, r3 = st.columns(3)
    r1.metric("ระดับความเสี่ยง", risk['risk_level'])
    r2.metric("Best Case", f"{risk['best_yield']:,.0f} kg/ha")
    r3.metric("Worst Case", f"{risk['worst_yield']:,.0f} kg/ha")
    st.info(f"💡 {risk['recommendation']}")


# ═══════════ TAB 6: MODEL COMPARISON ═════════════════════════════
with tab6:
    st.header("🏆 เปรียบเทียบโมเดล (Model Comparison)")
    st.caption("ใช้ Time-based validation (Train <2022, Test ≥2022) ตาม Paper 8")

    comparison_data = {
        'Model': ['Baseline (Mean)', 'Linear Regression', 'Random Forest', 'Neural Network (ANN)', 'Stacking (RF+ANN)'],
        'RMSE': [946.54, 530.64, 255.96, 242.97, 261.79],
        'R²': [-0.0125, 0.6818, 0.9260, 0.9333, 0.9226],
        'Status': ['❌ Weak', '⚠️ Fair', '✅ Strong', '✅ Best', '✅ Strong']
    }
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Bar chart for R²
    r2_chart = comp_df[['Model', 'R²']].set_index('Model')
    st.bar_chart(r2_chart)

    st.success("📌 ANN ให้ R² สูงสุด (0.933) แต่ Random Forest (0.926) ก็ใกล้เคียงมากและเสถียรกว่า")
    st.info("📖 อ้างอิง Paper 8: โมเดลทุกตัวของเราสามารถเอาชนะ Baseline (Mean Yield) ได้อย่างชัดเจน ซึ่งพิสูจน์ว่า ML มีประสิทธิภาพจริง")


# ─── Footer ───────────────────────────────────────────────────────
st.divider()
st.caption("🌱 FarmTwin v2 — Digital Twin-based AI for Agriculture | Conference Paper Prototype")
