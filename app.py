# ============================================================
#   CROP RECOMMENDATION SYSTEM — Streamlit Web App
# ============================================================
#   Run this after training the model:
#     1. python train_model.py     (generates crop_model.pkl)
#     2. streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Advisor",
    page_icon="🌾",
    layout="wide",
)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("crop_model.pkl"):
        st.error("❌ Model not found. Run `python train_model.py` first.")
        st.stop()
    with open("crop_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Crop metadata ─────────────────────────────────────────────
CROP_INFO = {
    "rice"       : {"icon": "🌾", "tip": "Grows best in flooded paddy fields. Needs warm temps and abundant water."},
    "maize"      : {"icon": "🌽", "tip": "Versatile cereal crop. Ensure good drainage and moderate rainfall."},
    "chickpea"   : {"icon": "🫘", "tip": "Drought-tolerant pulse. Ideal for drier climates with well-drained soil."},
    "kidneybeans": {"icon": "🫘", "tip": "Warm season legume. Needs fertile, well-drained loam."},
    "pigeonpeas" : {"icon": "🌿", "tip": "Drought-resistant. Great for tropical semi-arid regions."},
    "mothbeans"  : {"icon": "🌿", "tip": "Extremely drought-tolerant. Grows well in sandy soils."},
    "mungbean"   : {"icon": "🫘", "tip": "Short-duration pulse. Good for crop rotation."},
    "blackgram"  : {"icon": "🫘", "tip": "High protein content. Grows well in tropical humid climate."},
    "lentil"     : {"icon": "🫘", "tip": "Cool-season pulse. Prefers well-drained loam or clay-loam soil."},
    "pomegranate": {"icon": "🍎", "tip": "Drought-tolerant fruit. Grows in arid and semi-arid regions."},
    "banana"     : {"icon": "🍌", "tip": "Tropical fruit requiring warm, humid conditions and fertile soil."},
    "mango"      : {"icon": "🥭", "tip": "Thrives in tropical dry conditions. Needs long dry season for flowering."},
    "grapes"     : {"icon": "🍇", "tip": "Grows well in sandy loam with high potassium. Requires dry summers."},
    "watermelon" : {"icon": "🍉", "tip": "Warm season fruit. Needs warm temp and well-drained sandy soil."},
    "muskmelon"  : {"icon": "🍈", "tip": "Warm season crop. Grows best in sandy loam with good drainage."},
    "apple"      : {"icon": "🍎", "tip": "Temperate fruit. Needs cold winters for dormancy and chilling."},
    "orange"     : {"icon": "🍊", "tip": "Subtropical citrus. Needs mild winters and warm summers."},
    "papaya"     : {"icon": "🍑", "tip": "Fast-growing tropical fruit. Sensitive to waterlogging."},
    "coconut"    : {"icon": "🥥", "tip": "Coastal tropical crop. Needs very high humidity and warm temperature."},
    "cotton"     : {"icon": "🌿", "tip": "Cash crop for warm semi-arid conditions. Needs high nitrogen."},
    "jute"       : {"icon": "🌿", "tip": "Grows fast in fertile alluvial soil under warm, humid climate."},
    "coffee"     : {"icon": "☕", "tip": "Best in highland tropical regions with moderate, well-distributed rainfall."},
}

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("🌾 Crop Advisor")
st.sidebar.markdown("Enter your soil and climate data to get a personalised crop recommendation.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** 2,200 samples · 22 crops · 7 features")
st.sidebar.markdown("**Model:** Random Forest · ~99% accuracy")

# ── Main header ───────────────────────────────────────────────
st.title("🌾 Crop Recommendation System")
st.markdown("Adjust the sliders below to match your local soil and climate, then click **Recommend Crop**.")

st.markdown("---")

# ── Input form ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧪 Soil Nutrients")
    N        = st.slider("Nitrogen (N) — mg/kg",        0,   140,  60)
    P        = st.slider("Phosphorus (P) — mg/kg",      5,   145,  50)
    K        = st.slider("Potassium (K) — mg/kg",       5,   205,  40)
    ph       = st.slider("Soil pH Level",              3.5,  10.0, 6.5, step=0.1)

with col2:
    st.subheader("☁️ Climate Conditions")
    temp     = st.slider("Temperature (°C)",           8.0,  44.0, 25.0, step=0.5)
    humidity = st.slider("Humidity (%)",              14.0, 100.0, 70.0, step=0.5)
    rainfall = st.slider("Rainfall (mm)",             20.0, 300.0,100.0, step=1.0)

st.markdown("---")

# ── Input summary ─────────────────────────────────────────────
with st.expander("📋 Your current input values"):
    summary = pd.DataFrame({
        "Feature"     : ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature", "Humidity", "pH", "Rainfall"],
        "Your Value"  : [N, P, K, temp, humidity, ph, rainfall],
        "Unit"        : ["mg/kg", "mg/kg", "mg/kg", "°C", "%", "—", "mm"],
        "Typical Range": ["0–140", "5–145", "5–205", "8–44", "14–100", "3.5–10", "20–300"],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

# ── Predict ───────────────────────────────────────────────────
if st.button("🌱 Recommend Crop", use_container_width=True, type="primary"):
    input_arr  = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    prediction = model.predict(input_arr)[0]
    proba_arr  = model.predict_proba(input_arr)[0]
    confidence = proba_arr.max()

    info = CROP_INFO.get(prediction, {"icon": "🌿", "tip": "A great choice for these conditions."})

    st.markdown("---")
    st.success(f"### {info['icon']} Recommended Crop: **{prediction.upper()}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Crop",       prediction.capitalize())
    c2.metric("Confidence", f"{confidence:.1%}")
    c3.metric("Model",      "Random Forest")

    st.info(f"💡 **Tip:** {info['tip']}")

    # Top 5 alternatives
    top5_idx   = proba_arr.argsort()[::-1][:5]
    top5_crops = model.classes_[top5_idx]
    top5_conf  = proba_arr[top5_idx]

    st.markdown("#### 📊 Top 5 Predictions")
    chart_data = pd.DataFrame({
        "Crop"      : [c.capitalize() for c in top5_crops],
        "Confidence": [round(v * 100, 1) for v in top5_conf],
    })
    st.bar_chart(chart_data.set_index("Crop"), color="#1D9E75")

    st.balloons()

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;font-size:12px'>"
    "Built with Python · scikit-learn · Streamlit &nbsp;|&nbsp; "
    "Dataset: Kaggle Crop Recommendation"
    "</div>",
    unsafe_allow_html=True,
)
