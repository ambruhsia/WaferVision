import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from xgboost import Booster
from uhm import preprocess_wafer_data

def plot_wafer_map(wafer_map):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(wafer_map, cmap='gray', interpolation='nearest')
    ax.set_title("Wafer Map Input", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)

st.set_page_config(page_title="WaferMap Defect Detection", layout="wide", page_icon="🔬")

filename = 'C:/Users/rajpu/atarashi no desu yo/templates/savemodle3.sav'
model = pickle.load(open(filename, 'rb'))

if isinstance(model, Booster):
    model.save_model('saved_model.json')
    model_loaded = Booster()
    model_loaded.load_model('saved_model.json')
else:
    st.error("Ready")

st.sidebar.header('📌 About')
st.sidebar.info("This tool helps semiconductor engineers detect defects in wafer maps using an XGBoost classifier.")

st.sidebar.header('🛠 How to Use')
st.sidebar.markdown("1️⃣ Enter wafer map data\n2️⃣ Click '🔍 Predict'\n3️⃣ View results and visualization.")

st.sidebar.header('📞 Contact')
st.sidebar.markdown("✉️ rajputamrita54@gmail.com")

st.title("🔬 WaferMap Defect Detection")
st.markdown("Predict defects in wafer maps using a pre-trained XGBoost model.")
st.header("📝 Input Wafer Map Data")

x = st.text_area("Enter Wafer Map Data", height=150, placeholder="[[0, 1, 0], [1, 0, 1], [0, 1, 0]]")

if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.wafer_shape = None

def predict():
    try:
        wafer_map = np.array(eval(x))
        st.session_state.wafer_shape = wafer_map.shape
        preprocessed_data = preprocess_wafer_data(wafer_map)
        st.session_state.prediction = model.predict(preprocessed_data)[0]
        st.session_state.wafer_map = wafer_map  # Store for plotting
    except Exception as e:
        st.session_state.prediction = f"⚠️ Error: {e}"
        st.session_state.wafer_map = None

st.button("🔍 Predict", on_click=predict)

if st.session_state.prediction is not None:
    st.markdown("---")
    if isinstance(st.session_state.prediction, str) and "Error" in st.session_state.prediction:
        st.error(st.session_state.prediction)
    else:
        st.success(f"✅ Input converted to array with shape: `{st.session_state.wafer_shape}`")
        st.info(f"📊 Predicted Defect Category: `{st.session_state.prediction}`")
        
        if st.session_state.wafer_map is not None:
            plot_wafer_map(st.session_state.wafer_map)

st.markdown("---")