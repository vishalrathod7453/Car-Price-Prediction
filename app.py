import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("ModelCR.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        color: #2E86C1;
        font-size: 40px;
        font-weight: bold;
    }
    .result {
        text-align: center;
        color: green;
        font-size: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<div class='title'>🚗 Car Price Prediction App</div>", unsafe_allow_html=True)

st.write("### Enter Car Details")

# ---------------- INPUTS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", min_value=2000, max_value=2025, value=2018)
    present_price = st.number_input("Present Price (Lakhs)", value=5.0)

with col2:
    kms_driven = st.number_input("Kilometers Driven", value=30000)
    owner = st.selectbox("Owner", [0, 1, 2, 3])

with col3:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# ---------------- FEATURE ENGINEERING ----------------
fuel = 1 if fuel_type == "Diesel" else 0
trans = 1 if transmission == "Manual" else 0
car_age = 2024 - year

# ---------------- PREDICT ----------------
if st.button("🔮 Predict Price"):
    
    input_data = np.array([[present_price, kms_driven, owner, fuel, trans, car_age]])
    prediction = model.predict(input_data)

    st.markdown(f"<div class='result'>💰 Predicted Price: ₹ {round(prediction[0],2)} Lakhs</div>", unsafe_allow_html=True)
    
    st.balloons()
