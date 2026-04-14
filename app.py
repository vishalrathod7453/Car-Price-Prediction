import streamlit as st
import pickle
import numpy as np
import requests
from streamlit_lottie import st_lottie

# 1. Page Configuration
st.set_page_config(page_title="Car Value Predictor", page_icon="🚗", layout="wide")

# 2. Load Assets (Animation)
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_car = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_96bovp06.json")

# 3. Load the Model
with open('ModelCR.pkl', 'rb') as file:
    model = pickle.load(file)

# 4. Header Section
st.title("🚗 Smart Car Price Estimator")
st.write("Enter the vehicle details below to get an instant valuation powered by Machine Learning.")

col_anim, col_text = st.columns([1, 2])
with col_anim:
    st_lottie(lottie_car, height=200, key="car_anim")
with col_text:
    st.info("""
    **Model Details:**
    - Type: Linear Regression
    - Features: 9 Vehicle Parameters
    - Version: Scikit-Learn 1.6.1
    """)

st.markdown("---")

# 5. User Input Form
with st.container():
    st.subheader("📋 Vehicle Specifications")
    
    # Organizing 9 features into 3 columns
    c1, c2, c3 = st.columns(3)
    
    with c1:
        car_id = st.number_input("Car ID", min_value=1, value=1001)
        brand = st.number_input("Brand (Encoded)", min_value=0, value=1)
        year = st.number_input("Year", min_value=1990, max_value=2026, value=2020)
        
    with c2:
        engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
        fuel_type = st.number_input("Fuel Type (Encoded)", min_value=0, value=1)
        transmission = st.number_input("Transmission (Encoded)", min_value=0, value=1)
        
    with c3:
        mileage = st.number_input("Mileage (km)", min_value=0, value=50000, step=1000)
        condition = st.number_input("Condition (Encoded)", min_value=0, value=1)
        model_type = st.number_input("Model (Encoded)", min_value=0, value=1)

# 6. Prediction Logic
st.markdown("###")
if st.button("🚀 Calculate Estimated Value", use_container_width=True):
    # Prepare features in the exact order found in ModelCR.pkl 
    features = np.array([[car_id, brand, year, engine_size, fuel_type, 
                          transmission, mileage, condition, model_type]])
    
    prediction = model.predict(features)
    
    st.markdown("---")
    st.balloons()
    st.success(f"## 💰 Estimated Market Value: ${prediction[0]:,.2f}")
