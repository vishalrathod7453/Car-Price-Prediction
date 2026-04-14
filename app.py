import streamlit as st
import pickle
import pandas as pd

# ---------------- LOAD FILES ----------------
model = pickle.load(open("ModelCR.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

st.title("🚗 Car Price Prediction App")

# ---------------- INPUTS ----------------
year = st.number_input("Year", 2000, 2025, 2019)
present_price = st.number_input("Present Price (Lakhs)", 0.0, 50.0, 5.0)
kms_driven = st.number_input("Kilometers Driven", 0, 200000, 30000)
owner = st.selectbox("Owner", [0, 1, 2, 3])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# ---------------- FEATURE ENGINEERING ----------------
car_age = 2024 - year

# ---------------- CREATE INPUT DICTIONARY ----------------
input_dict = {
    'Present_Price': present_price,
    'Kms_Driven': kms_driven,
    'Owner': owner,
    'Fuel_Type_Diesel': 1 if fuel_type == "Diesel" else 0,
    'Seller_Type_Individual': 1 if seller_type == "Individual" else 0,
    'Transmission_Manual': 1 if transmission == "Manual" else 0,
    'Car_Age': car_age
}

# ---------------- CONVERT TO DATAFRAME ----------------
input_df = pd.DataFrame([input_dict])

# Ensure correct column order
input_df = input_df.reindex(columns=columns, fill_value=0)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Price: ₹ {round(prediction[0], 2)} Lakhs")
    st.balloons()
