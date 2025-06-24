
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("flight_safety_model.pkl")
scaler = joblib.load("scaler.pkl")

# UI
st.title("✈️ AI Flight Safety Predictor")
st.markdown("Predict if a flight condition is **Safe** or **Risky** using AI.")

# Sidebar input
st.sidebar.header("Enter Flight Telemetry Data")

altitude = st.sidebar.number_input("Altitude (ft)", 20000, 45000, 35000)
speed = st.sidebar.number_input("Speed (knots)", 300, 700, 500)
pitch = st.sidebar.slider("Pitch Angle (°)", -15.0, 15.0, 2.0)
temp = st.sidebar.number_input("Engine Temp (°C)", 500, 800, 650)
pressure = st.sidebar.slider("Cabin Pressure (psi)", 8.0, 15.0, 10.5)
weather = st.sidebar.slider("Weather Risk (0–1)", 0.0, 1.0, 0.3)

# Prediction logic
columns = ['Altitude', 'Speed', 'PitchAngle', 'EngineTemp', 'CabinPressure', 'WeatherRisk']
input_df = pd.DataFrame([[altitude, speed, pitch, temp, pressure, weather]], columns=columns)
input_scaled = scaler.transform(input_df)

if st.button("🧠 Predict Safety Status"):
    prediction = model.predict(input_scaled)[0]
    result = "✅ SAFE" if prediction == 0 else "🚨 RISKY"
    st.subheader(f"Prediction: {result}")
    st.write("Flight Input Data:")
    st.dataframe(input_df)
