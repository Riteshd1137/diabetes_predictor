# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import pickle

# Set page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")

# Title and description
st.title("🩺 Diabetes Prediction App")
st.markdown("""
This app uses a machine learning model to predict whether a patient is likely to have **Diabetes**.

Fill in the patient details below and click **Predict** to get the result.
""")

# Input fields
pregnancies = st.number_input("👶 Number of Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("🍬 Glucose Level", min_value=0)
blood_pressure = st.number_input("💓 Blood Pressure", min_value=0)
skin_thickness = st.number_input("📏 Skin Thickness", min_value=0)
insulin = st.number_input("💉 Insulin Level", min_value=0)
bmi = st.number_input("⚖️ Body Mass Index (BMI)", min_value=0.0, format="%.1f")
dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, format="%.2f")
age = st.number_input("🎂 Age", min_value=1, max_value=120, step=1)

# Load the trained model
try:
    with open("diabetes_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file not found. Please make sure 'diabetes_model.pkl' is in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Predict button
if st.button("🔍 Predict Diabetes"):
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                       'SkinThickness', 'Insulin', 'BMI',
                                       'DiabetesPedigreeFunction', 'Age'])

    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("⚠️ The patient is likely to have **Diabetes**.")
        else:
            st.success("✅ The patient is **not likely to have Diabetes**.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
