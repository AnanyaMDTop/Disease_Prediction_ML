import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open("model.pkl", "rb") as file:
    model_data = pickle.load(file)

rf_model = model_data["rf_model"]
svm_model = model_data["svm_model"]
nb_model = model_data["nb_model"]
voting_model = model_data["voting_model"]
encoder = model_data["encoder"]
scaler = model_data["scaler"]
symptom_index = model_data["symptom_index"]

st.set_page_config(page_title="Disease Prediction App", layout="centered")

st.title("🩺 Disease Prediction System")
st.write("Enter the symptoms you’re experiencing to predict possible diseases.")

# Convert input symptoms to model format
symptoms_list = list(symptom_index.keys())
selected_symptoms = st.multiselect("Select your symptoms", symptoms_list)

if st.button("Predict Disease"):
    input_data = np.zeros(len(symptoms_list))
    for symptom in selected_symptoms:
        index = symptom_index[symptom]
        input_data[index] = 1

    scaled_input = scaler.transform([input_data])
    prediction = voting_model.predict(scaled_input)[0]
    predicted_disease = encoder.inverse_transform([prediction])[0]

    st.success(f"🧠 Predicted Disease: **{predicted_disease}**")

st.markdown("---")
st.caption("Developed by Ananya M D & Ananya G Devadiga — Sahyadri College of Engineering & Management, Mangaluru")
