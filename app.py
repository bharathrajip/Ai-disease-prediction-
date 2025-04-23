
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('disease_prediction_model.pkl')

# Streamlit UI
st.title("AI Disease Prediction App")
st.write("Enter symptoms to predict possible disease")

# Symptoms input
symptoms = {
    "Fever": st.checkbox("Fever"),
    "Cough": st.checkbox("Cough"),
    "Headache": st.checkbox("Headache"),
    "Nausea": st.checkbox("Nausea"),
    "Fatigue": st.checkbox("Fatigue"),
    "Pain": st.checkbox("Pain"),
    "Vomiting": st.checkbox("Vomiting"),
    "Diarrhea": st.checkbox("Diarrhea"),
    "Sore_throat": st.checkbox("Sore_throat")
}

# Predict button
if st.button("Predict"):
    input_data = {k: int(v) for k, v in symptoms.items()}
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Disease: {prediction}")
