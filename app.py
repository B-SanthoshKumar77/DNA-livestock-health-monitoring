import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Livestock Health Monitoring", page_icon="🐄", layout="centered")

# Load artifacts
@st.cache_resource
def load_artifacts():
    return joblib.load("livestock_health_model_artifacts.joblib")

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
le = artifacts["label_encoder"]
mlb = artifacts["mlb"]
symptom_feature_names = artifacts["symptom_feature_names"]
animal_columns = artifacts["animal_columns"]

st.title("🐄 DNA-based Livestock Health Monitoring System")
st.markdown("Predict livestock diseases based on age, temperature, and symptoms using trained ML models.")

# User Inputs
animal = st.selectbox("Select Animal", [a.replace("animal_", "") for a in animal_columns])
age = st.number_input("Age (in years)", min_value=0.0, max_value=30.0, step=0.1)
temperature = st.number_input("Body Temperature (°F)", min_value=90.0, max_value=110.0, step=0.1)

# Dynamic symptom input
st.markdown("### Select Observed Symptoms")
user_symptoms = st.multiselect("Choose symptoms", mlb.classes_)

# Predict Button
if st.button("🔍 Predict Disease"):
    # Prepare features
    sym_arr = mlb.transform([user_symptoms]) if hasattr(mlb, "classes_") else np.zeros((1, len(symptom_feature_names)))
    sym_df = pd.DataFrame(sym_arr, columns=symptom_feature_names)

    # Animal one-hot
    animal_row = {c: 0 for c in animal_columns}
    col_name = f"animal_{animal.lower()}"
    if col_name in animal_row:
        animal_row[col_name] = 1

    feat = pd.DataFrame([{'Age': age, 'Temperature': temperature, **animal_row}])
    Xnew = pd.concat([feat.reset_index(drop=True), sym_df.reset_index(drop=True)], axis=1)

    for c in model.feature_names_in_:
        if c not in Xnew.columns:
            Xnew[c] = 0
    Xnew = Xnew[model.feature_names_in_]

    Xnew[['Age', 'Temperature']] = scaler.transform(Xnew[['Age', 'Temperature']])
    pred_idx = model.predict(Xnew)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    probas = model.predict_proba(Xnew)[0]
    class_probs = dict(zip(le.classes_, probas))

    st.success(f"### 🧬 Predicted Disease: **{pred_label.upper()}**")
    st.write("**Confidence Scores:**")
    st.json(class_probs)
else:
    st.info("👆 Enter details and click 'Predict Disease' to see results.")
