import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Load model and preprocessing objects
model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('onehot.pkl', 'rb') as f:
    onehot = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# App title
st.title("📊 Telco Customer Churn Prediction")

st.write("Enter customer details to predict the probability of churn.")

# User inputs
with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.number_input("SeniorCitizen (0 or 1)", min_value=0, max_value=1)
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0)
    phone_service = st.selectbox("PhoneService", ["Yes", "No"])
    multiple_lines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])
    payment_method = st.selectbox("PaymentMethod", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("MonthlyCharges", min_value=0.0)
    total_charges = st.number_input("TotalCharges", min_value=0.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Create dataframe from inputs
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    df_input = pd.DataFrame([input_dict])

    # Binary label encoding using stored label encoders
    label_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'PaperlessBilling']
    for col in label_cols:
        mapping = {cls: idx for idx, cls in enumerate(label_encoders[col].classes_)}
        df_input[col] = df_input[col].map(mapping).fillna(0).astype(int)

    # One-hot encoding with safe handling of unknown categories
    onehot_cols = ['InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                   'TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']
    onehot_df = pd.DataFrame(
        onehot.transform(df_input[onehot_cols]),
        columns=onehot.get_feature_names_out(onehot_cols),
        index=df_input.index
    )

    # Combine encoded features
    df_input = df_input.drop(columns=onehot_cols)
    df_input = pd.concat([df_input, onehot_df], axis=1)

    # Ensure same column order as training
    X_columns = scaler.feature_names_in_
    for col in X_columns:
        if col not in df_input.columns:
            df_input[col] = 0  # Add missing columns as 0
    df_input = df_input[X_columns]

    # Scale
    df_scaled = scaler.transform(df_input)

    # Predict
    prob = model.predict(df_scaled)[0][0]
    churn_pred = "Churn" if prob >= 0.5 else "No Churn"

    # Display result
    if churn_pred == "Churn":
        st.error(f"⚠️ Customer likely to churn! Probability: **{prob:.2%}**")
    else:
        st.success(f"✅ Customer likely to stay. Probability: **{prob:.2%}**")
