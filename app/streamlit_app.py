# app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# Load model
model = joblib.load(r'C:\Users\abdul\Downloads\Customer Churn Prediction\model\churn_xgboost_model.pkl')

# Title
st.title("📉 Customer Churn Predictor")
st.write("Predict whether a customer will churn based on input features.")

# Sidebar inputs
st.sidebar.header("Customer Info")

# Define inputs (same features as used during training)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Has Partner?", ['Yes', 'No'])
dependents = st.sidebar.selectbox("Has Dependents?", ['Yes', 'No'])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ['Yes', 'No'])
internet = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
contract = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
paperless = st.sidebar.selectbox("Paperless Billing", ['Yes', 'No'])
payment = st.sidebar.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])
monthly = st.sidebar.slider("Monthly Charges", 0, 150, 70)
total = st.sidebar.slider("Total Charges", 0, 9000, 2000)

# Convert inputs to DataFrame
input_dict = {
    'tenure': tenure,
    'MonthlyCharges': monthly,
    'TotalCharges': total,
    'gender_Male': 1 if gender == 'Male' else 0,
    'SeniorCitizen': senior,
    'Partner_Yes': 1 if partner == 'Yes' else 0,
    'Dependents_Yes': 1 if dependents == 'Yes' else 0,
    'PhoneService_Yes': 1 if phone_service == 'Yes' else 0,
    'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
    'InternetService_No': 1 if internet == 'No' else 0,
    'Contract_One year': 1 if contract == 'One year' else 0,
    'Contract_Two year': 1 if contract == 'Two year' else 0,
    'PaperlessBilling_Yes': 1 if paperless == 'Yes' else 0,
    'PaymentMethod_Electronic check': 1 if payment == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if payment == 'Mailed check' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment == 'Credit card (automatic)' else 0,
}

input_df = pd.DataFrame([input_dict])

# Load list of training columns (used to train the model)
training_columns = model.get_booster().feature_names

# Reindex input_df to match training columns
input_df = input_df.reindex(columns=training_columns, fill_value=0)



# Predict
if st.button("Predict Churn"):
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    st.subheader("🔮 Prediction:")
    st.write("Churn" if pred == 1 else "No Churn")
    st.write(f"📊 Probability of Churn: {proba:.2%}")

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    st.subheader("📌 Top Features Driving Prediction")
    # Get top 3 features
    top_features = np.abs(shap_values[0]).argsort()[-3:][::-1]
    for i in top_features:
        st.write(f"🔹 {input_df.columns[i]} — Impact: {shap_values[0][i]:.4f}")
    

    # SHAP bar plot (clean matplotlib version for Streamlit)
    import shap
    import matplotlib.pyplot as plt
    
    st.subheader("📉 SHAP Feature Impact")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # Bar plot of top SHAP values
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], input_df.iloc[0], max_display=10, show=False)
    fig = plt.gcf()
    st.pyplot(fig)

