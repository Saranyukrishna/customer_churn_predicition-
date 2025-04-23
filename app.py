import streamlit as st
import pandas as pd
import joblib

model = joblib.load('adaboost_decision_tree_model.pkl')
scaler = joblib.load('scaler.pkl')

gender_map = {'Male': 1, 'Female': 0}

st.title('🏦 Customer Churn Prediction')
st.write("""
This app predicts whether a customer will **exit** or **stay** with the bank based on the details you provide below.
""")

credit_score = st.number_input('Credit Score', min_value=0.0, value=0.0)
gender = st.selectbox('Gender', options=['Male', 'Female'])
age = st.number_input('Age', min_value=0.0, value=0.0)
tenure = st.number_input('Tenure (Years)', min_value=0.0, value=0.0)
balance = st.number_input('Account Balance', min_value=0.0, value=0.0)
num_of_products = st.number_input('Number of Products', min_value=0, max_value=4, value=0)
has_cr_card = st.selectbox('Has Credit Card?', options=['No', 'Yes'])
is_active_member = st.selectbox('Is Active Member?', options=['No', 'Yes'])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=0.0)

gender_encoded = gender_map[gender]
has_cr_card_val = 1 if has_cr_card == 'Yes' else 0
is_active_val = 1 if is_active_member == 'Yes' else 0

raw_input = pd.DataFrame({
    'CreditScore': [credit_score],
    'EstimatedSalary': [estimated_salary],
    'Balance': [balance],
    'Age': [age],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'Gender': [gender_encoded],
    'HasCrCard': [has_cr_card_val],
    'IsActiveMember': [is_active_val]
})

numerical_columns = ['CreditScore', 'EstimatedSalary', 'Balance', 'Age', 'Tenure', 'NumOfProducts']
raw_input[numerical_columns] = scaler.transform(raw_input[numerical_columns])

if st.button('Predict'):
    prediction = model.predict(raw_input)
    prediction_proba = model.predict_proba(raw_input)

    if prediction[0] == 0:
        st.success(f"✅ Result: Customer will NOT exit (Probability: {prediction_proba[0][1]:.2f})")
    else:
        st.warning(f"🚨 Result: Customer is likely to EXIT (Probability: {prediction_proba[0][0]:.2f})")
