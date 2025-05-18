import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import dill

model = tf.keras.models.load_model('model.keras')
with open('preprocessor.dill', 'rb') as file:
    preprocessor = dill.load(file)
# preprocessor = joblib.load('preprocessor.joblib')

geography_categories = preprocessor.named_steps['column_transformer'].named_transformers_[
    'ohe'].categories_[0]
gender_categories = preprocessor.named_steps['column_transformer'].named_transformers_[
    'oe'].categories_[0]

# Streamlit
st.title("Customer Churn Prediction")
geography = st.selectbox('Geography', geography_categories)
gender = st.selectbox('Gender', gender_categories)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Create the dataframe
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Preprocess the dataframe
preprocessed_data = preprocessor.transform(input_data)

# Predict Churn
prediction = model.predict(preprocessed_data)  # type: ignore
pred_proba = prediction[0][0]

st.write(f'Churn Probability: {pred_proba:.2f}')

if pred_proba > 0.5:
    st.write("Customer is likely to Churn")
else:
    st.write("Customer is not likely to Churn")
