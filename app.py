import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

# Load trained model
model =load_model('D:\\NLP_Tutorial\\ann_model.h5')

# Load encoders and scaler
with open(r"D:\NLP_Tutorial\label_enocder_gender.pkl",'rb') as file:
    label_encoder_gender=pickle.load(file)

with open(r"D:\NLP_Tutorial\onehot_enocder.pkl",'rb') as file:
    onehot_encoder=pickle.load(file) 

with open(r"D:\NLP_Tutorial\scaler.pkl",'rb') as file:
    scaler=pickle.load(file)    

# Streamlit app
st.title("Customer Churn Prediction")

# User inputs
geography = st.selectbox("Geography", onehot_encoder.categories_[0])  
gender = st.selectbox('Gender',label_encoder_gender.classes_) 
age=st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure], 
    'Balance': [balance],
    'NumOfProducts': [num_of_products], 
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})
geo_encoded = onehot_encoder.transform([[geography]])
geo_encoded_df= pd.DataFrame(geo_encoded,columns=onehot_encoder.get_feature_names_out())

# concat onehot encoded data with input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
   
# scale input data

input_data_scaled = scaler.transform(input_data)  

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability: {prediction_proba:.2f}")
if prediction_proba > 0.5:
    st.write( "The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")
