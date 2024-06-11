# -*- coding: utf-8 -*-
"""
Created on Sat May 18 22:54:10 2024

@author: nandi
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open(r"C:\Users\nandi\Desktop\cp2\piper.pkl", 'rb'))
car = pickle.load(open(r"C:\Users\nandi\Desktop\cp2\car.pkl", 'rb'))
X_test = pickle.load(open(r"C:\Users\nandi\Desktop\cp2\X_test.pkl", 'rb'))
print(X_test)


# creating a function for Prediction
def Car_price_prediction(input_data):
    # create a DataFrame with the correct column names
    input_data_df =pd.DataFrame(columns=X_test.columns,data=np.array([input_data]).reshape(1,-1))
            
    # predict the price
    prediction = loaded_model.predict(input_data_df)
    
    # Print the raw prediction for debugging
    print("Raw prediction:", prediction[0])
    
    # exponential transformation if model outputs log(price)
    j = prediction[0]
    return round(j, 2)

def main():
    # giving a title
    st.title('Car Price Prediction')
    
    # getting the input data from the user
    name = st.selectbox('Name', car['name'].unique())
    company = st.selectbox('Company', car['company'].unique())
    year = st.slider('Year', min_value=1995, max_value=2019, step=1)
    kms_driven = st.number_input('Kms Driven')
    fuel_type = st.selectbox('Fuel Type', car['fuel_type'].unique())
    
    # code for Prediction
    Price = ''
    
    # creating a button for Prediction
    if st.button('Predict Price'):
        # encode categorical variables as necessary (example shown below)
        # you might need to encode these categories as your model was trained with
        Year=int(year)
        Price = Car_price_prediction([name,company,Year,kms_driven,fuel_type])
        st.write("The Final Car Price is:")        
        st.success(Price)
        print(Price)
if __name__ == '__main__':
    main()
