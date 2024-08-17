import streamlit as st
import joblib

# Load the model from the pickle file
model = joblib.load('random_forest_model.pkl')

# Function to make predictionsdefmake_prediction(input_data):
    prediction = model.predict([input_data])
    return prediction[0]

# Streamlit app UI
st.title("Sales Prediction App")

# Collect user inputs
input_data = []

# Example input fields for the features your model uses
feature_1 = st.number_input("Enter value for Feature 1")
feature_2 = st.number_input("Enter value for Feature 2")
# Add more input fields as needed for your features# Store the inputs in a list
input_data.extend([feature_1, feature_2])

# Make predictionsif st.button("Predict"):
    prediction = make_prediction(input_data)
    st.write(f"The predicted sales amount is: {prediction}")
