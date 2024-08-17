import streamlit as st
import joblib
import numpy as np

# Load the model from the pickle file
model = joblib.load('random_forest_model.pkl')

# Function to make predictionsdefmake_prediction(input_data):
prediction = model.predict([input_data])
    return prediction[0]

# Streamlit app UI
st.title("Previsão quantidade de vendas")

# Collect user inputs
Vlr_Liquido = st.number_input("Insira o valor líquido")
Quantidade_de_Acessos = st.number_input("Insira a quantidade de acessos")
qtd_campanha = st.number_input("Insira a quantidade de campanha")
qtd_treinamento = st.number_input("Insira a quantidade de treinamento")
qtd_feedback = st.number_input("Insira a quantidade de feedback")
# Add more input fields as needed# Make predictionif st.button("Predict"):

input_data = [Vlr_Liquido, Quantidade_de_Acessos, qtd_campanha, qtd_treinamento, qtd_feedback]  # Adjust this to match your model's input
    prediction = make_prediction(input_data)
    st.write(f"The predicted sales amount is: {prediction}"
