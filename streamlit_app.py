import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats

st.title("Previsão de vendas")


# Load the model
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)
    
# File uploaders for each CSV file
acessos = st.file_uploader("Upload do arquivo de acessos", type=["csv"], key="acessos")
campanha = st.file_uploader("Upload do arquivo de campanha", type=["csv"], key="campanha")
feedback = st.file_uploader("Upload do arquivo de feedback", type=["csv"], key="feedback")
treinamento = st.file_uploader("Upload do arquivo de treinamento", type=["csv"], key="treinamento")
vendas = st.file_uploader("Upload do arquivo de vendas", type=["csv"], key="vendas")

# Read the files into DataFrames
if acessos is not None:
    df_acessos = pd.read_csv(acessos)
    st.write("Preview do arquivo de acessos:")
    st.dataframe(df_acessos)

if campanha is not None:
    df_campanha = pd.read_csv(campanha)
    st.write("Preview do arquivo de campanha:")
    st.dataframe(df_campanha)

if feedback is not None:
    df_feedback = pd.read_csv(feedback)
    st.write("Preview do arquivo de feedback:")
    st.dataframe(df_feedback)

if treinamento is not None:
    df_treinamento = pd.read_csv(treinamento)
    st.write("Preview do arquivo de treinamento:")
    st.dataframe(df_treinamento)

if vendas is not None:
    df_vendas = pd.read_csv(vendas)
    st.write("Preview do arquivo de vendas:")
    st.dataframe(df_vendas)

if vendas is not None:
    df_vendas['Qtd_Vendas'] = df_vendas.groupby('cli_codigo')['cli_codigo'].transform('count')

# Data Preparation
if 'df_vendas' in globals():
    vendas_cliente = df_vendas.groupby('cli_codigo')[['Vlr_Liquido', 'Qtd_Vendas', 'N_Produtos', 'Vlr_Desconto']].sum().reset_index()
    st.write("Summed Sales Data by Client Code:")
    st.dataframe(vendas_cliente)

if 'df_acessos' in globals():
    acessos_cliente = df_acessos.groupby('CLI_CODIGO')['Quantidade_de_Acessos'].sum().reset_index()
    st.write("Summed Access Data by Client Code:")
    st.dataframe(acessos_cliente)

if 'df_feedback' in globals():
    qtd_feedback = df_feedback.groupby('CLI_CODIGO')['Data'].count().reset_index()
    qtd_feedback.rename(columns={'CLI_CODIGO': 'cli_codigo', 'Data': 'qtd_feedback'}, inplace=True)
    st.write("Feedback Count by Client Code:")
    st.dataframe(qtd_feedback)

if 'df_campanha' in globals():
    qtd_campanha = df_campanha.groupby('Cliente')['Campanha_Nome'].count().reset_index()
    qtd_campanha.rename(columns={'Cliente': 'cli_codigo', 'Campanha_Nome': 'qtd_campanha'}, inplace=True)
    st.write("Campaign Count by Client Code:")
    st.dataframe(qtd_campanha)

if 'df_treinamento' in globals():
    qtd_treinamento = df_treinamento.groupby('Cliente').count().reset_index()
    qtd_treinamento = qtd_treinamento[['Cliente', 'Treinamento']]
    qtd_treinamento.rename(columns={'Cliente': 'cli_codigo', 'Treinamento': 'qtd_treinamento'}, inplace=True)
    st.write("Training Count by Client Code:")
    st.dataframe(qtd_treinamento)

# Merges
    if 'qtd_campanha' in globals():
        analise_vendas = vendas_cliente.merge(acessos_cliente, left_on='cli_codigo', right_on='CLI_CODIGO', how='left')
        analise_vendas.fillna(0, inplace=True)
        analise_vendas = analise_vendas.merge(qtd_treinamento, left_on='cli_codigo', right_on='cli_codigo', how='left')
        analise_vendas.fillna(0, inplace=True)
        analise_vendas = analise_vendas.merge(qtd_campanha, left_on='cli_codigo', right_on='cli_codigo', how='left')
        analise_vendas.fillna(0, inplace=True)
        analise_vendas = analise_vendas.merge(qtd_feedback, left_on='cli_codigo', right_on='cli_codigo', how='left')
        analise_vendas.fillna(0, inplace=True)
        
        # Filter columns
#     analise_vendas = analise_vendas[['cli_codigo', 'Vlr_Liquido', 'Qtd_Vendas', 'Quantidade_de_Acessos', 'qtd_treinamento', 'qtd_campanha', 'qtd_feedback', 'N_Produtos', 'Vlr_Desconto']]
#     st.write("Filtered analise_vendas:", analise_vendas.head())

if 'analise_vendas' in globals():
    st.write("'analise_vendas' is a global variable.")
else:
    st.write("'analise_vendas' is not a global variable.")

    # Prepare the features for prediction

if analise_vendas is not None:
    # Make predictions
    X = analise_vendas[['Quantidade_de_Acessos', 'qtd_treinamento', 'qtd_campanha', 'qtd_feedback', 'N_Produtos', 'Vlr_Desconto']]  # Select features used in training
    predictions = model.predict(X)
    st.write("Predictions:", predictions)

#model = joblib.load('random_forest_model.pkl')

#st.write("Model loaded successfully!")

#def predict(analise_vendas, model):
#    X = analise_vendas[['Quantidade_de_Acessos', 'qtd_treinamento', 'Vlr_Desconto', 'qtd_feedback', 'N_Produtos', 'qtd_campanha']]  # Define your X variables here
#y_pred = model.predict(X)  # Use these variables to make predictions
#    return y_pred

#prediction = predict(analise_vendas, model)

# Load your trained model (assuming you have it saved as a pickle file)
#@st.cache_resource
#def load_model():
#    with open('random_forest_model.pkl', 'rb') as file:
#        model = pickle.load(file)
#    return model

#model = load_model()

# Define the prediction function
#def predict(analise_vendas, model):
    # Define the features (X variables)
#    X = analise_vendas[['Quantidade_de_Acessos', 'qtd_treinamento', 'Vlr_Desconto', 'qtd_feedback', 'N_Produtos', 'qtd_campanha']]
    # Make predictions
#    y_pred = model.predict(X)
#    return y_pred

#st.write('teste')

#st.write('teste 2')
    # Make predictions
#prediction = predict(analise_vendas, model)

    # Add predictions to the dataframe
#analise_vendas['Predicted Sales'] = prediction

    # Display the dataframe with predictions
#st.write("Predicted vs Actual Sales:")
 #   st.write(analise_vendas[['Quantidade_de_Acessos', 'qtd_treinamento', 'Vlr_Desconto', 'qtd_feedback', 'N_Produtos', 'qtd_campanha', 'Predicted Sales']])


#st.write("Columns in 'analise_vendas':", analise_vendas.columns)

#st.write(analise_vendas.head())

#X = analise_vendas.drop(columns=['cli_codigo', 'Vlr_Liquido', 'Qtd_Vendas', 'N_Produtos'])

    # Make predictions
#predictions = model.predict(X)

    # Display the predictions
#st.write("Predictions:")
#st.dataframe(pd.DataFrame(predictions, columns=['Predicted_Qtd_Vendas']))


# Display Feature Importances
#importances = model.feature_importances_
#feature_names = X.columns
#feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
#feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

#st.write("Feature Importances:")
#st.dataframe(feature_importance_df)
#st.bar_chart(feature_importance_df.set_index('Feature')['Importance'])
