import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

st.markdown(
    """
    <style>
    .logos {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .logo-container {
        margin-bottom: 30px; /* Adjust spacing as needed */
    }
    </style>
    <div class="logos">
        <div class="logo-container">
            <img src="https://raw.githubusercontent.com/oliviagmartins/ser_casting_G05/main/logo_ser.png" width="300" alt="Company Logo">
        </div>
        <div class="logo-container">
            <img src="https://raw.githubusercontent.com/oliviagmartins/ser_casting_G05/main/logo_dnc.png" width="200" alt="School Logo">
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Previsão de Vendas")

# URLs of the pickle files in your GitHub repository
pickle_url_amount = 'https://raw.githubusercontent.com/oliviagmartins/ser_casting_G05/main/rf_quantidade.pkl'
pickle_url_value = 'https://raw.githubusercontent.com/oliviagmartins/ser_casting_G05/main/rf_valor.pkl'

# Download and load both pickle files
def load_model(pickle_url):
    response = requests.get(pickle_url)
    if response.status_code == 200:
        pickle_file = io.BytesIO(response.content)
        model = joblib.load(pickle_file)
        return model
    else:
        st.write("Falha em carregar o modelo.")
        return None

model_amount = load_model(pickle_url_amount)
model_value = load_model(pickle_url_value)

if model_amount and model_value:
    st.write("Modelos carregados com sucesso!")

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
    df_vendas['Qtd_Vendas'] = df_vendas.groupby('cli_codigo')['cli_codigo'].transform('count')

    # Data Preparation
    vendas_cliente = df_vendas.groupby('cli_codigo')[['Vlr_Liquido', 'Qtd_Vendas', 'N_Produtos', 'Vlr_Desconto']].sum().reset_index()
    #st.write("Summed Sales Data by Client Code:")
    #st.dataframe(vendas_cliente)

if 'df_acessos' in globals():
    acessos_cliente = df_acessos.groupby('CLI_CODIGO')['Quantidade_de_Acessos'].sum().reset_index()
    #st.write("Summed Access Data by Client Code:")
    #st.dataframe(acessos_cliente)

if 'df_feedback' in globals():
    qtd_feedback = df_feedback.groupby('CLI_CODIGO')['Data'].count().reset_index()
    qtd_feedback.rename(columns={'CLI_CODIGO': 'cli_codigo', 'Data': 'qtd_feedback'}, inplace=True)
    #st.write("Feedback Count by Client Code:")
    #st.dataframe(qtd_feedback)

if 'df_campanha' in globals():
    qtd_campanha = df_campanha.groupby('Cliente')['Campanha_Nome'].count().reset_index()
    qtd_campanha.rename(columns={'Cliente': 'cli_codigo', 'Campanha_Nome': 'qtd_campanha'}, inplace=True)
    #st.write("Campaign Count by Client Code:")
    #st.dataframe(qtd_campanha)

if 'df_treinamento' in globals():
    qtd_treinamento = df_treinamento.groupby('Cliente').count().reset_index()
    qtd_treinamento = qtd_treinamento[['Cliente', 'Treinamento']]
    qtd_treinamento.rename(columns={'Cliente': 'cli_codigo', 'Treinamento': 'qtd_treinamento'}, inplace=True)
    #st.write("Training Count by Client Code:")
    #st.dataframe(qtd_treinamento)

# Check if all necessary DataFrames are available
required_dfs = ['vendas_cliente', 'acessos_cliente', 'qtd_feedback', 'qtd_campanha', 'qtd_treinamento']
if all(df_name in globals() for df_name in required_dfs):
    # Merges
    analise_vendas = vendas_cliente.merge(acessos_cliente, left_on='cli_codigo', right_on='CLI_CODIGO', how='left')
    analise_vendas.fillna(0, inplace=True)
    analise_vendas = analise_vendas.merge(qtd_treinamento, left_on='cli_codigo', right_on='cli_codigo', how='left')
    analise_vendas.fillna(0, inplace=True)
    analise_vendas = analise_vendas.merge(qtd_campanha, left_on='cli_codigo', right_on='cli_codigo', how='left')
    analise_vendas.fillna(0, inplace=True)
    analise_vendas = analise_vendas.merge(qtd_feedback, left_on='cli_codigo', right_on='cli_codigo', how='left')
    analise_vendas.fillna(0, inplace=True)
    
    # Filter columns
    analise_vendas = analise_vendas[['cli_codigo', 'Vlr_Liquido', 'Qtd_Vendas', 'Quantidade_de_Acessos', 'qtd_treinamento', 'qtd_campanha', 'qtd_feedback', 'N_Produtos', 'Vlr_Desconto']]
    st.write("Dados agrupados do cliente:", analise_vendas.head())

    # Make predictions
    X = analise_vendas[['Quantidade_de_Acessos', 'qtd_treinamento', 'qtd_campanha', 'qtd_feedback', 'N_Produtos', 'Vlr_Desconto']]  # Select features used in training

    # Predict with both models
    if model_amount:
        predictions_amount = model_amount.predict(X)
        st.write("Predição da quantidade de vendas:", predictions_amount)

    if model_value:
        predictions_value = model_value.predict(X)
        st.write("Predição do valor líquido:", predictions_value)

else:
    st.write("Data is not fully loaded or prepared yet.")


#### ATÉ AQUI FUNCIONA ######
#st.write(X)
#st.write(analise_vendas.head())

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
