import streamlit as st
import pandas as pd

# Streamlit app UI
st.title("Previsão de vendas")

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

if vendsa is not None:
    df_vendas = pd.read_csv(vendas)
    st.write("Preview do arquivo de vendas:")
    st.dataframe(df_vendas)

# Proceed with any further data processing or model predictions

# Data Preparation
if 'df_vendas' in globals():
    vendas_cliente = df_vendas.groupby('cli_codigo')[['Vlr_Liquido', 'Qtd_Vendas', 'N_Produtos', 'Vlr_Desconto']].sum().reset_index()
    st.write("Summed Sales Data by Client Code:")

if 'df_acessos' in globals():
    acessos_cliente = df_acessos.groupby('CLI_CODIGO')['Quantidade_de_Acessos'].sum().reset_index()
    st.write("Summed Access Data by Client Code:")

if 'df_feedback' in globals():
    qtd_feedback = df_feedback.groupby('CLI_CODIGO')['Data'].count().reset_index()
    qtd_feedback.rename(columns={'CLI_CODIGO': 'cli_codigo', 'Data': 'qtd_feedback'}, inplace=True)
    st.write("Feedback Count by Client Code:")

if 'df_campanha' in globals():
    qtd_campanha = df_campanha.groupby('Cliente')['Campanha_Nome'].count().reset_index()
    qtd_campanha.rename(columns={'Cliente': 'cli_codigo', 'Campanha_Nome': 'qtd_campanha'}, inplace=True)
    st.write("Campaign Count by Client Code:")

if 'df_treinamento' in globals():
    qtd_treinamento = df_treinamento.groupby('Cliente').count().reset_index()
    qtd_treinamento = qtd_treinamento[['Cliente', 'Treinamento']]
    qtd_treinamento.rename(columns={'Cliente': 'cli_codigo', 'Treinamento': 'qtd_treinamento'}, inplace=True)
    st.write("Training Count by Client Code:")

# Merging DataFrames
if 'vendas_cliente' in globals() and 'acessos_cliente' in globals():
    # Merge with access data
    analise_vendas = vendas_cliente.merge(acessos_cliente, left_on='cli_codigo', right_on='CLI_CODIGO', how='left')
    analise_vendas.fillna(0, inplace=True)

if 'qtd_treinamento' in globals():
    # Merge with training data
    analise_vendas = analise_vendas.merge(qtd_treinamento, left_on='cli_codigo', right_on='cli_codigo', how='left')
    analise_vendas.fillna(0, inplace=True)

if 'qtd_campanha' in globals():
    # Merge with campaign data
    analise_vendas = analise_vendas.merge(qtd_campanha, left_on='cli_codigo', right_on='cli_codigo', how='left')
    analise_vendas.fillna(0, inplace=True)

if 'qtd_feedback' in globals():
    # Merge with feedback data
    analise_vendas = analise_vendas.merge(qtd_feedback, left_on='cli_codigo', right_on='cli_codigo', how='left')
    analise_vendas.fillna(0, inplace=True)

# Final merge for sales analysis
analise_vendas = analise_vendas[['cli_codigo', 'Vlr_Liquido', 'Qtd_Vendas', 'Quantidade_de_Acessos', 'qtd_treinamento', 'qtd_campanha', 'qtd_feedback', 'N_Produtos', 'Vlr_Desconto']]

# Display the final merged DataFrame
st.write("Final Merged Sales Data:")
st.dataframe(analise_vendas)

# You can then use `analise_vendas` for model prediction or further analysis

# Assuming 'analise_vendas' DataFrame is already available
if 'analise_vendas' in globals():
    # Apply Box-Cox transformation
    for column in ['Vlr_Liquido', 'Qtd_Vendas', 'Quantidade_de_Acessos', 'qtd_treinamento', 'qtd_campanha', 'qtd_feedback', 'N_Produtos', 'Vlr_Desconto']:
        if column in analise_vendas.columns:
            analise_vendas[column], lambda_ = stats.boxcox(analise_vendas[column] + 1)
            st.write(f"Box-Cox transformation applied to {column}")

    # Display the transformed DataFrame
    st.write("Transformed Sales Data:")
    st.dataframe(analise_vendas)
