import streamlit as st
import pandas as pd

# Streamlit app UI
st.title("Sales Prediction App")

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
