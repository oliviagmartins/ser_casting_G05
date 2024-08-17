import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats

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

        # Display columns for debugging
    st.write("Columns in analise_vendas after merges:", analise_vendas.columns)
        
        # Filter columns
    analise_vendas = analise_vendas[['cli_codigo', 'Vlr_Liquido', 'Qtd_Vendas', 'Quantidade_de_Acessos', 'qtd_treinamento', 'qtd_campanha', 'qtd_feedback', 'N_Produtos', 'Vlr_Desconto']]
    st.write("Filtered analise_vendas:", analise_vendas.head())

#    else:
#        st.error("qtd_campanha is not defined.")
#else:
#    st.error("Por favor, faça upload dos arquivos.")

# Final merge for sales analysis
#analise_vendas = analise_vendas[['cli_codigo', 'Vlr_Liquido', 'Qtd_Vendas', 'Quantidade_de_Acessos', 'qtd_treinamento', 'qtd_campanha', 'qtd_feedback', 'N_Produtos', 'Vlr_Desconto']]

# Display the final merged DataFrame
#st.write("Final Merged Sales Data:")
#st.dataframe(analise_vendas)

# Load the model from the pickle file
model = joblib.load('random_forest_model.pkl')

st.write("Model loaded successfully!")

# Check the initial data
#st.write("Before transformation:")
#st.write(analise_vendas[['Vlr_Liquido', 'Qtd_Vendas', 'Quantidade_de_Acessos', 'qtd_treinamento', 'qtd_campanha', 'qtd_feedback', 'N_Produtos', 'Vlr_Desconto']].head())

# Check for constant columns
#constant_columns = [col for col in ['Vlr_Liquido', 'Qtd_Vendas', 'Quantidade_de_Acessos', 'qtd_treinamento', 'qtd_campanha', 'qtd_feedback', 'N_Produtos', 'Vlr_Desconto'] if analise_vendas[col].nunique() == 1]

#if constant_columns:
#    st.write(f"Columns with constant values (Box-Cox transformation skipped): {constant_columns}")

# Apply Box-Cox transformation
#for col in ['Vlr_Liquido', 'Qtd_Vendas', 'Quantidade_de_Acessos', 'qtd_treinamento', 'qtd_campanha', 'qtd_feedback', 'N_Produtos', 'Vlr_Desconto']:
#    if col not in constant_columns:  # Apply only to non-constant columns
#        try:
#            analise_vendas[col], _ = stats.boxcox(analise_vendas[col] + 1)
#        except Exception as e:
#            st.error(f"Error during Box-Cox transformation for column {col}: {e}")

# Check the transformed data
#st.write("After transformation:")
#st.write(analise_vendas.head())


# Apply Box-Cox transformation
#try:
#    analise_vendas['Vlr_Liquido'], lambda_ = stats.boxcox(analise_vendas['Vlr_Liquido'] + 1)
#    analise_vendas['Qtd_Vendas'], lambda_ = stats.boxcox(analise_vendas['Qtd_Vendas'] + 1)
#    analise_vendas['Quantidade_de_Acessos'], lambda_ = stats.boxcox(analise_vendas['Quantidade_de_Acessos'] + 1)
#    analise_vendas['qtd_treinamento'], lambda_ = stats.boxcox(analise_vendas['qtd_treinamento'] + 1)
#    analise_vendas['qtd_campanha'], lambda_ = stats.boxcox(analise_vendas['qtd_campanha'] + 1)
#    analise_vendas['qtd_feedback'], lambda_ = stats.boxcox(analise_vendas['qtd_feedback'] + 1)
#    analise_vendas['N_Produtos'], lambda_ = stats.boxcox(analise_vendas['N_Produtos'] + 1)
#    analise_vendas['Vlr_Desconto'], lambda_ = stats.boxcox(analise_vendas['Vlr_Desconto'] + 1)
    
    # Check the transformed data
    #st.write("After transformation:")
    #st.write(analise_vendas[['Vlr_Liquido', 'Qtd_Vendas', 'Quantidade_de_Acessos', 'qtd_treinamento', 'qtd_campanha', 'qtd_feedback', 'N_Produtos', 'Vlr_Desconto']].head())
#except Exception as e:
 #   st.error(f"Error during transformation: {e}")

    # Apply Box-Cox transformations
    #for column in ['Vlr_Liquido', 'Qtd_Vendas', 'Quantidade_de_Acessos', 'qtd_treinamento', 'qtd_campanha', 'qtd_feedback', 'N_Produtos', 'Vlr_Desconto']:
        #if column in analise_vendas.columns:
            #analise_vendas[column], lambda_ = stats.boxcox(analise_vendas[column] + 1)
            #st.write(f"Box-Cox transformation applied to {column}")


if 'analise_vendas' in globals():
    st.write("analise_vendas is defined and ready to use.")
else:
    st.write("analise_vendas is not defined.")


    # Prepare the features for prediction
X = analise_vendas.drop(columns=['Qtd_Vendas', 'cli_codigo', 'N_Produtos', 'Vlr_Liquido'])

    # Make predictions
predictions = model.predict(X)

    # Display the predictions
st.write("Predictions:")
st.dataframe(pd.DataFrame(predictions, columns=['Predicted_Qtd_Vendas']))


# Display Feature Importances
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

st.write("Feature Importances:")
st.dataframe(feature_importance_df)
st.bar_chart(feature_importance_df.set_index('Feature')['Importance'])

    # Plot Predicted vs Actual Values
#st.write("Predicted vs Actual Values:")
#plt.figure(figsize=(8, 6))
#plt.scatter(y_test, y_pred, alpha=0.5)
#plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Line of perfect fit
#plt.xlabel('Actual Values')
#plt.ylabel('Predicted Values')
#plt.title('Predicted vs Actual Values')
    
    # Render plot in Streamlit
#st.pyplot(plt)
