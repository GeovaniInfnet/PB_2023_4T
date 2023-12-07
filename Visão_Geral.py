
import streamlit as st
import pandas as pd
from PIL import Image

@st.cache_data
def upload_file():
    try:
        df = pd.read_csv('Credit_Card_Customers.csv')

        if {'Customer_Age', 'Gender', 'Dependent_count', 
            'Income_Category','Card_Category', 'Avg_Open_To_Buy',
            'Total_Trans_Amt', 'Total_Trans_Ct', 'Credit_Limit',
            'Avg_Utilization_Ratio'}.issubset(df.columns):
            return (True, df)
        else:
            st.error('O arquivo não possui o formato adequado!', icon="🚨")
            return (False, None)  
    except FileNotFoundError:        
        st.info("""
            Não foram encontrados os dados sobre a movimentação financeira dos clientes!\n
            Verifique se o arquivo 'Credit_Card_Customers.csv' está na pasta raiz.""", icon="ℹ️") 
        return (False, None)

st.set_page_config(layout='wide')
st.title('Visão Geral')
st.subheader('Projeto de Bloco - Geovani Balardin')

file_loaded, bank = upload_file()

if file_loaded:
    with st.expander('Workflow_Canvas'):
        img = Image.open('./Workflow_Canvas.png')
        st.image(img)

    with st.expander('Dicionário de Dados'):
        st.write("""
            - **Customer_Age**: Idade do cliente.
            - <b>Gender</b>: Gênero do cliente.
            - <b>Dependent_count</b>: Número de dependentes.
            - <b>Income_Category</b>: Renda anual.
            - <b>Card_Category</b>: Tipo de cartão.
            - <b>Avg_Open_To_Buy</b>: Valor total das linhas de crédito disponíveis para empréstimo e financiamento.
            - <b>Total_Trans_Amt</b>: Valor total das transações feitas no cartão (últimos 12 meses).
            - <b>Total_Trans_Ct</b>: Contagem total das transações feitas no cartão (últimos 12 meses).
            - <b>Credit_Limit</b>: Limite de crédito no cartão de crédito.
            - <b>Avg_Utilization_Ratio</b>: Índice médio de utilização do cartão.        
        """)
