# Importação das bibliotecas necessárias
import streamlit as st
import pandas as pd
from PIL import Image

# Carrega o arquivo CSV e valida os dados
@st.cache_data
def upload_file():
    try:
        # Leitura do arquivo CSV
        df = pd.read_csv('Credit_Card_Customers.csv')
        
        # Verifica se o arquivo possui as colunas necessárias
        if {'Customer_Age', 'Gender', 'Dependent_count', 
            'Income_Category','Card_Category', 'Avg_Open_To_Buy',
            'Total_Trans_Amt', 'Total_Trans_Ct', 'Credit_Limit',
            'Avg_Utilization_Ratio'}.issubset(df.columns):
            return (True, df)
        else:
            # Se o arquivo não possuir as colunas adequadas, exibe um erro            
            st.error('O arquivo não possui o formato adequado!', icon="🚨")
            return (False, None)  
    except FileNotFoundError:   
        # Se o arquivo não for encontrado, exibe uma mensagem informativa        
        st.info("""
            Não foram encontrados os dados sobre a movimentação financeira dos clientes!\n
            Verifique se o arquivo 'Credit_Card_Customers.csv' está na pasta raiz.""", icon="ℹ️") 
        return (False, None)

# Configuração da página com layout amplo, título e subtítulo
st.set_page_config(layout='wide')
st.title('Visão Geral')
st.subheader('Projeto de Bloco - Geovani Balardin')

# Carrega o arquivo e verifica se foi carregado com sucesso
file_loaded, bank = upload_file()

if file_loaded:
    # Exibe uma seção expansível com uma imagem do workflow    
    with st.expander('Workflow_Canvas'):
        img = Image.open('./Workflow_Canvas.png')
        st.image(img)
        
    # Exibe uma seção expansível com um dicionário de dados explicando cada coluna
    with st.expander('Dicionário de Dados'):
        st.write("""
            - **Customer_Age**: Idade do cliente.
            - **Gender**: Gênero do cliente.
            - **Dependent_count**: Número de dependentes.
            - **Income_Category**: Renda anual.
            - **Card_Category**: Tipo de cartão.
            - **Avg_Open_To_Buy**: Valor total das linhas de crédito disponíveis para empréstimo e financiamento.
            - **Total_Trans_Amt**: Valor total das transações feitas no cartão (últimos 12 meses).
            - **Total_Trans_Ct**: Contagem total das transações feitas no cartão (últimos 12 meses).
            - **Credit_Limit**: Limite de crédito no cartão de crédito.
            - **Avg_Utilization_Ratio**: Índice médio de utilização do cartão.        
        """)
