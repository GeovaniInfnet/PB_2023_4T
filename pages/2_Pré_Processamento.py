# Importação das bibliotecas necessárias
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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
        
# Função para adicionar features com transformação logarítmica
@st.cache_data
def add_log_features(data):
    # Aplica log em algumas variáveis específicas
    data['Avg_Open_To_Buy_Log'] = data['Avg_Open_To_Buy'].apply(lambda s: np.log(s))
    data['Total_Trans_Amt_Log'] = data['Total_Trans_Amt'].apply(lambda s: np.log(s))
    data['Total_Trans_Ct_Log'] = data['Total_Trans_Ct'].apply(lambda s: np.log(s))

    # Realiza a transformação 'yeo-johnson' na variável 'Avg_Utilization_Ratio'    
    data_transformed, _ = stats.yeojohnson(data['Avg_Utilization_Ratio'])
    data['Avg_Utilization_Ratio_Log'] = data_transformed    
    return data

# Função para adicionar features com codificação de variáveis categóricas
@st.cache_data
def add_encoded_features(data):
    # Utiliza LabelEncoder para codificar variáveis categóricas    
    label_encoder = LabelEncoder()

    # Realiza a codificação das variáveis
    data['Gender_Encoded'] = label_encoder.fit_transform(data['Gender'].values)
    data['IncomeCategory_Encoded'] = label_encoder.fit_transform(data['Income_Category'].values)
    data['CardCategory_Encoded'] = label_encoder.fit_transform(data['Card_Category'].values)
    return data

# Função para plotar distribuições antes e depois das transformações logarítmicas
def plot_numeric_features(data, x):    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    x_list = []
    x_list.append(x)
    x_list.append(x + '_Log')

    for col in range(2):
        # Criação do histograma com KDE
        sns.histplot(data=data, x=x_list[col], kde=True, ax=ax[col])
        ax[col].set_ylabel('Frequência')
        ax[col].set_xlabel(x_list[col])
        ax[col].lines[0].set_color('black')

        if col == 0:
            ax[col].set_title('Antes')
        else:
            ax[col].set_title('Depois')

        # Inclui as linhas verticais com valores da média e mediana
        ax[col].axvline(data[x_list[col]].mean(), label='Média', color='red')
        ax[col].axvline(data[x_list[col]].median(), label='Mediana', color='green')
        ax[col].legend()

    plt.tight_layout()
    return fig

# Função para aplicar StandardScaler em variáveis numéricas
def apply_standard_scaler(data):
    # Cria uma instância do objeto de escalonamento
    ss = StandardScaler()

    # Cria uma cópia da base original
    df_ss = data.copy()
    df_ss.drop([
        'Credit_Limit', 'Gender', 'Income_Category', 'Card_Category',
        'Avg_Open_To_Buy', 'Total_Trans_Amt', 'Total_Trans_Ct',
        'Avg_Utilization_Ratio'
    ], axis=1, inplace=True)

    # Coloca todas as variáveis numéricas na mesma escala
    df_ss.loc[:, df_ss.columns] = ss.fit_transform(df_ss)
    return df_ss

# Configuração da página com layout amplo e título
st.set_page_config(layout='wide')
st.title('Pré-Processamento')

# Carrega o arquivo e verifica se foi carregado com sucess    
file_loaded, bank = upload_file()

if file_loaded:
    # Seção expansível para Transformação Logarítmica
    with st.expander('Transformação Logarítmica'):
        bank = add_log_features(bank)

        features_list = ['Avg_Open_To_Buy', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Avg_Utilization_Ratio']
        features = st.multiselect('Variáveis que sofreram a transformação', options=features_list)

        if features:
            for feature in features:
                # Mostra gráficos de distribuição antes e depois das transformações
                st.pyplot(plot_numeric_features(bank, feature))

    # Seção expansível para Codificação de Dados Categóricos
    with st.expander('Codificação de Dados Categóricos'):
        bank = add_encoded_features(bank)

        # Exibe tabelas mostrando as relações entre as categorias originais e as codificadas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('#### Gender')
            df_gender = bank[['Gender', 'Gender_Encoded']].copy()
            df_gender.rename(columns={'Gender_Encoded': 'Encoded'}, inplace=True)
            df_gender.drop_duplicates(inplace=True)
            
            st.data_editor(df_gender, disabled=True, hide_index=True, use_container_width=True)

        with col2:
            st.write('#### Income_Category')
            df_income = bank[['Income_Category', 'IncomeCategory_Encoded']].copy()
            df_income.rename(columns={'IncomeCategory_Encoded': 'Encoded'}, inplace=True)
            df_income.drop_duplicates(inplace=True)

            st.data_editor(df_income, disabled=True, hide_index=True, use_container_width=True)

        with col3:
            st.write('#### Card_Category')
            df_card = bank[['Card_Category', 'CardCategory_Encoded']].copy()
            df_card.rename(columns={'CardCategory_Encoded': 'Encoded'}, inplace=True)
            df_card.drop_duplicates(inplace=True)

            st.data_editor(df_card, disabled=True, hide_index=True, use_container_width=True)

    # Seção expansível para Escalonamento
    with st.expander('Escalonamento'):
        bank = add_log_features(bank)
        bank = add_encoded_features(bank)
        bank_scaler = apply_standard_scaler(bank)

        # Mostra estatísticas descritivas antes do escalonamento
        st.write('#### Antes')
        st.data_editor(bank.describe(), disabled=True, use_container_width=True)

        # Mostra estatísticas descritivas depois do escalonamento
        st.write('#### Depois')
        st.data_editor(bank_scaler.describe(), disabled=True, use_container_width=True)
