# Importação das bibliotecas necessárias
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

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


# Função que realiza a codificação de variáveis categóricas utilizando LabelEncoder
def encode_features(data):
    dict_encoders = {}

    # Guarda a referência do objeto (para desfazer a transformação) e realiza a codificação das variáveis
    dict_encoders['Gender'] = [LabelEncoder(), 'Gender_Encoded']
    data['Gender_Encoded'] = dict_encoders['Gender'][0].fit_transform(data['Gender'].values)

    income_order = [['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']]
    dict_encoders['Income_Category'] = [OrdinalEncoder(categories=income_order), 'IncomeCategory_Encoded']
    data['IncomeCategory_Encoded'] = dict_encoders['Income_Category'][0].fit_transform(data[['Income_Category']].values)

    card_order = [['Blue', 'Silver', 'Gold', 'Platinum']]
    dict_encoders['Card_Category'] = [OrdinalEncoder(categories=card_order), 'CardCategory_Encoded']
    data['CardCategory_Encoded'] = dict_encoders['Card_Category'][0].fit_transform(data[['Card_Category']].values)

    return data, dict_encoders 


# Função que realiza todo o pré-processamento dos dados
@st.cache_data
def pre_processing(data):
    data, dict_encoders = encode_features(data)

    # Cria um novo dataframe com as colunas categóricas codificadas
    bank_model = data[[
        'Customer_Age',
        'Gender_Encoded',
        'Dependent_count',
        'IncomeCategory_Encoded',
        'CardCategory_Encoded',
        'Avg_Open_To_Buy',
        'Total_Trans_Amt',
        'Total_Trans_Ct',    
        'Avg_Utilization_Ratio',
        'Credit_Limit'
    ]].copy()

    # Cria um array com as variáveis independentes
    X = bank_model.iloc[:, :9].values

    # Cria um array com a variável dependente
    y = bank_model.iloc[:, 9].values    
    return X, y, dict_encoders, bank_model


# Função para treinar o modelo de regressão linear
def model_training(X_train_transformed, y_train_transformed, X_test_transformed, y_test, pt_y):
    model = LinearRegression()
    model.fit(X_train_transformed, y_train_transformed)
    
    y_pred_transformed = model.predict(X_test_transformed)

    # Reverte as previsões para a escala original
    y_pred = pt_y.inverse_transform(y_pred_transformed)    
    y_pred = y_pred.reshape(-1)

    # Calcula métricas de desempenho usando os valores originais de y_test (sem transformação)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)

    return y_pred, mae, mape, mse, rmse


# Função para plotar o gráfico de regressão
def plot_regression(X_test_inverse, feature_idx, feature_name, encoded=False, dict_encoders=None):
    plt.figure(figsize=(10,4))

    # Seleciona somente os dados pedidos pelo plot
    X_test_plot = X_test_inverse[:, feature_idx]

    # Invertendo a codificação do LabelEncoder nos dados de teste
    if encoded:        
        if isinstance(dict_encoders[feature_name][0], LabelEncoder):
            X_test_plot = dict_encoders[feature_name][0].inverse_transform(X_test_plot.astype(int))
        else:
            X_test_plot = X_test_plot.reshape(-1, 1).astype(int)
            X_test_plot = dict_encoders[feature_name][0].inverse_transform(X_test_plot).ravel() 

    # Plota o gráfico de dispersão com os dados reais
    ax = sns.scatterplot(x=X_test_plot, y=y_test, label=feature_name)

    # Plota a linha de regressão
    sns.lineplot(x=X_test_plot, y=y_pred, color='red', label='Regressão Linear', errorbar=None, ax=ax)

    # Cria um título e altera o rótulo dos eixos
    ax.set_title('Regressão Linear')
    ax.set_xlabel(f'Variável Independente - {feature_name}')
    ax.set_ylabel('Variável Dependente - Credit_Limit')
 
    return ax.figure

# Configuração da página com layout amplo e título
st.set_page_config(layout='wide')
st.title('Resultados do Modelo')

# Carrega o arquivo e verifica se foi carregado com sucess  
file_loaded, bank = upload_file()

if file_loaded:
    bank_original = bank.copy()
    
    # Pré-processamento dos dados
    X, y, dict_encoders, bank_model = pre_processing(bank)

    # 70% serão usados para treino. 30% serão usados para teste.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7070)

    # Transformação de escala das variáveis independentes
    pt_X = PowerTransformer()
    X_train_transformed = X_train
    X_train_transformed[:, [5, 6, 7, 8]] = pt_X.fit_transform(X_train_transformed[:, [5, 6, 7, 8]])

    X_test_transformed = X_test
    X_test_transformed[:, [5, 6, 7, 8]] = pt_X.fit_transform(X_test_transformed[:, [5, 6, 7, 8]])

    # Transformação de escala da variáveis dependente
    pt_y = PowerTransformer()
    y_train_transformed = pt_y.fit_transform(y_train.reshape(-1, 1))
    
    # Treinamento e avaliação do modelo
    y_pred, mae, mape, mse, rmse = model_training(X_train_transformed, 
                                                  y_train_transformed, 
                                                  X_test_transformed, 
                                                  y_test, 
                                                  pt_y)

    # Seção expansíveis para métricas e visualização dos resultados    
    with st.expander('Métricas'):
        dict_metric = {
            'MAE (Mean Absolute Error)': mae,
            'MAPE (Mean Absolute Percentage Error)': mape,
            'MSE (Mean Squared Error)': mse,
            'RMSE (Root Mean Squared Error)': rmse
        }

        df_metric = pd.DataFrame(dict_metric.items(), columns=['Métrica', 'Valor'])
        st.data_editor(df_metric, disabled=True, hide_index=True, use_container_width=True)

    # Seção expansíveis para visualização dos resultados 
    with st.expander('Visualização dos Resultados'):
        # Lista de variáveis do banco de dados original
        var_list = bank_original.drop('Credit_Limit', axis=1).columns.to_list()
        features = st.multiselect('Selecione a variável:', options=var_list)
        categorical_list = bank.select_dtypes(exclude='number').columns.to_list()

        # Inverte a escala nos dados de teste antes da plotagem
        X_test_inverse = X_test_transformed
        X_test_inverse[:, [5, 6, 7, 8]] = pt_X.inverse_transform(X_test_inverse[:, [5, 6, 7, 8]])          
        
        if features:
            for feature in features:
                if feature in categorical_list:
                    # Plota gráfico de dispersão para variáveis categóricas codificadas
                    st.pyplot(plot_regression(X_test_inverse, 
                                              bank_model.columns.get_loc(dict_encoders[feature][1]), 
                                              feature_name=feature, 
                                              encoded=True, 
                                              dict_encoders=dict_encoders))
                else:
                    # Plota gráfico de dispersão para variáveis numéricas
                    st.pyplot(plot_regression(X_test_inverse,
                                              bank_model.columns.get_loc(feature), 
                                              feature_name=feature))
