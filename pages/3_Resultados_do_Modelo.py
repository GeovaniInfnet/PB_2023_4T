import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


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

def transform_log_features(data):
    data['Avg_Open_To_Buy'] = data['Avg_Open_To_Buy'].apply(lambda s: np.log(s))
    data['Total_Trans_Amt'] = data['Total_Trans_Amt'].apply(lambda s: np.log(s))
    data['Total_Trans_Ct'] = data['Total_Trans_Ct'].apply(lambda s: np.log(s))

    data_transformed, _ = stats.yeojohnson(data['Avg_Utilization_Ratio'])
    data['Avg_Utilization_Ratio'] = data_transformed    
    return data

def encode_features(data):
    dict_encoders = {}

    # Guarda a referência do objeto (para desfazer a transformação) e realiza a codificação das variáveis
    dict_encoders['Gender'] = [LabelEncoder(), 'Gender_Encoded']
    data['Gender_Encoded'] = dict_encoders['Gender'][0].fit_transform(data['Gender'].values)

    dict_encoders['Income_Category'] = [LabelEncoder(), 'IncomeCategory_Encoded']
    data['IncomeCategory_Encoded'] = dict_encoders['Income_Category'][0].fit_transform(data['Income_Category'].values)

    dict_encoders['Card_Category'] = [LabelEncoder(), 'CardCategory_Encoded']
    data['CardCategory_Encoded'] = dict_encoders['Card_Category'][0].fit_transform(data['Card_Category'].values)
    
    return data, dict_encoders

@st.cache_data
def pre_processing(data):
    data = transform_log_features(data)
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

@st.cache_data
def model_training(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)

    return y_pred, mae, mape, mse, rmse

def plot_regression(feature_idx, feature_name, encoded=False, dict_encoders=None):
    plt.figure(figsize=(10,4))

    # Seleciona somente os dados pedidos pelo plot
    X_test_plot = X_test_inverse[:, feature_idx]

    # Invertendo a codificação do LabelEncoder nos dados de teste
    if encoded:
        X_test_plot = dict_encoders[feature_name][0].inverse_transform(X_test_plot.astype(int))

    # Plota o gráfico de dispersão com os dados reais
    ax = sns.scatterplot(x=X_test_plot, y=y_test, label=feature_name)

    # Plota a linha de regressão
    sns.lineplot(x=X_test_plot, y=y_pred, color='red', label='Regressão Linear', errorbar=None, ax=ax)

    # Cria um título e altera o rótulo dos eixos
    ax.set_title('Regressão Linear')
    ax.set_xlabel(f'Variável Independente - {feature_name}')
    ax.set_ylabel('Variável Dependente - Credit_Limit')
 
    return ax.figure


st.set_page_config(layout='wide')
st.title('Resultados do Modelo')

file_loaded, bank = upload_file()

if file_loaded:
    bank_selection = bank.copy()
    X, y, dict_encoders, bank_model = pre_processing(bank)

    # 70% serão usados para treino. 30% serão usados para teste.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Cria uma instância do objeto de escalonamento
    ss = StandardScaler()

    # Transformação de escala das variáveis independentes
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)

    y_pred, mae, mape, mse, rmse = model_training(X_train, y_train, X_test, y_test)

    with st.expander('Métricas'):
        dict_metric = {
            'MAE (Mean Absolute Error)': mae,
            'MAPE (Mean Absolute Percentage Error)': mape,
            'MSE (Mean Squared Error)': mse,
            'RMSE (Root Mean Squared Error)': rmse
        }

        df_metric = pd.DataFrame(dict_metric.items(), columns=['Métrica', 'Valor'])
        st.data_editor(df_metric, disabled=True, hide_index=True, use_container_width=True)

    with st.expander('Visualização dos Resultados'):
        # Inverte a escala nos dados de teste antes da plotagem
        X_test_inverse = ss.inverse_transform(X_test)
        
        var_list = bank_selection.drop('Credit_Limit', axis=1).columns.to_list()
        features = st.multiselect('Selecione a variável:', options=var_list)

        categorical_list = bank.select_dtypes(exclude='number').columns.to_list()
        
        if features:
            for feature in features:
                if feature in categorical_list:
                    st.pyplot(plot_regression(bank_model.columns.get_loc(dict_encoders[feature][1]), 
                                              feature_name=feature, encoded=True, dict_encoders=dict_encoders))
                else:
                    st.pyplot(plot_regression(bank_model.columns.get_loc(feature), feature_name=feature))
