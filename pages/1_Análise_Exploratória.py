
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
from sklearn.preprocessing import StandardScaler

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

def plot_numeric_features(serie, kde=True):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    # Criação do histograma com KDE
    sns.histplot(data=serie, kde=kde, ax=ax[0])
    ax[0].set_ylabel('Frequência')
    ax[0].set_xlabel(None)

    if kde:
        ax[0].lines[0].set_color('black')

    # Inclui as linhas verticais com valores da média e mediana
    ax[0].axvline(serie.mean(), label='Média', color='red')
    ax[0].axvline(serie.median(), label='Mediana', color='green')
    ax[0].legend()

    # Criação do boxplor
    sns.boxplot(data=serie, orient='h', ax=ax[1])
    ax[1].set_xlabel(None)

    fig.suptitle(f"Distribuição dos dados da variável '{serie.name}'")
    plt.tight_layout()
    return fig

def plot_categorical_features(serie):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))

    # Ordena os dados em ordem decrescente
    data = serie.value_counts().sort_values(ascending=False)
    data_categories = data.index.tolist()

    # Cria o gráfico de barras
    sns.countplot(x=serie, order=data_categories, ax=ax[0])
    ax[0].set_ylabel('Ocorrências')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30)
    ax[0].set_xlabel(None)

    # Adiciona o total de cada categoria sobre as barras
    for container in ax[0].containers:
        ax[0].bar_label(container)

    # Número de elementos da categoria
    ncat = len(serie.unique())

    if ncat <= 3:
        #Cria o gráfico de pizza
        ax[1].pie(x=data, autopct='%.1f%%', radius=1.1)
        ax[1].legend(data_categories, loc='upper right')
    else:
        # Textos do gráfico tree map
        data_pct = serie.value_counts(normalize=True)
        data_pct_desc = [f'{cat}\n{round(data_pct[cat] * 100, 1)}%' for cat in data_categories]

        #Cria o gráfico tree map
        squarify.plot(sizes=data, label=data_pct_desc, pad=0.05, 
                      color=sns.color_palette("tab10", len(data)), 
                      alpha = 0.7, ax=ax[1])

        ax[1].axis('off')

    plt.suptitle(f"Exibe a contagem das categorias da variável '{serie.name}'")
    plt.tight_layout()
    return fig

def plot_credit_limit_relationships(data, x, hue, split):
    if hue != '' and split:
        g = sns.relplot(data=data, x=x, y='Credit_Limit', hue=hue, col=hue, col_wrap=2, legend=False)
        fig = g.fig
    else:
        plt.figure(figsize=(10,6))

        if hue != '':
            ax = sns.scatterplot(data=data, x=x, y='Credit_Limit', hue=hue) 
            plt.legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncols=2, borderaxespad=0)
        else:
            ax = sns.scatterplot(data=data, x=x, y='Credit_Limit')   
            
        fig = ax.figure

        # Cria um título e altera o rótulo dos eixos
        plt.title(f"Relação entre as variáveis '{x}' e 'Credit_Limit'")

    plt.tight_layout()
    return fig

@st.cache_data
def plot_correlation_matrix(data):
    # Cria uma instância do objeto de escalonamento
    ss = StandardScaler()

    # Cria uma cópia da base original
    df_scale = data.select_dtypes('number').copy()

    # Coloca todas as variáveis numéricas na mesma escala
    df_scale.loc[:, df_scale.columns] = ss.fit_transform(df_scale)

    # Cria um mapa de calor com a correlação entre as variáveis.
    plt.figure(figsize=(8,5))
    ax = sns.heatmap(df_scale.corr(), annot=True, fmt=".1f", linewidth=.5, cmap="YlOrRd")
    return ax.figure


st.set_page_config(layout='wide')
st.title('Análise Exploratória de Dados (EDA)')

file_loaded, bank = upload_file()

if file_loaded:
    with st.expander('Visualização Unidimensional'):
        col1, col2, col3 = st.columns(3)

        with col1:
            var_type = st.radio('Selecione o tipo da variável:', options=['Numérica', 'Categórica'], horizontal=True)           

        with col2:
            if var_type == 'Numérica':
                var_list = bank.select_dtypes('number').drop('Credit_Limit', axis=1).columns.to_list()
                numeric = True
            else:
                var_list = bank.select_dtypes(exclude='number').columns.to_list()
                numeric = False

            features = st.multiselect('Nome', options=var_list) 

        with col3:
            st.write('')
            st.write('')
            kde = st.checkbox('Visualizar KDE', value=False)        

        if features:
            for var_name in features:
                if numeric:
                    st.pyplot(plot_numeric_features(bank[var_name], kde))
                else:
                    st.pyplot(plot_categorical_features(bank[var_name]))


    with st.expander('Visualização Multidimensional'):
        col1, col2, col3 = st.columns(3)

        with col1:
            x_list = bank.select_dtypes('number').drop('Credit_Limit', axis=1).columns.to_list()
            x_list.insert(0, '')
            axis_x = st.selectbox('Eixo X', options=x_list)

        with col2:
            axis_y = st.selectbox('Eixo Y', options=['Credit_Limit'], disabled=True)

        with col3:
            group_list = bank.select_dtypes(exclude='number').columns.to_list()
            group_list.insert(0, '')

            group = st.selectbox('Grupo', options=group_list)    
            split_group = st.checkbox('Separar categorias', value=False)  

        if axis_x != '':
            st.pyplot(plot_credit_limit_relationships(bank, axis_x, group, split_group))      
        
    with st.expander('Matriz de Correlação'):
        st.pyplot(plot_correlation_matrix(bank))
