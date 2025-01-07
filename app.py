import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Geração de dados sintéticos de exemplo para treinar o modelo (adaptado para o Brasil)
def generate_house_data(n_samples=100):
    np.random.seed(42)
    # Tamanho em metros quadrados (m²) - média de 150 m², desvio padrão de 50 m²
    size = np.random.normal(150, 50, n_samples)
    size = np.abs(size) # Garante que não teremos áreas negativas
    
    # Preço em Reais (R$) - valor base por m² mais um ruído aleatório
    # Assumindo um valor médio de R$ 5.000 por m² (valor fictício para o exemplo)
    price = size * 5000 + np.random.normal(0, 50000, n_samples)
    price = np.abs(price) # Garante que não teremos preços negativos
    
    return pd.DataFrame({'size_m2': size, 'price_reais': price})

# Função para instanciar e treinar o modelo de regressão linear
def train_model():
    df = generate_house_data()
    
    # Divisão de dados em treino e teste
    X = df[['size_m2']]
    y = df['price_reais']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treina o modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

# Interface de Usuário do Streamlit para o modelo implantado
def main():
    st.title('🏠 Simulador de Preço de Imóveis (Brasil)')
    st.write('Insira o tamanho do imóvel em metros quadrados para prever seu preço de venda.')
    
    # Treina o modelo
    model = train_model()
    
    # Input do usuário
    size = st.number_input('Tamanho do imóvel (metros quadrados)', 
                           min_value=30, 
                           max_value=500, 
                           value=100)
    
    if st.button('Prever Preço'):
        # Realiza a previsão
        prediction = model.predict([[size]])
        
        # Mostra o resultado
        st.success(f'Preço estimado: R$ {prediction[0]:,.2f}')
        
        # Visualização
        df = generate_house_data()
        fig = px.scatter(df, x='size_m2', y='price_reais', 
                         title='Relação entre Tamanho e Preço',
                         labels={'size_m2': 'Tamanho (m²)', 'price_reais': 'Preço (R$)'})
        fig.add_scatter(x=[size], y=[prediction[0]], 
                        mode='markers', 
                        marker=dict(size=15, color='red'),
                        name='Previsão')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()