# modelo-regressao-salario-python
Modelo de regressão linear em Python para prever salário com base em idade, anos de experiência, nível de educação e área de atuação, usando pandas e scikit‑learn.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib

df = pd.read_csv('clientes-v3-preparado.csv')

x = df[['anos_experiencia']]  # Preditor
x = df[['idade', 'anos_experiencia', 'nivel_educacao_cod', 'area_atuacao_cod']] # Preditor
y = df['salario'] # Prever

# Dividir dados: Treinamento e Teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Criare e treinar modelo - Regressão Linear
modelo_linear = LinearRegression()
modelo_linear.fit(x_train, y_train)

# Provar valores de teste
y_previsto = modelo_linear.predict(x_test)

# Metricas de avaliação
r2 = r2_score(y_test, y_previsto)
print(f'Coeficiente de determinacao - R2: {r2:.2f}')

rmse = root_mean_squared_error(y_test, y_previsto)
print(f"Raiz do Erro Quadratico Médio - RMSE: {rmse:.2f}")
print(f"Desvio Padrão do campo Salário: {df['salario'].std()}")


# Salvar modelo treinado
joblib.dump(modelo_linear, 'modelo_regressao_linear.pkl')
