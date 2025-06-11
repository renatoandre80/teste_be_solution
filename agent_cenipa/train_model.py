# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

print("Iniciando o treinamento do modelo CENIPA...")

# --- 1. Carregamento e Preparação dos Dados ---
# Carregar o dataset
try:
    # O delimitador é ponto e vírgula (;)
    df = pd.read_csv('fator_contribuinte.csv', delimiter=';', encoding='latin-1') 
except FileNotFoundError:
    print("Erro: Arquivo 'fator_contribuinte.csv' não encontrado.")
    print("Por favor, certifique-se de que o arquivo está no mesmo diretório que este script.")
    exit()

# Selecionar as colunas que serão usadas para prever (features) e a coluna alvo (target)
# Para este exemplo, usaremos o nome, aspecto e condicionante do fator.
features = ['fator_nome', 'fator_aspecto', 'fator_condicionante']
target = 'fator_area'

# Remover linhas onde a variável alvo ou as features são nulas
df.dropna(subset=[target] + features, inplace=True)

# Converter as variáveis categóricas em números, pois o modelo não entende texto
# Usaremos um LabelEncoder para cada coluna
encoders = {}
for col in features + [target]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le # Guardar o encoder para usar no 'agente'

# Separar os dados em X (features) e y (target)
X = df[features]
y = df[target]

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dados carregados e preparados. Total de {len(X_train)} amostras de treino.")

# --- 2. Treinamento do Modelo ---

# Criar o modelo de classificação (Random Forest)
# RandomForest é um bom ponto de partida para problemas de classificação com dados tabulares
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Treinar o modelo com os dados de treino
model.fit(X_train, y_train)

print("Modelo treinado com sucesso!")

# --- 3. Avaliação (Opcional, mas recomendado) ---

# Fazer previsões com os dados de teste
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo no conjunto de teste: {accuracy:.2f}")


# --- 4. Salvando o Modelo e os Encoders ---

# O caminho para salvar o modelo
model_path = 'cenipa_model.pkl'
joblib.dump(model, model_path)
print(f"Modelo salvo em: '{model_path}'")

# Salvar também os encoders, pois serão necessários para a previsão no agente
encoders_path = 'cenipa_encoders.pkl'
joblib.dump(encoders, encoders_path)
print(f"Encoders salvos em: '{encoders_path}'")