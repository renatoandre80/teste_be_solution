
import joblib
import pandas as pd
from google.adk.agents import Agent
import os

# --- Bloco de Carregamento ---
# Obtém o caminho absoluto para o diretório onde este script (agent.py) está
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define os caminhos para o modelo e para os encoders
MODEL_PATH = os.path.join(script_dir, 'cenipa_model.pkl')
ENCODERS_PATH = os.path.join(script_dir, 'cenipa_encoders.pkl')

# Tenta carregar o modelo e os encoders UMA VEZ na inicialização do agente
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    print("--- Modelo e Encoders CENIPA carregados com sucesso! ---")
except FileNotFoundError:
    model = None
    encoders = None
    print("--- ERRO: Arquivos de modelo (cenipa_model.pkl) ou encoders (cenipa_encoders.pkl) não encontrados. ---")
    print("--- Certifique-se de executar o 'train_model.py' primeiro. ---")
except Exception as e:
    model = None
    encoders = None
    print(f"--- ERRO ao carregar modelo ou encoders: {e} ---")



def prever_area_fator(fator_nome: str, fator_aspecto: str, fator_condicionante: str) -> str:
    """
    Prevê a área de um fator contribuinte de acidente aeronáutico.
    Carrega um modelo RandomForest pré-treinado para classificar novos dados.
    """
    print(f"\n--- [DEBUG] Iniciando a função prever_area_fator ---")

    # Verifica se o modelo foi carregado corretamente na inicialização
    if model is None or encoders is None:
        error_msg = "O modelo e/ou os encoders não estão disponíveis. A previsão não pode ser realizada."
        print(f"--- [DEBUG] ERRO: {error_msg} ---")
        return error_msg

    try:
        print(f"--- [DEBUG] Dados recebidos: nome='{fator_nome}', aspecto='{fator_aspecto}', condicionante='{fator_condicionante}'")
        
        #DataFrame com os dados de entrada
        input_data = pd.DataFrame([[fator_nome, fator_aspecto, fator_condicionante]], 
                                  columns=['fator_nome', 'fator_aspecto', 'fator_condicionante'])

        # Codifica os dados de entrada usando os encoders salvos
        # É CRUCIAL usar os mesmos encoders do treinamento
        for col in input_data.columns:
            le = encoders[col]
            # Usamos 'transform'. Se uma categoria for nova, causará um erro.
            # Isso é esperado, pois o modelo só conhece o que viu no treino.
            input_data[col] = le.transform(input_data[col])
        
        print(f"--- [DEBUG] Dados de entrada codificados: {input_data.values}")

        # Realiza a previsão
        predicted_code = model.predict(input_data)[0]
        print(f"--- [DEBUG] Código da previsão: {predicted_code}")
        
        # Decodifica a previsão para obter o nome da área
        # o encoder da variável alvo ('fator_area')
        predicted_area = encoders['fator_area'].inverse_transform([predicted_code])[0]
        
        result = f"A previsão da área do fator contribuinte é: {predicted_area}"

    except ValueError as e:
        # Este erro ocorre se um dos valores de entrada (ex: 'fator_nome') for uma categoria que o modelo nunca viu
        error_msg = f"Erro de valor: Uma das categorias fornecidas não foi vista durante o treinamento. Detalhes: {e}"
        print(f"--- [DEBUG] ERRO: {error_msg} ---")
        return error_msg
    except Exception as e:
        error_msg = f"Ocorreu um erro inesperado durante a previsão: {e}"
        print(f"--- [DEBUG] ERRO: {error_msg} ---")
        return error_msg
    
    print(f"--- [DEBUG] Previsão concluída com sucesso. ---")
    return result


# --- Definição do Agente ---
# O framework ADK irá carregar esta variável quando chamado pela linha de comando.
root_agent = Agent(
    name="agent_cenipa",
    model= 'gemini-2.0-flash', 
    tools=[prever_area_fator],
)