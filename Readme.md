# Agente Classificador de Fatores de Acidentes Aeronáuticos (CENIPA)

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.0-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## 📖 Descrição

Este projeto utiliza Machine Learning clássico para analisar e classificar fatores contribuintes em acidentes aeronáuticos, com base nos dados abertos fornecidos pelo **Centro de Investigação e Prevenção de Acidentes Aeronáuticos (CENIPA)**.

O objetivo principal é treinar um modelo de classificação capaz de prever a **área do fator contribuinte** (`fator_area`) com base em suas características descritivas, como nome, aspecto e condicionante.

Além do modelo, este repositório contém um **Agente Inteligente** construído com o framework **Google ADK (Agent Development Kit)**, que permite interagir com o modelo de forma conversacional.

## ⚙️ Como Funciona

O projeto é dividido em dois componentes principais:

1.  **Treinamento do Modelo (`agent_cenipa/train_model.py`)**:
    * Carrega o dataset `fator_contribuinte.csv` utilizando a biblioteca Pandas.
    * Realiza um pré-processamento nos dados, tratando valores ausentes e codificando variáveis categóricas (texto) para um formato numérico que o modelo possa entender, utilizando `LabelEncoder`.
    * Treina um modelo de classificação `RandomForestClassifier` da biblioteca Scikit-learn.
    * Salva o modelo treinado (`cenipa_model.pkl`) e os codificadores (`cenipa_encoders.pkl`) no disco usando `joblib`.

2.  **Agente Inteligente (`agent_cenipa/agent.py`)**:
    * Utiliza o framework **Google ADK** para criar uma interface de conversação.
    * Carrega o modelo e os codificadores salvos na etapa de treinamento.
    * Define uma ferramenta (`tool`) chamada `prever_area_fator`, que recebe as características de um fator contribuinte como entrada.
    * Processa a entrada, utiliza o modelo para fazer a previsão e retorna a área do fator prevista em linguagem natural.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.11
* **Análise de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Serialização do Modelo:** Joblib
* **Framework do Agente:** Google ADK
* **Containerização:** Docker

## 📂 Estrutura do Projeto

A estrutura de pastas do projeto é a seguinte:

```
teste_be_solution/
├── .venv/
└── agent_cenipa/
    ├── .gitignore
    ├── agent.py              # Define o agente e suas ferramentas
    ├── train_model.py        # Script para treinar e salvar o modelo
    ├── Dockerfile            # Define o ambiente para containerização
    ├── requirements.txt      # Lista de dependências Python
    ├── cenipa_model.pkl      # Modelo treinado (gerado)
    ├── cenipa_encoders.pkl   # Encoders (gerado)
    └── fator_contribuinte.csv# Dataset original do CENIPA
```

## 🚀 Como Rodar Localmente

Existem duas maneiras de executar este projeto no seu ambiente local: usando um Ambiente Virtual Python ou usando Docker.

### Pré-requisitos

Antes de começar, garanta que você tenha os seguintes softwares instalados:
* [Git](https://git-scm.com/)
* [Python 3.11+](https://www.python.org/downloads/)
* [Docker](https://www.docker.com/products/docker-desktop/) (para a Opção B)

---

### Opção A: Ambiente Virtual Python (Desenvolvimento)

**1. Clone o Repositório**
```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd teste_be_solution
```

**2. Crie e Ative o Ambiente Virtual**
Este comando deve ser executado na pasta raiz `teste_be_solution`.
```bash
# Criar o ambiente virtual
python -m venv .venv

# Ativar o ambiente
# No Windows (Git Bash ou PowerShell):
source .venv/Scripts/activate
# No macOS/Linux:
# source .venv/bin/activate
```
Seu prompt de comando agora deve mostrar `(.venv)` no início.

**3. Instale as Dependências**
O arquivo de requisitos está dentro da pasta `agent_cenipa`.
```bash
pip install -r agent_cenipa/requirements.txt
```
**3.1 Crie a pasta .env**
GOOGLE_API_KEY = SUACHAVE

**4. Treine o Modelo de Machine Learning**
Este passo é **obrigatório** e deve ser executado antes de usar o agente. Ele criará os arquivos `.pkl` dentro da pasta `agent_cenipa`.
```bash
python agent_cenipa/train_model.py
```
Você verá uma mensagem de sucesso e a acurácia do modelo.

**5. Execute o Agente**
Agora que o modelo está treinado, inicie o agente com o comando do ADK, apontando para o script correto.
```bash
adk run agent_cenipa/agent.py
```
O agente será iniciado no seu terminal, pronto para receber comandos.

---

### Opção B: Docker (Método Recomendado para Simplicidade)

Este método encapsula toda a aplicação, tornando a execução muito mais simples.

**1. Navegue até a Pasta Correta**
Os comandos do Docker devem ser executados de dentro da pasta que contém o `Dockerfile`.
```bash
cd teste_be_solution/agent_cenipa
```

**2. Construa a Imagem Docker**
Este comando lê o `Dockerfile` e monta a imagem da sua aplicação. Certifique-se de que os arquivos `.pkl` e `.csv` já estão na pasta.
```bash
# O "." no final significa "use o diretório atual como contexto de build"
docker build -t cenipa-agent .
```

**3. Execute o Contêiner**
Inicie um contêiner a partir da imagem que você acabou de criar. A flag `-it` garante que você possa interagir com o terminal do agente.
```bash
docker run -it cenipa-agent
```
O agente será iniciado dentro do contêiner, pronto para receber comandos.

## 💬 Como Usar o Agente

Após iniciar o agente (seja pela Opção A ou B), ele estará esperando por um comando. Você pode interagir com ele para usar a ferramenta `prever_area_fator`.

**Exemplo de comando no terminal do agente:**

```
prever_area_fator com fator_nome='JULGAMENTO DE PILOTAGEM', fator_aspecto='PLANEJAMENTO DE VOO', fator_condicionante='Deficiente'
```

**Resposta Esperada do Agente:**

O agente processará a informação, usará o modelo de ML e retornará uma resposta parecida com esta:

```
A previsão da área do fator contribuinte é: FATOR OPERACIONAL
```