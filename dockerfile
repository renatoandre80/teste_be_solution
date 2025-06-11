# Dockerfile

#Imagem Base
FROM python:3.11-slim

# Estágio 2: Configuração do Ambiente
WORKDIR /agent_cenipa

COPY requirements.txt .
# Instala as dependências
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

COPY agent_cenipa/ .

CMD ["python", "agent.py"]