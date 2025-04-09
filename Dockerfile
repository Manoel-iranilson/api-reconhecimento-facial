FROM python:3.10-slim-bullseye

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    libsm6 \
    libxext6 \
    libopenblas-dev \
    liblapack-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar wheel para compilação de pacotes
RUN pip install --no-cache-dir wheel

# Copiar requirements.txt e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o resto do código
COPY . .

# Expor a porta 8000 (padrão do FastAPI)
EXPOSE 8000

# Comando para executar a aplicação
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
