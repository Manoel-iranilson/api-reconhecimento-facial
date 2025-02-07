FROM python:3.10-slim-bullseye

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    version: '3.8'
    
    services:
      api:
        build: .
        container_name: reconhecimento-facial-api
        ports:
          - "8000:8000"
        env_file:
          - .env
        volumes:
          - ./:/app
        restart: unless-stopped
        deploy:
          resources:FROM python:3.10-slim-bullseye
          
          WORKDIR /app
          
          # Instalar dependências do sistema
          RUN apt-get update && apt-get install -y --no-install-recommends \
              build-essential \
              cmake \
              libopenblas-dev \
              liblapack-dev \
              libgl1-mesa-glx \
              libglib2.0-0 \
              && rm -rf /var/lib/apt/lists/*
          
          # Copiar requirements e instalar dependências
          COPY requirements.txt .
          RUN pip install --no-cache-dir -r requirements.txt
          
          # Copiar código da aplicação
          COPY . .
          
          # Expor porta da aplicação
          EXPOSE 8000
          
          # Variáveis de ambiente
          ENV PYTHONUNBUFFERED=1
          
          # Comando para iniciar a aplicação
          CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
            limits:
              cpus: '1.5'
              memory: 2G
            reservations:
              cpus: '1'
              memory: 1G&& rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Expor porta da aplicação
EXPOSE 8000

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1

# Comando para iniciar a aplicação
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
