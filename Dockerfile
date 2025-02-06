FROM python:3.9

WORKDIR /app

# Instalar dependências do sistema necessárias para dlib e face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Instalar dlib primeiro
RUN pip install dlib==19.24.1

# Copiar requirements e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY . .

# Expor a porta que a aplicação usa
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
