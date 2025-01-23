FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema necessárias para dlib e OpenCV
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

# Copiar requirements primeiro para aproveitar o cache do Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar dlib via pip primeiro
RUN pip install dlib==19.24.1 --no-cache-dir

# Instalar face-recognition por último
RUN pip install face-recognition==1.3.0

# Copiar o código da aplicação
COPY api.py .

# Expor a porta que a aplicação vai usar
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
