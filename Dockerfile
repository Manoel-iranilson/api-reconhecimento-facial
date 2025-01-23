FROM python:3.9-slim

# Definir variáveis de ambiente para otimização
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Otimizações de memória Python
    PYTHONMALLOC=malloc \
    MALLOC_TRIM_THRESHOLD_=100000 \
    # Otimizações OpenCV
    OPENCV_OPENCL_RUNTIME="" \
    OPENCV_OPENCL_DEVICE="" \
    # Otimizações face_recognition
    FACE_RECOGNITION_MODEL="hog"

WORKDIR /app

# Instalar dependências do sistema necessárias para dlib e OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir dlib==19.24.1 \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip/*

# Copiar apenas os arquivos necessários
COPY api.py .

# Expor a porta que a aplicação vai usar
EXPOSE 8000

# Comando para iniciar a aplicação
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --limit-max-requests 1000 --timeout-keep-alive 75
