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
    FACE_RECOGNITION_MODEL="hog" \
    # Configurações do Railway
    WEB_CONCURRENCY=1 \
    MAX_WORKERS=1 \
    WORKER_CLASS="uvicorn.workers.UvicornWorker"

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

# Instalar dlib primeiro
COPY requirements.txt .
RUN pip install --no-cache-dir dlib==19.24.1 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn \
    && rm -rf ~/.cache/pip/*

# Copiar apenas os arquivos necessários
COPY api.py .

# Expor a porta que a aplicação vai usar
EXPOSE 8000

# Configurar variáveis de ambiente para a porta
ENV PORT=8000

# Comando para iniciar com Gunicorn e configurações otimizadas
CMD gunicorn api:app \
    --bind 0.0.0.0:$PORT \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 1 \
    --threads 8 \
    --timeout 300 \
    --keep-alive 120 \
    --log-level info \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --backlog 2048
