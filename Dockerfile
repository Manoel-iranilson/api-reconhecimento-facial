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

# Configurar limites de memória para o container
ENV WORKERS=1 \
    TIMEOUT=300 \
    KEEP_ALIVE=120 \
    MAX_REQUESTS=100 \
    MAX_REQUESTS_JITTER=10

# Comando para iniciar a aplicação com Gunicorn
CMD ["gunicorn", "api:app", "--bind", "0.0.0.0:8000", "--worker-class", "uvicorn.workers.UvicornWorker", "--workers", "1", "--timeout", "300", "--keep-alive", "120", "--max-requests", "100", "--max-requests-jitter", "10", "--log-level", "info"]
