FROM python:3.9-slim

WORKDIR /app

# Instalar dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY . .

# Configurar variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV WORKERS=1
ENV TIMEOUT=300
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO

# Criar diretório para logs
RUN mkdir -p /var/log/app
RUN chmod 777 /var/log/app

# Expor a porta que a aplicação usa
EXPOSE 8000

# Comando para iniciar a aplicação com configurações otimizadas
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "300", \
     "--workers", "1", \
     "--log-level", "info", \
     "--reload-dir", "/app", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
