FROM python:3.10-slim

WORKDIR /code

# Instalar dependências do sistema e dlib pré-compilado
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

# Instalar dlib via pip primeiro
RUN pip install dlib==19.24.1 --no-cache-dir

COPY ./requirements.txt /code/requirements.txt
COPY ./api.py /code/api.py

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Instalar face-recognition por último
RUN pip install face-recognition==1.3.0

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
