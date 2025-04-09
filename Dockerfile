FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .

# Update pip and install build tools
RUN pip install --no-cache-dir -U pip wheel setuptools

# Install dlib using a pre-built wheel
RUN pip install --no-cache-dir --prefer-binary dlib==19.24.0

# Install face_recognition (which depends on dlib)
RUN pip install --no-cache-dir --prefer-binary face_recognition

# Now install the rest of the requirements
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]