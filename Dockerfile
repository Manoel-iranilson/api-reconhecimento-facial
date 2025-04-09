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
RUN pip install --no-cache-dir wheel setuptools
# Install dlib separately first with CMake configuration
RUN pip install --no-cache-dir dlib --install-option="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
# Install face_recognition before other requirements
RUN pip install --no-cache-dir face_recognition
# Now install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]