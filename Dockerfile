# Use Python 3.9 as base image to match your venv
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV PYTHONPATH=/app

# Install system dependencies based on packages.txt
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libfontconfig1 \
    libice6 \
    tesseract-ocr \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies but exclude torch-related packages
# We'll handle those separately
RUN grep -v "torch" requirements.txt > cpu_requirements.txt && \
    pip install --no-cache-dir -r cpu_requirements.txt && \
    rm cpu_requirements.txt

# Install PyTorch CPU version
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p input output

# Expose Streamlit port
EXPOSE 8501

# Using an entrypoint script allows for more flexibility
RUN echo '#!/bin/bash\nstreamlit run app.py --server.port=8501 --server.address=0.0.0.0' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Run the application
ENTRYPOINT ["/app/entrypoint.sh"] 