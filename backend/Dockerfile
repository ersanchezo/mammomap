# backend/Dockerfile
FROM python:3.10-slim

# Install system dependencies for OpenCV and Git
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install core ML and Image Processing libraries
# We install standard PyTorch; Modal will automatically inject CUDA drivers at runtime
RUN pip install --no-cache-dir \
    torch torchvision \
    huggingface_hub \
    opencv-python-headless \
    scikit-image \
    Pillow \
    numpy \
    "timm>=0.9.11" \
    peft \
    monai \
    fastapi

# Clone the SEQUOIA repository directly into the Docker image
RUN git clone https://github.com/gevaertlab/sequoia-pub.git /app/sequoia

# Ensure Python can import from the cloned repo
ENV PYTHONPATH="/app/sequoia"