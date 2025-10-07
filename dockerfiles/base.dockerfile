FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Set the CUDA architecture list.
# This should match your GPU.
# For example:
#   - RTX 4090 → 8.9
ENV TORCH_CUDA_ARCH_LIST="8.9"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    ninja-build \
    wget \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    build-essential \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Install Python 3.11
RUN add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
 && python3 -m ensurepip --upgrade \
 && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python3 -m pip install --no-cache-dir requests

# Provide the Weights & Biases secret
ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY
