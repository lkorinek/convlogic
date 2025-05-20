FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

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
    ca-certificates

# Install Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip install --upgrade pip setuptools wheel

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3

RUN pip install --upgrade pip
RUN pip install -U pip setuptools wheel ninja
RUN pip install --upgrade requests

# Provide the Weights & Biases secret
ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY
