FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
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
RUN pip install -U pip setuptools wheel
RUN pip install --upgrade requests

# Install Git LFS
RUN apt-get update && \
    apt-get install -y curl git && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

# Provide the Weights & Biases secret
ARG WANDB_API_KEY
ENV WANDB_API_KEY=$WANDB_API_KEY
