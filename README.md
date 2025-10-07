# ConvLogic

<p align="left">
  <img src="assets/logo.png" alt="ConvLogic Logo" width="250"/>
</p>

**ConvLogic** is a reimplementation of the convolutional differentiable logic gate network from the paper:
[**Convolutional Differentiable Logic Gate Networks**](https://arxiv.org/abs/2411.04732)

Since the official implementation is not yet available, this repository provides a custom implementation based on the original [**difflogic**](https://github.com/Felix-Petersen/difflogic) project, extending it with:
- Convolutional logic layer (`ConvLogicLayer`) and Tree logic layer (`TreeLogicLayer`)
- A modern PyTorch Lightning training pipeline
- Hydra-based config management for clean experimentation
- Updating the project to Python3.11+ and PyTorch 2.8.0+
- Support for distributed training

Features:
- Implementation of models for CIFAR-10 and MNIST datasets
- `ConvLogicLayer` also implements the complete 3-stage pipeline (Convolution + tree-like structure), which makes the training faster
  (the paper mentions that it takes ~30s to train one epoch on RTX 4090 GPU for the large CIFAR-10 model. This reimplementation takes ~45 seconds per epoch on the same GPU)
- Faster implementation of the original difflogic
- Training pipeline with modern config/logging support

Not yet implemented:
- Channel grouping (selections of channels separated into groups)
- FPGA support

---

## Quick Start

Create environment with Python 3.11:

```bash
conda create -n convlogic python=3.11
conda activate convlogic
```

Clone and install:

```bash
git clone https://github.com/lkorinek/convlogic.git
cd convlogic
pip install -v .
```

> [!TIP]
> Use `CPU_ONLY=1` to run without a GPU (does not support training).

Run training:

```bash
python3 src/train.py model=cifar10_s trainer.max_epochs=200
```

> [!NOTE]
> If you want to try the ConvLogic model right away without setting up anything locally, check out the [**ConvLogic MNIST Demo Notebook**](notebooks/ConvLogic_Demo.ipynb), which you can run directly in [**Google Colab**](https://colab.research.google.com/github/lkorinek/convlogic/blob/main/notebooks/ConvLogic_Demo.ipynb).

---

### 🧪 Tested Environment

| Python | PyTorch | CUDA   | cuDNN | OS           | GPU            |
|--------|---------|--------|-------|--------------|----------------|
| 3.11   | 2.8.0   | 12.2.2 | 8.x   | Ubuntu 22.04 | NVIDIA RTX 4090 |

> 🐳 Docker base: `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04`

---

## 🔧 Configuration

ConvLogic uses [Hydra](https://hydra.cc/) to manage modular configs stored in `configs/`.

### Available Models

| Model Name   | Dataset      | Size    | ConvLogic Accuracy  | Paper Accuracy |
|--------------|--------------|---------|---------------------|----------------|
| `mnist_s`    | MNIST        | Small   | 98.32%              | 98.46%         |
| `mnist_m`    | MNIST        | Medium  | 98.81%              | 99.23%         |
| `mnist_l`    | MNIST        | Large   | TBD                 | TBD            |
| `cifar10_s`  | CIFAR-10-3   | Small   | 59.84%              | 60.38%         |
| `cifar10_m`  | CIFAR-10-3   | Medium  | 69.15%              | 71.01%         |
| `cifar10_b`  | CIFAR-10-31  | Big     | TBD                 | TBD            |
| `cifar10_l`  | CIFAR-10-31  | Large   | TBD                 | TBD            |

Each model configuration is defined in `configs/model/`, e.g. `configs/model/mnist_s.yaml`.

### Example config:

```yaml
lr: 0.01
weight_decay: 0.0
k: 64
tau: 6.5
dataset_name: mnist
batch_size: 256
```

> [!NOTE]
> **Dataset names:** If the dataset includes a suffix like `cifar10-3`, the number (e.g., `3`) defines the number of threshold levels used to quantize each input channel. If no number is specified, the default is 1 threshold level (i.e., binary input per channel).

Override via CLI:

```bash
python src/train.py model.lr=0.001 model.k=32
```

For more options, see `config.yaml`.

---

## 🐳 Quick Start (Docker-based)

### 1. Build the Base Image

```bash
docker build --build-arg WANDB_API_KEY=your_api_key_here -t convlogic-base -f dockerfiles/base.dockerfile .
```

### 2. Build the Training Image

```bash
docker build -t convlogic-train -f dockerfiles/train.dockerfile .
```

---

## Run Training

```bash
docker run --rm --gpus device=0 --name convlogic convlogic-train model=mnist_s
docker run --rm --gpus device=0 --name convlogic convlogic-train model=mnist_m
docker run --rm --gpus device=0 --name convlogic convlogic-train model=cifar10_s
docker run --rm --gpus device=0 --shm-size=8g --name convlogic convlogic-train model=cifar10_m
```

> [!WARNING]
> For larger models like `cifar10_l`, you might need to increase Docker shared memory using `--shm-size=8g`.

### 📊 With wandb Logging

Set in config:

```yaml
logging:
  wandb: true
```

Or via CLI:

```bash
docker run --rm --gpus device=0 --name convlogic convlogic-train model=mnist_s logging.wandb=true
```

Make sure your environment includes `WANDB_API_KEY`.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This repository builds upon the [difflogic](https://github.com/Felix-Petersen/difflogic) project.
