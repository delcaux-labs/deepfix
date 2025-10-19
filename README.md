# 🔍 DeepFix

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-3.0+-orange.svg)](https://mlflow.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0+-yellow.svg)](https://pytorch-lightning.readthedocs.io/)

DeepFix is an AI agent assistant that automatically diagnoses common bugs in machine learning and provides a prioritized list of solutions backed by industry/research best practices. It integrates directly into ML workflows.

## ✨ Features

- **Automatic Bug Detection**: Identifies common ML issues automatically
- **Prioritized Solutions**: Get ranked suggestions based on best practices
- **Workflow Integration**: Seamlessly works with PyTorch Lightning and MLflow
- **Research-Backed**: Solutions are grounded in industry standards and research

## 🚀 Quick Start

### Installation

#### Option 1: Docker (Recommended for Server Deployment)

```bash
# Clone the repository
git clone https://github.com/delcaux-labs/deepfix.git
cd deepfix

# Copy environment example and configure
cp env.example .env
# Edit .env with your API keys

# Start the server using docker-compose
docker-compose up -d

# Or using Make
make docker-compose-up
```

See [Docker Deployment Guide](docs/DOCKER.md) for detailed instructions.

#### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/delcaux-labs/deepfix.git
cd deepfix

# Install with uv (recommended)
uv venv --python 3.11
uv pip install -e .
```

### Basic Usage

```python
from deepfix.client import DeepFixClient
from deepfix.client.zoo.datasets.foodwaste import load_train_and_val_datasets

client = DeepFixClient(api_url="http://localhost:8844",timeout=120)

# Load image datasets
train_data, val_data = load_train_and_val_datasets(
    image_size=448,
    batch_size=8,
    num_workers=4,
    pin_memory=False,)

dataset_name="cafetaria-foodwaste"
client.ingest_dataset(dataset_name=dataset_name,
                    train_data=train_data,
                    test_data=val_data,
                    train_test_validation=True,
                    data_integrity=True,
                    batch_size=8,
                    overwrite=False
                    )

result = client.diagnose_dataset(dataset_name=dataset_name)
print(result.to_text())
```

## 🐳 Docker Deployment

DeepFix can be deployed as a containerized service:

```bash
# Quick start with docker-compose
docker-compose up -d

# Or using Makefile commands
make build          # Build the Docker image
make run            # Run the container
make logs           # View logs
make stop           # Stop the container
```

For production deployment and advanced configuration, see the [Docker Deployment Guide](docs/DOCKER.md).

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/delcaux-labs/deepfix/issues)
- **Email**: Contact us at fadel.seydou@delcaux.com

---

**Built with ❤️ for the ML community**