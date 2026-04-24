# 🔍 DeepFix

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-3.0+-orange.svg)](https://mlflow.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0+-yellow.svg)](https://pytorch-lightning.readthedocs.io/)

DeepFix is a tool that automatically diagnoses common bugs in machine learning and provides a prioritized list of solutions backed by industry/research best practices. It integrates directly into ML workflows.

## ✨ Features

- **Automatic Bug Detection**: Identifies common ML issues automatically
- **Prioritized Solutions**: Get ranked suggestions based on best practices
- **Workflow Integration**: Seamlessly works with PyTorch Lightning and MLflow
- **Research-Backed**: Solutions are grounded in industry standards and research
- **Multi-Artifact Analysis**: Analyzes datasets, model checkpoints, training logs, and deepchecks reports

## 📚 Documentation

**Full documentation is available at**: [Documentation Site](https://delcaux-labs.github.io/deepfix/)

Or build and serve locally:
```bash
uv pip install -e .
uv run mkdocs serve
```

The documentation includes:
- [Getting Started Guide](docs/getting-started/installation.md) - Installation and setup
- [Quickstart Guide](docs/getting-started/quickstart.md) - Get up and running quickly
- [Guides](docs/guides/image-classification.md) - Use case guides and tutorials
- [API Reference](docs/api-reference/index.md) - Complete API documentation
- [Architecture](docs/architecture/overview.md) - System architecture and design
- [Deployment](docs/deployment/docker.md) - Deployment guides
- [Contributing](docs/contributing/guidelines.md) - Contributing guidelines

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/delcaux-labs/deepfix.git
cd deepfix

# Install with uv (recommended)
uv venv --python 3.11
uv pip install -e .
```

See the [Installation Guide](docs/getting-started/installation.md) for detailed instructions and Docker deployment.

### Basic Usage

```python
from deepfix_sdk.client import DeepFixClient
from deepfix_sdk.zoo.datasets.foodwaste import load_train_and_val_datasets
from deepfix_sdk.data.datasets import ImageClassificationDataset

# Start server (in separate terminal)
# uv run deepfix-server launch -e deepfix-server/.env -port 8844

# Initialize client
client = DeepFixClient(api_url="http://localhost:8844", timeout=120)

# Load and wrap dataset
dataset_name = "cafetaria-foodwaste"
train_data, val_data = load_train_and_val_datasets(
    image_size=448, batch_size=8, num_workers=4, pin_memory=False
)
train_data = ImageClassificationDataset(dataset_name=dataset_name, dataset=train_data)
val_data = ImageClassificationDataset(dataset_name=dataset_name, dataset=val_data)

# Diagnose dataset
result = client.get_diagnosis(
    train_data=train_data,
    test_data=val_data,
    batch_size=8,
    language="english",
)

print(result.to_text())
```

For more examples, see the [Quickstart Guide](docs/getting-started/quickstart.md).

## 📦 Packages

This repository contains multiple packages:

- **deepfix-core**: Core models and types
- **deepfix-sdk**: Client SDK for interacting with DeepFix server
- **deepfix-server**: Analysis server with agentic reasoning
- **deepfix-kb**: Knowledge base for best practices

See the [Architecture Documentation](docs/architecture/overview.md) for details.

## Deployment

### Initial Deployment

```bash
# Build images
docker compose -f docker-compose.prod.yml build

# Start services
docker compose -f docker-compose.prod.yml up -d

# Verify all services are healthy
docker compose -f docker-compose.prod.yml ps
```

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see the [Contributing Guidelines](docs/contributing/guidelines.md) for details.

## 📧 Support

- **Documentation**: [Full Documentation](https://docs.delcaux.com)
- **Issues**: [GitHub Issues](https://github.com/delcaux-labs/deepfix/issues)
- **Email**: Contact us at fadel.seydou@delcaux.com

## Acknowledgements

The project on which this publication is based was funded by the Federal Ministry of Research, Technology and Space under the funding code “KI-Servicezentrum Berlin-Brandenburg” **16IS22092**. Responsibility for the content of this publication remains with the author.


---

**Built with ❤️ for the ML community**
