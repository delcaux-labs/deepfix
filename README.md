# 🔍 DeepFix

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-3.0+-orange.svg)](https://mlflow.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-2.0+-yellow.svg)](https://pytorch-lightning.readthedocs.io/)

DeepFix is an intelligent AI assistant that integrates directly into the ML workflow (starting with PyTorch Lightning and MLflow). Our platform automatically diagnoses common bugs in machine learning and provides a prioritized list of solutions backed by industry/research best practices.

## ✨ Features

- **Automatic Bug Detection**: Identifies common ML issues automatically
- **Prioritized Solutions**: Get ranked suggestions based on best practices
- **Workflow Integration**: Seamlessly works with PyTorch Lightning and MLflow
- **Research-Backed**: Solutions are grounded in industry standards and research

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

### Basic Usage

```python
from deepfix import DeepFix

# Initialize DeepFix with your MLflow experiment
deepfix = DeepFix(mlflow_experiment="my_experiment")

# Run diagnostics on your model
results = deepfix.diagnose(model, data_loader)

# Get prioritized solutions
solutions = results.get_solutions()
```

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/delcaux-labs/deepfix/issues)
- **Email**: Contact us at team@delcaux-labs.com

---

**Built with ❤️ for the ML community**