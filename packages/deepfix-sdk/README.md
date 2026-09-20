# 🚀 DeepFix SDK

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](../LICENSE)

The **DeepFix SDK** is a Python client library that enables seamless communication with the DeepFix server. It provides an easy-to-use interface for diagnosing machine learning datasets, ingesting data with quality checks, and leveraging AI-powered recommendations to improve your ML workflows.

## 📋 Overview

DeepFix SDK simplifies interactions with the DeepFix analysis engine by providing:

- **Dataset Diagnosis**: Automatically analyze datasets to identify potential issues
- **Data Ingestion**: Ingest datasets with built-in quality validation
- **Multiple Data Types**: Support for image classification, tabular, NLP, and vision datasets
- **Integrated Reporting**: Comprehensive analysis results with actionable insights
- **MLflow Integration**: Seamless integration with MLflow for experiment tracking

## 🔧 Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Install from PyPI

```bash
pip install deepfix-sdk
```

### Install from Source with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/delcaux-labs/deepfix.git
cd deepfix/deepfix-sdk

# Create virtual environment
uv venv --python 3.11

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install SDK in development mode
uv pip install -e .
```

## 🚀 Quick Start

### 1. Set Up Your Environment

Create a `.env` file with your API credentials:

```bash
export DEEPFIX_API_KEY="your-api-key-here"
export MLFLOW_TRACKING_URI="http://localhost:5000"  # Optional
```

### 2. Initialize the Client

```python
from deepfix_sdk import DeepFixClient

# Create client pointing to your DeepFix server
client = DeepFixClient(
    api_url="http://localhost:8844",  # Default server address
    timeout=30,  # Request timeout in seconds
)
```

### 3. Diagnose a Dataset

```python
# Analyze an existing dataset
response = client.diagnose_dataset(
    dataset_name="my-dataset",
    language="english",  # Analysis language
)

# Display results
print(response.to_text())
```

## 💡 Usage Examples

### Example 1: Diagnose an Image Classification Dataset

```python
from deepfix_sdk import DeepFixClient
from deepfix_sdk.zoo.datasets.foodwaste import load_train_and_val_datasets
from deepfix_sdk.data.datasets import ImageClassificationDataset

# Initialize client
client = DeepFixClient(api_url="http://localhost:8844", timeout=120)

# Load image classification dataset
train_data, val_data = load_train_and_val_datasets(
    image_size=448, batch_size=8, num_workers=4, pin_memory=False
)

# Wrap datasets
train_dataset = ImageClassificationDataset(
    dataset_name="foodwaste-classification", dataset=train_data
)
val_dataset = ImageClassificationDataset(
    dataset_name="foodwaste-classification", dataset=val_data
)

# Ingest dataset with quality checks
client.ingest(
    dataset_name="foodwaste-classification",
    data_type="image",
    train_data=train_dataset,
    test_data=val_dataset,
    train_test_validation=True,
    data_integrity=True,
    batch_size=8,
    overwrite=False,
)

# Run diagnosis
result = client.diagnose_dataset(dataset_name="foodwaste-classification")

# View analysis results
print(result.to_text())
```

### Example 2: Ingest and Analyze Tabular Data

```python
import pandas as pd
from deepfix_sdk import DeepFixClient
from deepfix_sdk.data.datasets import TabularDataset

# Initialize client
client = DeepFixClient(api_url="http://localhost:8844")

# Load your tabular data
df_train = pd.read_csv("train_data.csv")
df_test = pd.read_csv("test_data.csv")

# Wrap datasets
train_dataset = TabularDataset(dataset_name="my-tabular-dataset", data=df_train)
test_dataset = TabularDataset(dataset_name="my-tabular-dataset", data=df_test)

# Ingest dataset
client.ingest(
    dataset_name="my-tabular-dataset",
    data_type="tabular",
    train_data=train_dataset,
    test_data=test_dataset,
    train_test_validation=True,
    data_integrity=True,
    overwrite=False,
)

# Diagnose
result = client.diagnose_dataset(dataset_name="my-tabular-dataset")
print(result.to_text())
```

### Example 3: Work with NLP Datasets

```python
from datasets import load_dataset
from deepfix_sdk import DeepFixClient
from deepfix_sdk.data.datasets import NLPDataset

# Initialize client
client = DeepFixClient(api_url="http://localhost:8844")

# Load NLP dataset
train_data = load_dataset("imdb", split="train")
test_data = load_dataset("imdb", split="test")

# Wrap datasets
train_dataset = NLPDataset(dataset_name="imdb-sentiment", dataset=train_data)
test_dataset = NLPDataset(dataset_name="imdb-sentiment", dataset=test_data)

# Ingest and diagnose
client.ingest(
    dataset_name="imdb-sentiment",
    data_type="nlp",
    train_data=train_dataset,
    test_data=test_dataset,
    batch_size=16,
)

result = client.diagnose_dataset(dataset_name="imdb-sentiment")
print(result.to_text())
```

## 📚 API Reference

### DeepFixClient

Main client class for interacting with the DeepFix server.

#### Constructor

```python
DeepFixClient(
    api_url: str = "http://localhost:8844",
    mlflow_config: Optional[MLflowConfig] = None,
    timeout: int = 30
)
```

**Parameters:**
- `api_url` (str): URL of the DeepFix server. Defaults to `http://localhost:8844`
- `mlflow_config` (MLflowConfig, optional): MLflow configuration for experiment tracking
- `timeout` (int): Request timeout in seconds. Defaults to 30

#### Methods

##### `diagnose_dataset()`

Analyzes a dataset and returns diagnostic results with recommendations.

```python
response = client.diagnose_dataset(
    dataset_name: str,
    language: str = "english"
) -> APIResponse
```

**Parameters:**
- `dataset_name` (str): Name of the dataset to analyze
- `language` (str): Language for analysis output. Defaults to "english"

**Returns:** `APIResponse` object containing analysis results and recommendations

##### `ingest()`

Ingests a dataset with optional quality validation.

```python
client.ingest(
    dataset_name: str,
    data_type: Union[str, DataType],
    train_data: BaseDataset,
    test_data: Optional[BaseDataset] = None,
    train_test_validation: bool = True,
    data_integrity: bool = True,
    batch_size: int = 8,
    overwrite: bool = False
)
```

**Parameters:**
- `dataset_name` (str): Name for the dataset
- `data_type` (str | DataType): Type of data - "image", "tabular", "nlp", or "vision"
- `train_data` (BaseDataset): Training dataset
- `test_data` (BaseDataset, optional): Test/validation dataset
- `train_test_validation` (bool): Enable train/test validation checks. Defaults to True
- `data_integrity` (bool): Enable data integrity checks. Defaults to True
- `batch_size` (int): Batch size for processing. Defaults to 8
- `overwrite` (bool): Overwrite existing dataset. Defaults to False

## 🏗️ Architecture

The SDK consists of several key modules:

- **`client.py`**: Main DeepFixClient class
- **`data/`**: Dataset handling and loading utilities
  - `datasets.py`: Dataset wrapper classes
  - `loader.py`: Data loading pipelines
  - `utils.py`: Data processing utilities
- **`pipelines/`**: Processing pipelines
  - `data_ingestion.py`: Dataset ingestion pipeline
  - `base.py`: Base pipeline class
- **`config.py`**: Configuration management
- **`artifacts/`**: Artifact handling and repository management
- **`zoo/`**: Pre-built datasets and model integrations
  - `datasets/`: Common datasets (food waste, deepchecks datasets)
  - `timm_models.py`: TIMM model integration
  - `trainers/`: Pre-built trainers

## 🔌 Configuration

### MLflow Configuration

Configure MLflow integration for experiment tracking:

```python
from deepfix_sdk.config import MLflowConfig
from deepfix_sdk import DeepFixClient

mlflow_config = MLflowConfig(
    tracking_uri="http://localhost:5000",
    experiment_name="my-ml-experiment",
    run_name="baseline-run",
)

client = DeepFixClient(api_url="http://localhost:8844", mlflow_config=mlflow_config)
```

### Artifact Configuration

```python
from deepfix_sdk.config import ArtifactConfig

artifact_config = ArtifactConfig(
    load_dataset_metadata=True,
    load_checks=True,
    load_model_checkpoint=False,
    load_training=False,
)
```

## 🖥️ Command Line Interface

The SDK includes a CLI for common operations:

```bash
# View available commands
deepfix-sdk --help

# Example commands
deepfix-sdk ingest --dataset-name my-data --data-type image
deepfix-sdk diagnose --dataset-name my-data
```

## 📦 Supported Data Types

- **Image Classification**: Computer vision classification tasks
- **Tabular**: Structured data in DataFrame format
- **NLP**: Natural language processing tasks
- **Vision**: General vision tasks

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_deepchecks_runners.py

# Run with coverage
pytest --cov=deepfix_sdk
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .
```

## 🐛 Troubleshooting

### Connection Issues

**Problem**: Cannot connect to DeepFix server

```python
# Solution: Check server URL and ensure server is running
client = DeepFixClient(api_url="http://your-server-address")
```

### API Key Issues

**Problem**: Authentication fails

```bash
# Solution: Set API key in environment
export DEEPFIX_API_KEY="your-api-key"

# Or in code
import os
os.environ["DEEPFIX_API_KEY"] = "your-api-key"
```

### Timeout Issues

**Problem**: Requests timing out

```python
# Solution: Increase timeout for large datasets
client = DeepFixClient(
    api_url="http://localhost:8844",
    timeout=120,  # Increase to 120 seconds
)
```

## 📖 Documentation

- [DeepFix Main Documentation](../README.md)
- [Server Documentation](../deepfix-server/README.md)
- [API Specification](../specs/service-spec.md)

## 🤝 Contributing

We welcome contributions! Please check out our contributing guidelines.

### Steps to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../LICENSE) file for details.

## 🆘 Support & Contact

- **GitHub Issues**: [Report Issues](https://github.com/delcaux-labs/deepfix/issues)
- **Email**: Contact us at fadel.seydou@delcaux.com
- **Documentation**: [Full Documentation](https://github.com/delcaux-labs/deepfix/docs)

---

**Built with ❤️ by the DeepFix Team**
