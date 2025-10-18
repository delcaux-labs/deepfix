"""
Dataset Registry for managing and discovering available datasets.

This module provides a centralized registry for all available datasets (vision, tabular, custom),
enabling easy discovery and loading of datasets through a unified interface.
"""

from typing import Any, Callable, Dict, List, Optional
import logging
from enum import Enum
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


class DatasetType(Enum):
    """Enumeration of dataset types."""
    VISION_CLASSIFICATION = "vision_classification"
    VISION_DETECTION = "vision_detection"
    VISION_SEGMENTATION = "vision_segmentation"
    TABULAR_CLASSIFICATION = "tabular_classification"
    TABULAR_REGRESSION = "tabular_regression"
    NLP_CLASSIFICATION = "nlp_classification"
    NLP_TOKEN_CLASSIFICATION = "nlp_token_classification"
    CUSTOM = "custom"


@dataclass
class DatasetMetadata:
    """Metadata about a registered dataset."""
    name: str
    display_name: str
    dataset_type: DatasetType
    description: str
    source: str
    required_params: List[str]
    optional_params: Dict[str, Any]
    loader_func: Callable


class DatasetRegistry:
    """Registry for managing dataset loaders."""
    
    def __init__(self):
        """Initialize the registry."""
        self._registry: Dict[str, DatasetMetadata] = {}
        self._populate_registry()
    
    def _populate_registry(self) -> None:
        """Populate the registry with all available datasets."""
        # Vision Classification Datasets
        self._register_vision_classification()
        # Vision Detection Datasets
        self._register_vision_detection()
        # Vision Segmentation Datasets
        self._register_vision_segmentation()
        # Tabular Classification Datasets
        self._register_tabular_classification()
        # Tabular Regression Datasets
        self._register_tabular_regression()
        # NLP Classification Datasets
        self._register_nlp_classification()
        # NLP Token Classification Datasets
        self._register_nlp_token_classification()
    
    def _register_vision_classification(self) -> None:
        """Register vision classification datasets."""
        from .deepchecks_vision import load_mnist_classification
        
        self.register(
            name="mnist_classification",
            display_name="MNIST Classification",
            dataset_type=DatasetType.VISION_CLASSIFICATION,
            description="MNIST handwritten digits dataset for image classification",
            source="DeepChecks",
            loader_func=load_mnist_classification,
            required_params=[],
            optional_params={
                "train": {"default": True, "type": bool, "description": "Load training or test set"},
                "batch_size": {"default": 8, "type": int, "description": "Batch size for data loader"},
                "shuffle": {"default": False, "type": bool, "description": "Whether to shuffle data"},
                "pin_memory": {"default": True, "type": bool, "description": "Pin memory for CUDA"},
                "object_type": {"default": "VisionData", "type": str, "description": "Return type (VisionData or DataLoader)"},
            },
        )
    
    def _register_vision_detection(self) -> None:
        """Register vision detection datasets."""
        from .deepchecks_vision import load_coco_detection, load_mask_detection
        
        self.register(
            name="coco_detection",
            display_name="COCO Object Detection",
            dataset_type=DatasetType.VISION_DETECTION,
            description="COCO (Common Objects in Context) dataset for object detection",
            source="DeepChecks",
            loader_func=load_coco_detection,
            required_params=[],
            optional_params={
                "train": {"default": True, "type": bool, "description": "Load training or validation set"},
                "batch_size": {"default": 8, "type": int, "description": "Batch size for data loader"},
                "shuffle": {"default": False, "type": bool, "description": "Whether to shuffle data"},
                "num_workers": {"default": 0, "type": int, "description": "Number of data loading workers"},
                "pin_memory": {"default": True, "type": bool, "description": "Pin memory for CUDA"},
                "object_type": {"default": "VisionData", "type": str, "description": "Return type (VisionData or DataLoader)"},
            },
        )
        
        self.register(
            name="mask_detection",
            display_name="Face Mask Detection",
            dataset_type=DatasetType.VISION_DETECTION,
            description="Face mask detection dataset with images of people wearing/not wearing masks",
            source="DeepChecks / Kaggle (CC0)",
            loader_func=load_mask_detection,
            required_params=[],
            optional_params={
                "batch_size": {"default": 8, "type": int, "description": "Batch size for data loader"},
                "shuffle": {"default": False, "type": bool, "description": "Whether to shuffle data"},
                "num_workers": {"default": 0, "type": int, "description": "Number of data loading workers"},
                "pin_memory": {"default": True, "type": bool, "description": "Pin memory for CUDA"},
                "object_type": {"default": "VisionData", "type": str, "description": "Return type (VisionData or DataLoader)"},
                "day_index": {"default": 0, "type": int, "description": "Day index (0=training, 1-59=production days)"},
            },
        )
    
    def _register_vision_segmentation(self) -> None:
        """Register vision segmentation datasets."""
        from .deepchecks_vision import load_segmentation_dataset
        
        self.register(
            name="coco_segmentation",
            display_name="COCO Segmentation",
            dataset_type=DatasetType.VISION_SEGMENTATION,
            description="COCO dataset for semantic segmentation tasks",
            source="DeepChecks",
            loader_func=load_segmentation_dataset,
            required_params=[],
            optional_params={
                "dataset_name": {"default": "coco", "type": str, "description": "Segmentation dataset name"},
                "train": {"default": True, "type": bool, "description": "Load training or validation set"},
                "batch_size": {"default": 8, "type": int, "description": "Batch size for data loader"},
                "shuffle": {"default": False, "type": bool, "description": "Whether to shuffle data"},
                "pin_memory": {"default": True, "type": bool, "description": "Pin memory for CUDA"},
                "object_type": {"default": "VisionData", "type": str, "description": "Return type (VisionData or DataLoader)"},
            },
        )
    
    def _register_tabular_classification(self) -> None:
        """Register tabular classification datasets."""
        from .deepchecks_tabular import load_adult_classification, load_iris_classification
        
        self.register(
            name="adult_classification",
            display_name="Adult Income Classification",
            dataset_type=DatasetType.TABULAR_CLASSIFICATION,
            description="Adult/Census income dataset - predict income > $50K",
            source="DeepChecks",
            loader_func=load_adult_classification,
            required_params=[],
            optional_params={},
        )
        
        self.register(
            name="iris_classification",
            display_name="Iris Flower Classification",
            dataset_type=DatasetType.TABULAR_CLASSIFICATION,
            description="Iris dataset - classify iris flower species",
            source="DeepChecks",
            loader_func=load_iris_classification,
            required_params=[],
            optional_params={},
        )
    
    def _register_tabular_regression(self) -> None:
        """Register tabular regression datasets."""
        from .deepchecks_tabular import load_airbnb_regression, load_wine_quality_regression, load_avocado_regression
        
        self.register(
            name="airbnb_regression",
            display_name="Airbnb Price Regression",
            dataset_type=DatasetType.TABULAR_REGRESSION,
            description="Airbnb dataset - predict accommodation prices",
            source="DeepChecks",
            loader_func=load_airbnb_regression,
            required_params=[],
            optional_params={
                "data_size": {"default": 15000, "type": int, "description": "Number of samples to load"},
            },
        )
        
        self.register(
            name="wine_quality_regression",
            display_name="Wine Quality Regression",
            dataset_type=DatasetType.TABULAR_REGRESSION,
            description="Wine quality dataset - predict wine quality scores",
            source="DeepChecks",
            loader_func=load_wine_quality_regression,
            required_params=[],
            optional_params={},
        )
        
        self.register(
            name="avocado_regression",
            display_name="Avocado Price Regression",
            dataset_type=DatasetType.TABULAR_REGRESSION,
            description="Avocado dataset - predict avocado prices",
            source="DeepChecks",
            loader_func=load_avocado_regression,
            required_params=[],
            optional_params={},
        )
    
    def _register_nlp_classification(self) -> None:
        """Register NLP classification datasets."""
        from .deepchecks_nlp import load_tweet_emotion_classification, load_just_dance_comment_classification
        
        self.register(
            name="tweet_emotion_classification",
            display_name="Tweet Emotion Classification",
            dataset_type=DatasetType.NLP_CLASSIFICATION,
            description="Tweet dataset with emotion labels (joy, sadness, anger, fear, neutral) for sentiment analysis",
            source="DeepChecks",
            loader_func=load_tweet_emotion_classification,
            required_params=[],
            optional_params={},
        )
        
        self.register(
            name="just_dance_comment_classification",
            display_name="Just Dance Comment Analysis",
            dataset_type=DatasetType.NLP_CLASSIFICATION,
            description="YouTube comments about Just Dance videos with binary classification labels",
            source="DeepChecks",
            loader_func=load_just_dance_comment_classification,
            required_params=[],
            optional_params={},
        )
    
    def _register_nlp_token_classification(self) -> None:
        """Register NLP token classification datasets."""
        from .deepchecks_nlp import load_scierc_ner
        
        self.register(
            name="scierc_ner",
            display_name="SciERC Named Entity Recognition",
            dataset_type=DatasetType.NLP_TOKEN_CLASSIFICATION,
            description="Scientific paper abstracts with token-level entity annotations (Method, Material, Task, Metric, etc.)",
            source="DeepChecks",
            loader_func=load_scierc_ner,
            required_params=[],
            optional_params={},
        )
    
    def register(
        self,
        name: str,
        display_name: str,
        dataset_type: DatasetType,
        description: str,
        source: str,
        loader_func: Callable,
        required_params: Optional[List[str]] = None,
        optional_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a dataset loader.
        
        Args:
            name: Unique identifier for the dataset
            display_name: Human-readable name
            dataset_type: Type of dataset
            description: Description of the dataset
            source: Source/origin of the dataset
            loader_func: Callable that loads the dataset
            required_params: List of required parameter names
            optional_params: Dict of optional parameters with defaults and metadata
        """
        metadata = DatasetMetadata(
            name=name,
            display_name=display_name,
            dataset_type=dataset_type,
            description=description,
            source=source,
            required_params=required_params or [],
            optional_params=optional_params or {},
            loader_func=loader_func,
        )
        self._registry[name] = metadata
        LOGGER.debug("Registered dataset: %s", name)
    
    def load(self, name: str, **kwargs) -> Any:
        """
        Load a dataset by name.
        
        Args:
            name: Name of the dataset to load
            **kwargs: Arguments to pass to the loader function
        
        Returns:
            The loaded dataset
        
        Raises:
            ValueError: If dataset name is not registered
        """
        if name not in self._registry:
            raise ValueError(f"Dataset '{name}' not found in registry")
        
        metadata = self._registry[name]
        LOGGER.info("Loading dataset: %s", metadata.display_name)
        
        try:
            return metadata.loader_func(**kwargs)
        except Exception as e:
            LOGGER.error("Failed to load dataset '%s': %s", name, str(e))
            raise
    
    def list_datasets(
        self,
        dataset_type: Optional[DatasetType] = None,
        names_only: bool = False,
    ) -> List[str] | List[DatasetMetadata]:
        """
        List all registered datasets, optionally filtered by type.
        
        Args:
            dataset_type: Filter by dataset type (None for all)
            names_only: If True, return only names; if False, return full metadata
        
        Returns:
            List of dataset names or metadata objects
        """
        datasets = list(self._registry.values())
        
        if dataset_type:
            datasets = [d for d in datasets if d.dataset_type == dataset_type]
        
        if names_only:
            return [d.name for d in datasets]
        
        return datasets
    
    def get_metadata(self, name: str) -> DatasetMetadata:
        """
        Get metadata about a dataset.
        
        Args:
            name: Name of the dataset
        
        Returns:
            DatasetMetadata object
        
        Raises:
            ValueError: If dataset name is not registered
        """
        if name not in self._registry:
            raise ValueError(f"Dataset '{name}' not found in registry")
        
        return self._registry[name]
    
    def search(self, query: str) -> List[DatasetMetadata]:
        """
        Search for datasets by name or description.
        
        Args:
            query: Search query (case-insensitive)
        
        Returns:
            List of matching datasets
        """
        query_lower = query.lower()
        results = []
        
        for metadata in self._registry.values():
            if (query_lower in metadata.name.lower() or
                query_lower in metadata.display_name.lower() or
                query_lower in metadata.description.lower()):
                results.append(metadata)
        
        return results
    
    def print_registry(self, dataset_type: Optional[DatasetType] = None) -> None:
        """
        Print a formatted view of the registry.
        
        Args:
            dataset_type: Filter by dataset type (None for all)
        """
        datasets = self.list_datasets(dataset_type=dataset_type)
        
        if not datasets:
            print("No datasets found in registry")
            return
        
        print("\n" + "="*80)
        print("DATASET REGISTRY")
        print("="*80)
        
        for metadata in datasets:
            print(f"\n[{metadata.name}]")
            print(f"  Display Name: {metadata.display_name}")
            print(f"  Type: {metadata.dataset_type.value}")
            print(f"  Source: {metadata.source}")
            print(f"  Description: {metadata.description}")
            
            if metadata.optional_params:
                print("  Optional Parameters:")
                for param_name, param_info in metadata.optional_params.items():
                    default = param_info.get("default", "N/A")
                    desc = param_info.get("description", "")
                    print(f"    - {param_name} (default: {default}): {desc}")


# Global registry instance
_registry: Optional[DatasetRegistry] = None


def get_registry() -> DatasetRegistry:
    """
    Get the global dataset registry instance.
    
    Returns:
        DatasetRegistry instance
    """
    global _registry  # pylint: disable=global-statement
    if _registry is None:
        _registry = DatasetRegistry()
    return _registry
