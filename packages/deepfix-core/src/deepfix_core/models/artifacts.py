from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator

from .defaults import DataType, TaskType


# Artifacts models
class ArtifactPath(StrEnum):
    """Enumeration of artifact path identifiers.

    Defines standard paths for different types of artifacts used in the system,
    including training artifacts, deepchecks results, and dataset metadata.
    """

    # training artifacts
    TRAINING = "training_artifacts"
    TRAINING_METRICS = "metrics.csv"
    MODEL_CHECKPOINT = "model_checkpoint"
    TRAINING_PARAMS = "params.yaml"
    # deepchecks artifacts
    DEEPCHECKS = "deepchecks"
    # dataset artifacts
    DATASET = "dataset"


class DeepchecksConfig(BaseModel):
    """Configuration for Deepchecks validation suites.

    Attributes:
        train_test_validation: Whether to run the train-test validation suite.
        data_integrity: Whether to run the data integrity suite.
        model_evaluation: Whether to run the model evaluation suite.
        max_samples: Optional maximum number of samples to use for validation.
        random_state: Random seed for reproducibility. Defaults to 42.
        save_results: Whether to save validation results to disk.
        output_dir: Optional directory to save results.
        batch_size: Batch size for processing. Defaults to 16.
        data_type: Type of data (vision, tabular, nlp). Defaults to VISION.
    """

    train_test_validation: bool = Field(
        default=True, description="Whether to run the train_test_validation suite"
    )
    data_integrity: bool = Field(
        default=True, description="Whether to run the data_integrity suite"
    )
    model_evaluation: bool = Field(
        default=False, description="Whether to run the model_evaluation suite"
    )
    max_samples: Optional[int] = Field(
        default=None, description="Maximum number of samples to run the suites on"
    )
    random_state: int = Field(
        default=42, description="Random seed to use for the suites"
    )
    save_results: bool = Field(default=False, description="Whether to save the results")
    output_dir: Optional[str] = Field(
        default=None, description="Output directory to save the results"
    )
    batch_size: int = Field(default=16, description="Batch size to use for the suites")
    data_type: DataType = Field(
        default=DataType.VISION, description="Type of data to run the suites on"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary with data_type converted to its string value.
        """
        dumped_dict = self.model_dump()
        dumped_dict["data_type"] = self.data_type.value
        return dumped_dict

    @classmethod
    def from_dict(cls, config: Union[Dict[str, Any], DictConfig]) -> "DeepchecksConfig":
        """Create DeepchecksConfig from a dictionary.

        Args:
            config: Dictionary or DictConfig containing configuration.

        Returns:
            DeepchecksConfig instance.
        """
        return cls(**config)

    @classmethod
    def from_file(cls, file_path: str) -> "DeepchecksConfig":
        """Load DeepchecksConfig from a file.

        Args:
            file_path: Path to the configuration file (YAML/OmegaConf format).

        Returns:
            DeepchecksConfig instance loaded from file.

        Raises:
            FileNotFoundError: If the file does not exist.
            OmegaConfException: If the file contains invalid configuration.
        """
        return cls.from_dict(OmegaConf.load(file_path))


## Deepchecks
class DeepchecksResultHeaders(StrEnum):
    """Enumeration of Deepchecks result header names.

    Defines standard header names for Deepchecks validation and integrity checks,
    including train-test validation checks and data integrity checks.
    """

    # Train-Test Validation
    LabelDrift = "Label Drift"
    ImageDatasetDrift = "Image Dataset Drift"
    ImagePropertyDrift = "Image Property Drift"
    PropertyLabelCorrelationChange = "Property Label Correlation Change"
    HeatmapComparison = "Heatmap Comparison"
    NewLabels = "New Labels"
    # Data Integrity
    ImagePropertyOutliers = "Image Property Outliers"
    PropertyLabelCorrelation = "Property Label Correlation"
    LabelPropertyOutliers = "Label Property Outliers"
    ClassPerformance = "Class Performance"


class DeepchecksConditionResult(BaseModel):
    """Result of a Deepchecks condition evaluation.

    Attributes:
        status: Status of the condition (e.g., "pass", "fail", "warning").
        condition: Description of the condition that was evaluated.
        more_info: Additional information about the condition result.
    """

    status: str = Field(description="Status of the condition")
    condition: str = Field(description="Condition of the condition")
    more_info: str = Field(description="More info of the condition")


class DeepchecksCheckResult(BaseModel):
    """Result of a Deepchecks validation check.

    Attributes:
        check: Name of the check that was performed.
        params: Parameters used for the check.
        summary: Summary description of the check result.
        value: Result value of the check (can be numeric, dict, or list).
        conditions_results: List of condition evaluation results.
        link_in_summary: Optional link to additional information.
        display_text: Human-readable text for display.
        display_images: Base64-encoded images for visualization.
    """

    check: Optional[str] = Field(default=None, description="Name of the check")
    params: Optional[dict] = Field(default=None, description="Parameters of the check")
    summary: Optional[str] = Field(default=None, description="Summary of the check")
    value: Optional[Any] = Field(default=None, description="Value of the check")
    conditions_results: List[DeepchecksConditionResult] = Field(
        default=[], description="Conditions results of the check"
    )
    link_in_summary: Optional[str] = Field(
        default=None, description="Link in summary of the check"
    )
    display_text: Optional[str] = Field(
        default=None, description="Display text of the check"
    )
    display_images: Optional[List[str]] = Field(
        default=None,
        description="Display images of the result as base64 encoded strings",
    )

    @field_validator("value", mode="before")
    @classmethod
    def convert_nan_infinity_to_string(cls, v):
        """Convert NaN and Infinity values to string representations.

        Recursively processes values to convert float NaN and Infinity values
        to string representations for JSON serialization.

        Args:
            v: Value to convert, can be float, dict, list, or other types.

        Returns:
            Converted value with NaN/Infinity as strings, or original value if not applicable.
        """
        if v is None:
            return v

        if isinstance(v, float):
            if math.isnan(v):
                return "NaN"
            elif math.isinf(v):
                return "Infinity" if v > 0 else "-Infinity"
        elif isinstance(v, dict):
            return {k: cls.convert_nan_infinity_to_string(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [cls.convert_nan_infinity_to_string(item) for item in v]

        return v

    def to_dict(self, exclude: list[str] = []) -> dict:
        """Convert the check result to a dictionary.

        Args:
            exclude: List of keys to exclude from the output. None values are
                automatically excluded.

        Returns:
            Dictionary representation of the check result with None values removed.
        """
        dumped = self.model_dump()
        keys_to_remove = set(exclude + [k for k, v in dumped.items() if v is None])
        for key in keys_to_remove:
            dumped.pop(key)
        return dumped


class DeepchecksParsedResult(BaseModel):
    """Parsed Deepchecks result with header and check result.

    Attributes:
        header: Header/title of the result.
        result: The check result containing all validation information.
    """

    header: str = Field(description="Header of the result")
    result: DeepchecksCheckResult = Field(description="Result of the check")

    def to_dict(self, exclude_images: bool = False) -> Dict[str, Any]:
        """Convert the parsed result to a dictionary.

        Args:
            exclude_images: If True, exclude display_images from the output.

        Returns:
            Dictionary representation of the parsed result.
        """
        dumped_dict = self.model_dump()
        if exclude_images:
            dumped_dict.pop("display_images")
        return dumped_dict

    @classmethod
    def from_dict(
        cls, d: Union[Dict[str, Any], DictConfig]
    ) -> "DeepchecksParsedResult":
        """Create a DeepchecksParsedResult from a dictionary.

        Args:
            d: Dictionary or DictConfig containing the parsed result data.

        Returns:
            DeepchecksParsedResult instance created from the dictionary.
        """
        return cls(**d)


class Artifacts(BaseModel):
    """Base class for all artifact types.

    Provides common functionality for serialization and deserialization
    of artifacts from dictionaries and YAML files.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert the artifact to a dictionary.

        Returns:
            Dictionary representation of the artifact.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict):
        """Create an artifact instance from a dictionary.

        Args:
            d: Dictionary containing artifact data.

        Returns:
            Artifact instance created from the dictionary.
        """
        return cls(**d)

    @classmethod
    def from_file(cls, path: str) -> "Artifacts":
        """Load an artifact from a YAML file.

        Args:
            path: Path to the YAML file containing artifact data.

        Returns:
            Artifact instance loaded from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


class DeepchecksArtifacts(Artifacts):
    """Artifacts containing Deepchecks validation results.

    Attributes:
        dataset_name: Name of the dataset that was validated.
        model_name: Optional name of the model used in validation.
        results: Dictionary mapping check categories to lists of parsed results.
        config: Optional Deepchecks configuration used for validation.
    """

    dataset_name: str = Field(description="Name of the dataset")
    model_name: Optional[str] = Field(default=None, description="Name of the model")
    results: Dict[str, List[DeepchecksParsedResult]] = Field(
        description="Results of the artifact"
    )
    config: Optional[DeepchecksConfig] = Field(default=None, description="Config of the artifact")

    def to_dict(self) -> Dict[str, Any]:
        """Convert Deepchecks artifacts to a dictionary.

        Returns:
            Dictionary representation with nested results and config serialized.
        """
        dumped_dict = self.model_dump()
        dumped_dict["results"] = {
            k: [r.to_dict() for r in v] for k, v in self.results.items()
        }
        dumped_dict["config"] = self.config.to_dict() if self.config else None
        return dumped_dict

    @classmethod
    def from_dict(self, d: Union[Dict[str, Any], DictConfig]) -> "DeepchecksArtifacts":
        """Create DeepchecksArtifacts from a dictionary.

        Args:
            d: Dictionary or DictConfig containing artifact data.

        Returns:
            DeepchecksArtifacts instance with parsed results and config.
        """
        results = {
            k: [DeepchecksParsedResult.from_dict(r) for r in v]
            for k, v in d["results"].items()
        }
        config = None
        if d.get("config"):
            config = DeepchecksConfig.from_dict(d["config"])
        return DeepchecksArtifacts(
            dataset_name=d["dataset_name"],
            model_name=d.get("model_name"),
            results=results,
            config=config,
        )

    @classmethod
    def from_file(cls, file_path: str) -> "DeepchecksArtifacts":
        """Load Deepchecks artifacts from a YAML file.

        Args:
            file_path: Path to the YAML file containing artifact data.

        Returns:
            DeepchecksArtifacts instance loaded from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        artifacts = cls.from_dict(d)
        return artifacts


class ModelCheckpointArtifacts(Artifacts):
    """Artifacts containing model checkpoint information.

    Attributes:
        path: Path to the model checkpoint file.
        config: Optional model configuration dictionary.
        model_type: Optional type or class name of the model.
        hyperparameters: Optional dictionary of model hyperparameters.
        context: Optional context information about the checkpoint.
    """

    path: Optional[str] = Field(
        default=None, description="Path to the model checkpoint"
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None, description="Config of the model"
    )
    model_type: Optional[str] = Field(
        default=None, description="Type/class name of the model"
    )
    hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Model hyperparameters"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Context of the model checkpoint"
    )


## Training Artifacts
class TrainingArtifacts(Artifacts):
    """Artifacts containing training metrics and parameters.

    Attributes:
        metrics_path: Optional path to a CSV file containing metrics.
        metrics_values: Optional pandas DataFrame or dictionary containing metrics.
        params: Optional dictionary of training parameters.
    """

    model_config = {"arbitrary_types_allowed": True}

    metrics_path: Optional[str] = Field(
        default=None, description="Path to the metrics file"
    )
    metrics_values: Optional[Any] = Field(
        default=None, description="Metrics of the artifact"
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Parameters of the training routine"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert training artifacts to a dictionary.

        Converts pandas DataFrame metrics to dictionary format for serialization.

        Returns:
            Dictionary representation with metrics as lists.
        """
        dumped_dict = self.model_dump()
        if (
            isinstance(self.metrics_values, pd.DataFrame)
            and self.metrics_values is not None
        ):
            dumped_dict["metrics_values"] = self.metrics_values.to_dict(orient="list")
        return dumped_dict

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingArtifacts":
        """Create TrainingArtifacts from a dictionary.

        Args:
            d: Dictionary containing artifact data. If metrics_values is present,
                it's converted to a DataFrame. If metrics_path is present, the CSV
                is loaded.

        Returns:
            TrainingArtifacts instance with loaded metrics and parameters.
        """
        if d.get("metrics_values"):
            metrics_values = pd.DataFrame.from_dict(d.get("metrics_values"))
        elif d.get("metrics_path"):
            metrics_values = pd.read_csv(d.get("metrics_path"))
        else:
            metrics_values = None
        return cls(
            metrics_path=d.get("metrics_path"),
            metrics_values=metrics_values,
            params=d.get("params"),
        )

    @classmethod
    def from_file(cls, metrics_path: str) -> "TrainingArtifacts":
        """Load training artifacts from a metrics CSV file.

        Args:
            metrics_path: Path to the CSV file containing training metrics.

        Returns:
            TrainingArtifacts instance with metrics loaded from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            pd.errors.EmptyDataError: If the CSV file is empty.
        """
        return cls(metrics_path=metrics_path, metrics_values=pd.read_csv(metrics_path))


class BaseDatasetStatistics(BaseModel):
    """Base class for dataset statistics.

    Provides common functionality for converting statistics to dictionaries
    with None values removed.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to a dictionary.

        Returns:
            Dictionary representation with None values removed.
        """
        dumped_dict = self.model_dump()
        for k in list(dumped_dict.keys()):
            if dumped_dict[k] is None:
                dumped_dict.pop(k)
        return dumped_dict

    @classmethod
    def from_dict(cls, d: dict) -> "BaseDatasetStatistics":
        raise NotImplementedError(
            "from_dict is not implemented for BaseDatasetStatistics"
        )


class ObjectDetectionStatistics(BaseDatasetStatistics):
    """Statistics specific to object detection datasets.

    Attributes:
        num_negative_samples: Number of negative samples (images without objects).
        num_positive_samples: Number of positive samples (images with objects).
        negative_positive_ratio: Ratio of negative to positive samples.
        num_boxes: Total number of bounding boxes in the dataset.
        boxes_per_image: Optional statistics about boxes per image distribution.
        box_width_stats: Optional statistics about box widths.
        box_height_stats: Optional statistics about box heights.
        box_area_stats: Optional statistics about box areas.
    """

    num_negative_samples: int = Field(
        description="Number of negative samples in the dataset"
    )
    num_positive_samples: int = Field(
        description="Number of positive samples in the dataset"
    )
    negative_positive_ratio: float = Field(
        description="Ratio of negative to positive samples"
    )
    num_boxes: int = Field(description="Number of boxes in the dataset")
    boxes_per_image: Optional[Dict[str, Any]] = Field(
        default=None, description="Boxes per image statistics"
    )
    box_width_stats: Optional[Dict[str, Any]] = Field(
        default=None, description="Box width statistics"
    )
    box_height_stats: Optional[Dict[str, Any]] = Field(
        default=None, description="Box height statistics"
    )
    box_area_stats: Optional[Dict[str, Any]] = Field(
        default=None, description="Box area statistics"
    )

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectDetectionStatistics":
        return cls(**d)


class VisionStatistics(BaseDatasetStatistics):
    """Statistics for vision/image datasets.

    Attributes:
        num_samples: Total number of samples in the dataset.
        image_color_means: Optional mean values for each color channel (RGB).
        image_color_stds: Optional standard deviations for each color channel.
        class_distribution: Optional class distribution for classification tasks.
        pixel_class_ratio: Optional pixel class ratios for segmentation tasks.
        object_detection_statistics: Optional object detection specific statistics.
        other_statistics: Optional additional custom statistics.
    """

    num_samples: int = Field(description="Number of samples in the dataset")
    image_color_means: Optional[List[float]] = Field(
        default=None, description="Mean of the image color channels"
    )
    image_color_stds: Optional[List[float]] = Field(
        default=None, description="Standard deviation of the image color channels"
    )
    class_distribution: Optional[Dict[str, int]] = Field(
        default=None,
        description="Class distribution of target variable for classification tasks",
    )
    pixel_class_ratio: Optional[Dict[str, float]] = Field(
        default=None, description="Pixel class ratio for semantic segmentation"
    )
    object_detection_statistics: Optional[ObjectDetectionStatistics] = Field(
        default=None, description="Object detection statistics of the dataset"
    )

    other_statistics: Optional[Dict[str, Any]] = Field(
        default=None, description="Other statistics of the dataset"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert vision statistics to a dictionary.

        Returns:
            Dictionary representation with nested object detection statistics
            serialized and None values removed.
        """
        dumped_dict = self.model_dump()
        if self.object_detection_statistics is not None:
            dumped_dict["object_detection_statistics"] = (
                self.object_detection_statistics.to_dict()
            )
        for k in list(dumped_dict.keys()):
            if dumped_dict[k] is None:
                dumped_dict.pop(k)
        return dumped_dict

    @classmethod
    def from_dict(cls, d: dict) -> "VisionStatistics":
        if "object_detection_statistics" in d:
            d["object_detection_statistics"] = ObjectDetectionStatistics.from_dict(
                d["object_detection_statistics"]
            )
        return cls(**d)


class TabularStatistics(BaseDatasetStatistics):
    """Statistics for tabular datasets.

    Attributes:
        feature_statistics: Optional nested dictionary mapping feature names to
            statistical summaries (count, mean, std, min, 25%, 50%, 75%, max).
        number_unique_values: Optional dictionary mapping feature names to unique
            value counts.
        percentage_unique_values: Optional dictionary mapping feature names to
            percentage of unique values.
        categorical_features: Optional list of categorical feature names.
        numerical_features: Optional list of numerical feature names.
    """

    # Feature statistics from describe() - nested dict: feature_name -> {stat_name -> value}
    feature_statistics: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Statistical summary for each feature (count, mean, std, min, 25%, 50%, 75%, max)",
    )
    number_unique_values: Optional[Dict[str, int]] = Field(
        default=None, description="Number of unique values per feature"
    )
    percentage_unique_values: Optional[Dict[str, float]] = Field(
        default=None, description="Percentage of unique values per feature"
    )
    categorical_features: Optional[List[str]] = Field(
        default=None, description="List of categorical feature names"
    )
    numerical_features: Optional[List[str]] = Field(
        default=None, description="List of numerical feature names"
    )

    @classmethod
    def from_dict(cls, d: dict) -> "TabularStatistics":
        return cls(**d)


class TextStatistics(BaseModel):
    """Statistics about text content.

    Attributes:
        character_length: Optional statistical summary of character lengths
            (count, mean, std, min, 25%, 50%, 75%, max).
        word_count: Optional statistical summary of word counts
            (count, mean, std, min, 25%, 50%, 75%, max).
        vocabulary_size: Optional total number of unique words in the dataset.
        avg_chars_per_word: Optional average number of characters per word.
    """

    character_length: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Statistical summary of character lengths (count, mean, std, min, 25%, 50%, 75%, max)",
    )
    word_count: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Statistical summary of word counts (count, mean, std, min, 25%, 50%, 75%, max)",
    )
    vocabulary_size: Optional[int] = Field(
        default=None, description="Total number of unique words in the dataset"
    )
    avg_chars_per_word: Optional[float] = Field(
        default=None, description="Average number of characters per word"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert text statistics to a dictionary.

        Returns:
            Dictionary representation with None values removed.
        """
        dumped_dict = self.model_dump()
        for k in list(dumped_dict.keys()):
            if dumped_dict[k] is None:
                dumped_dict.pop(k)
        return dumped_dict

    @classmethod
    def from_dict(cls, d: dict) -> "TextStatistics":
        return cls(**d)


class LabelStatistics(BaseModel):
    """Statistics about labels.

    Attributes:
        is_multi_label: Optional flag indicating if the task is multi-label classification.
        labels_per_sample: Optional statistical summary of labels per sample for
            multi-label tasks (count, mean, std, min, 25%, 50%, 75%, max).
        class_distribution: Optional distribution of classes for single-label tasks
            (class name -> count).
        num_classes: Optional number of classes in the dataset.
        label_distribution_stats: Optional statistical summary of label distribution
            (count, mean, std, min, 25%, 50%, 75%, max).
    """

    is_multi_label: Optional[bool] = Field(
        default=None, description="Whether the task is multi-label classification"
    )
    labels_per_sample: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Statistical summary of labels per sample for multi-label tasks (count, mean, std, min, 25%, 50%, 75%, max)",
    )
    class_distribution: Optional[Dict[str, int]] = Field(
        default=None,
        description="Distribution of classes for single-label tasks (class -> count)",
    )
    num_classes: Optional[int] = Field(
        default=None, description="Number of classes in the dataset"
    )
    label_distribution_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Statistical summary of label distribution (count, mean, std, min, 25%, 50%, 75%, max)",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert label statistics to a dictionary.

        Returns:
            Dictionary representation with None values removed.
        """
        dumped_dict = self.model_dump()
        for k in list(dumped_dict.keys()):
            if dumped_dict[k] is None:
                dumped_dict.pop(k)
        return dumped_dict

    @classmethod
    def from_dict(cls, d: dict) -> "LabelStatistics":
        return cls(**d)


class PropertiesStatistics(BaseModel):
    """Statistics about text properties (similar to TabularStatistics).

    Attributes:
        feature_statistics: Optional nested dictionary mapping property names to
            statistical summaries (count, mean, std, min, 25%, 50%, 75%, max).
        number_unique_values: Optional dictionary mapping property names to unique
            value counts.
        percentage_unique_values: Optional dictionary mapping property names to
            percentage of unique values.
    """

    feature_statistics: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Statistical summary for each property (count, mean, std, min, 25%, 50%, 75%, max)",
    )
    number_unique_values: Optional[Dict[str, int]] = Field(
        default=None, description="Number of unique values per property"
    )
    percentage_unique_values: Optional[Dict[str, float]] = Field(
        default=None, description="Percentage of unique values per property"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert properties statistics to a dictionary.

        Returns:
            Dictionary representation with None values removed.
        """
        dumped_dict = self.model_dump()
        for k in list(dumped_dict.keys()):
            if dumped_dict[k] is None:
                dumped_dict.pop(k)
        return dumped_dict

    @classmethod
    def from_dict(cls, d: dict) -> "PropertiesStatistics":
        return cls(**d)


class NLPStatistics(BaseDatasetStatistics):
    """Statistics for NLP/text datasets.

    Attributes:
        num_samples: Total number of samples in the dataset.
        task_type: Optional type of NLP task (e.g., text_classification,
            text_token_classification).
        text_statistics: Optional statistics about text content.
        label_statistics: Optional statistics about labels.
        properties_statistics: Optional statistics about text properties.
        categorical_properties: Optional list of categorical property names.
        numerical_properties: Optional list of numerical property names.
        categorical_metadata: Optional list of categorical metadata column names.
        numerical_metadata: Optional list of numerical metadata column names.
    """

    num_samples: int = Field(description="Number of samples in the dataset")
    task_type: Optional[str] = Field(
        default=None,
        description="Type of NLP task (e.g., text_classification, text_token_classification)",
    )
    text_statistics: Optional[TextStatistics] = Field(
        default=None, description="Statistics about the text content"
    )
    label_statistics: Optional[LabelStatistics] = Field(
        default=None, description="Statistics about the labels"
    )
    properties_statistics: Optional[PropertiesStatistics] = Field(
        default=None, description="Statistics about text properties"
    )
    categorical_properties: Optional[List[str]] = Field(
        default=None, description="List of categorical property names"
    )
    numerical_properties: Optional[List[str]] = Field(
        default=None, description="List of numerical property names"
    )
    categorical_metadata: Optional[List[str]] = Field(
        default=None, description="List of categorical metadata column names"
    )
    numerical_metadata: Optional[List[str]] = Field(
        default=None, description="List of numerical metadata column names"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert NLP statistics to a dictionary.

        Returns:
            Dictionary representation with nested statistics serialized and
            None values removed.
        """
        dumped_dict = self.model_dump()
        if self.text_statistics is not None:
            dumped_dict["text_statistics"] = self.text_statistics.to_dict()
        if self.label_statistics is not None:
            dumped_dict["label_statistics"] = self.label_statistics.to_dict()
        if self.properties_statistics is not None:
            dumped_dict["properties_statistics"] = self.properties_statistics.to_dict()
        for k in list(dumped_dict.keys()):
            if dumped_dict[k] is None:
                dumped_dict.pop(k)
        return dumped_dict

    @classmethod
    def from_dict(cls, d: dict) -> "NLPStatistics":
        if "text_statistics" in d:
            d["text_statistics"] = TextStatistics.from_dict(d["text_statistics"])
        if "label_statistics" in d:
            d["label_statistics"] = LabelStatistics.from_dict(d["label_statistics"])
        if "properties_statistics" in d:
            d["properties_statistics"] = PropertiesStatistics.from_dict(
                d["properties_statistics"]
            )
        return cls(**d)


class IRStatistics(BaseDatasetStatistics):
    """Statistics specific to Information Retrieval (IR) datasets.

    Composite of NLP and Tabular statistics, plus IR-specific metrics.
    """

    num_samples: int = Field(..., description="Total number of samples")
    num_queries: Optional[int] = Field(
        default=None, description="Total number of unique queries"
    )
    num_relevant_docs: Optional[int] = Field(
        default=None, description="Total number of relevant documents"
    )
    num_non_relevant_docs: Optional[int] = Field(
        default=None, description="Total number of non-relevant documents"
    )
    nlp_statistics: Optional[NLPStatistics] = Field(
        default=None, description="NLP-specific statistics"
    )
    tabular_statistics: Optional[TabularStatistics] = Field(
        default=None, description="Tabular-specific statistics"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert IR statistics to a dictionary."""
        dumped_dict = self.model_dump()
        if self.nlp_statistics:
            dumped_dict["nlp_statistics"] = self.nlp_statistics.to_dict()
        if self.tabular_statistics:
            dumped_dict["tabular_statistics"] = self.tabular_statistics.to_dict()
        # Remove None values for cleaner output
        return {k: v for k, v in dumped_dict.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "IRStatistics":
        """Create IRStatistics from a dictionary."""
        if "nlp_statistics" in d and d["nlp_statistics"]:
            d["nlp_statistics"] = NLPStatistics.from_dict(d["nlp_statistics"])
        if "tabular_statistics" in d and d["tabular_statistics"]:
            d["tabular_statistics"] = TabularStatistics.from_dict(
                d["tabular_statistics"]
            )
        return cls(**d)


## Dataset
class DatasetArtifacts(Artifacts):
    """Artifacts containing dataset statistics and metadata.

    Attributes:
        dataset_name: Name of the dataset.
        train_statistics: Training dataset statistics (can be any BaseDatasetStatistics
            subclass or a dictionary).
        task_type: Type of ML task (classification, regression, etc.).
        test_statistics: Optional test/validation dataset statistics.
    """

    dataset_name: str = Field(..., description="Name of the dataset")
    train_statistics: BaseDatasetStatistics = Field(
        ..., description="Train statistics of the dataset"
    )
    task_type: TaskType = Field(..., description="Task type of the dataset")
    test_statistics: Optional[BaseDatasetStatistics] = Field(
        default=None, description="Test statistics of the dataset"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset artifacts to a dictionary.

        Returns:
            Dictionary representation with statistics serialized and task_type
            converted to its string value.
        """
        dumped_dict = self.model_dump()
        dumped_dict["train_statistics"] = self.train_statistics.to_dict()
        if self.test_statistics is not None:
            dumped_dict["test_statistics"] = self.test_statistics.to_dict()
        dumped_dict["task_type"] = self.task_type.value
        return dumped_dict

    @classmethod
    def from_file(cls, path: str) -> "DatasetArtifacts":
        """Load dataset artifacts from a YAML file.

        Args:
            path: Path to the YAML file containing artifact data.

        Returns:
            DatasetArtifacts instance loaded from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        # print("d", d)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetArtifacts":
        task_type = TaskType(d["task_type"])
        d["task_type"] = task_type

        def load_statistics(d: dict) -> BaseDatasetStatistics:
            if task_type in [
                TaskType.OBJECT_DETECTION,
                TaskType.IMAGE_CLASSIFICATION,
                TaskType.IMAGE_SEGMENTATION,
            ]:
                return VisionStatistics.from_dict(d)
            elif task_type in [
                TaskType.TABULAR_CLASSIFICATION,
                TaskType.TABULAR_REGRESSION,
            ]:
                return TabularStatistics.from_dict(d)
            elif task_type in [
                TaskType.TEXT_CLASSIFICATION,
                TaskType.TEXT_TOKEN_CLASSIFICATION,
            ]:
                return NLPStatistics.from_dict(d)
            elif task_type == TaskType.INFORMATION_RETRIEVAL:
                return IRStatistics.from_dict(d)
            else:
                raise ValueError(f"Invalid task type: {task_type.value}")

        if "train_statistics" in d:
            d["train_statistics"] = load_statistics(d["train_statistics"])
        if "test_statistics" in d and d["test_statistics"] is not None:
            d["test_statistics"] = load_statistics(d["test_statistics"])

        return cls(**d)


IRStatistics.model_rebuild()
DatasetArtifacts.model_rebuild()
