from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np
import pandas as pd
import torch
from deepchecks.nlp import TextData
from deepchecks.core.errors import DeepchecksNotSupportedError
from deepfix_core.models import (
    BaseDatasetStatistics,
    DataType,
    NLPStatistics,
    ObjectDetectionStatistics,
    TabularStatistics,
    TaskType,
    VisionStatistics,
    IRStatistics,
)
from deepfix_core.models.artifacts import (
    LabelStatistics,
    PropertiesStatistics,
    TextStatistics,
)
from tqdm import tqdm

from .datasets import (
    BaseDataset,
    ImageClassificationDataset,
    InformationRetrievalDataset,
    NLPDataset,
    ObjectDetectionDataset,
    SemanticSegmentationDataset,
    TabularDataset,
    VisionDataset,
)


def get_data_statistics(
    data_type: Union[str, DataType],
    train_data: BaseDataset,
    test_data: Optional[BaseDataset] = None,
):
    if data_type == DataType.VISION:
        return VisionDataStatistics(
            train_data=train_data, test_data=test_data
        ).get_statistics()
    elif data_type == DataType.TABULAR:
        return TabularDataStatistics(
            train_data=train_data, test_data=test_data
        ).get_statistics()
    elif data_type == DataType.NLP:
        return NLPDataStatistics(
            train_data=train_data, test_data=test_data
        ).get_statistics()
    elif data_type == DataType.IR:
        return IRDataStatistics(
            train_data=train_data, test_data=test_data
        ).get_statistics()
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


class BaseDataStatistics(Protocol):
    def __init__(
        self, train_data: BaseDataset, test_data: Optional[BaseDataset] = None
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.task_type = self._get_task_type(train_data)
        self.task_type = self._get_task_type(train_data)
        if test_data is not None:
            test_task_type = self._get_task_type(test_data)
            assert self.task_type == test_task_type, (
                f"Task type of train_data and test_data must be the same. Got {self.task_type} and {test_task_type}"
            )

    def _get_task_type(self, dataset: BaseDataset) -> TaskType:
        if isinstance(dataset, ImageClassificationDataset):
            return TaskType.IMAGE_CLASSIFICATION
        elif isinstance(dataset, ObjectDetectionDataset):
            return TaskType.OBJECT_DETECTION
        elif isinstance(dataset, SemanticSegmentationDataset):
            return TaskType.IMAGE_SEGMENTATION
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    def get_statistics(self) -> Dict[str, Any]:
        stats = {"train": self.get_train_statistics()}
        if self.test_data is not None:
            stats["test"] = self.get_test_statistics()
        stats["task_type"] = self.task_type
        return stats

    def get_train_statistics(self) -> BaseDatasetStatistics:
        raise NotImplementedError(
            "get_train_statistics method must be implemented in the subclass"
        )

    def get_test_statistics(self) -> BaseDatasetStatistics:
        raise NotImplementedError(
            "get_test_statistics method must be implemented in the subclass"
        )

    def _compute_statistics(self, dataset: BaseDataset) -> BaseDatasetStatistics:
        raise NotImplementedError(
            "_compute_statistics method must be implemented in the subclass"
        )


class VisionDataStatistics(BaseDataStatistics):
    def __init__(
        self,
        train_data: VisionDataset,
        test_data: Optional[VisionDataset] = None,
    ):
        assert isinstance(train_data, VisionDataset), (
            f"train_data must be an instance of {type(VisionDataset)}, got {type(train_data)}"
        )
        if test_data is not None:
            assert isinstance(test_data, VisionDataset), (
                f"test_data must be an instance of {type(VisionDataset)}, got {type(test_data)}"
            )
        self.train_data = train_data
        self.test_data = test_data
        self.task_type = self._get_task_type(train_data)

    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()
        # stats.update({"t-statistic":self.compute_t_statistics(stats)})
        return stats

    def compute_t_statistics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        if "test_stats" not in stats:
            return {}

        train_color_means = stats["train_stats"]["mean"]
        train_color_stds = stats["train_stats"]["std"]
        train_n = stats["num_train_samples"]

        test_color_means = stats["test_stats"]["mean"]
        test_color_stds = stats["test_stats"]["std"]
        test_n = stats["num_test_samples"]

        t = np.abs(np.array(train_color_means) - np.array(test_color_means)) / np.sqrt(
            np.array(train_color_stds) ** 2 / train_n
            + np.array(test_color_stds) ** 2 / test_n
        )
        t = map(float, t)

        return dict(
            zip([f"color_channel_{i}" for i in range(len(train_color_means))], t)
        )

    def get_train_statistics(
        self,
    ) -> VisionStatistics:
        return self._compute_statistics(self.train_data)

    def get_test_statistics(
        self,
    ) -> VisionStatistics:
        return self._compute_statistics(self.test_data)

    def _compute_statistics(self, dataset: VisionDataset) -> VisionStatistics:
        stats = self._compute_base_statistics(dataset)
        if isinstance(dataset, ObjectDetectionDataset):
            stats.object_detection_statistics = self._compute_box_statistics(dataset)
        return stats

    def _compute_base_statistics(self, dataset: VisionDataset) -> VisionStatistics:
        assert isinstance(dataset, VisionDataset), (
            f"dataset must be an instance of VisionDataset. Received: {type(dataset)}"
        )

        # Initialize accumulators
        num_samples = len(dataset)
        if num_samples == 0:
            return VisionStatistics(num_samples=0)

        # Get first sample to determine number of channels
        first_image = dataset[0]["image"]

        H, W, C = first_image.shape  # C dimension
        assert C in [1, 3], (
            f"Expected image of shape H*W*1 or H*W*3. But got {H}*{W}*{C}."
        )

        # Initialize accumulators for mean and variance computation (use float64 for stability)
        sum_pixels = torch.zeros(C, dtype=torch.float64)
        sum_squared_pixels = torch.zeros(C, dtype=torch.float64)
        count = 0
        class_counts = {}  # Will store count of each class
        pixel_class_ratio = dict()

        # Compute mean
        for idx in tqdm(range(len(dataset)), desc="Computing dataset base statistics"):
            image = dataset[idx]["image"]
            label = dataset[idx]["label"]
            assert image.shape[2] == C, (
                f"Expected image of shape H*W*{C}. But got {image.shape[0]}*{image.shape[1]}*{image.shape[2]}."
            )

            if isinstance(dataset, ImageClassificationDataset):
                if isinstance(label, int):
                    pass
                elif isinstance(label, torch.Tensor):
                    label = int(label.cpu().item())
                else:
                    label = int(label)
                class_counts[label] = class_counts.get(label, 0) + 1
            elif isinstance(dataset, ObjectDetectionDataset):
                class_ids = label.class_id
                for class_id in map(int, class_ids):
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
            elif isinstance(dataset, SemanticSegmentationDataset):
                class_ids = set(label.flatten())
                for class_id in map(int, class_ids):
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                total = max(sum(class_counts.values()), 1)
                pixel_class_ratio = {
                    k: round(v / total, 3) for k, v in class_counts.items()
                }
            else:
                raise ValueError(f"Unsupported dataset type: {type(dataset)}")

            # Ensure image is a tensor and floating type
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            elif isinstance(image, torch.Tensor):
                pass
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            # Cast to float32 and normalize if in 0-255 range
            if image.dtype.is_floating_point:
                image = image.to(torch.float32)
            else:
                image = image.to(torch.float32)
                # Heuristic: if values look like 0-255, scale to 0-1
                if torch.max(image) > 1.0 and torch.min(image) >= 0.0:
                    image = image / 255.0

            # Flatten spatial dimensions (H, W) and keep channel dimension
            image_flat = image.permute(2, 0, 1).reshape(C, -1)  # Shape: (C, H*W)

            # Accumulate sums
            sum_pixels += image_flat.sum(dim=1).to(
                torch.float64
            )  # Sum over spatial dimensions
            count += image_flat.shape[1]  # H*W
            sum_squared_pixels += (image_flat**2).sum(dim=1).to(torch.float64)

        # Compute mean and standard deviation
        mean = sum_pixels / count

        # Compute variance using the formula: Var(X) = E[X^2] - E[X]^2
        variance = (sum_squared_pixels / count) - (mean**2)
        std = torch.sqrt(variance)

        return VisionStatistics(
            image_color_means=mean.tolist(),
            image_color_stds=std.tolist(),
            class_distribution={str(k): v for k, v in class_counts.items()},
            pixel_class_ratio={str(k): float(v) for k, v in pixel_class_ratio.items()},
            num_samples=num_samples,
        )

    def _compute_box_statistics(
        self, dataset: ObjectDetectionDataset
    ) -> ObjectDetectionStatistics:
        """
        Compute statistics on bounding boxes for an object detection dataset.

        Args:
            dataset: ObjectDetectionDataset containing images and annotations

        Returns:
            dict: Dictionary containing various box statistics and negative sample count
        """
        assert isinstance(dataset, ObjectDetectionDataset), (
            f"dataset must be an ObjectDetectionDataset. Received: {type(dataset)}"
        )

        # Initialize accumulators
        num_samples = len(dataset)
        if num_samples == 0:
            return ObjectDetectionStatistics(
                num_negative_samples=0,
                num_positive_samples=0,
                negative_positive_ratio=0,
                num_boxes=0,
            )

        num_negative_samples = 0
        num_boxes_total = 0
        boxes_per_image_list = []
        box_widths = []
        box_heights = []
        box_areas = []

        # Iterate through dataset
        annotations_dict = dataset.get_annotations()
        for annotation in tqdm(
            annotations_dict.values(), desc="Computing base box statistics"
        ):
            # Check if there are any detections
            num_boxes_in_image = len(annotation.xyxy)

            if num_boxes_in_image == 0:
                num_negative_samples += 1
                boxes_per_image_list.append(0)
                continue

            num_boxes_total += num_boxes_in_image
            boxes_per_image_list.append(num_boxes_in_image)

            # Extract box coordinates and class IDs
            boxes = annotation.xyxy  # Shape: (N, 4) with format [x1, y1, x2, y2]

            # Compute box dimensions and areas
            width = boxes[:, 2] - boxes[:, 0]  # width
            heights = boxes[:, 3] - boxes[:, 1]  # height
            areas = width * heights
            box_areas.extend(areas)
            box_widths.extend(width)
            box_heights.extend(heights)

        # Compute statistics using pandas Series and describe()
        boxes_per_image_stats = {}
        box_width_stats = {}
        box_height_stats = {}
        box_area_stats = {}

        if len(boxes_per_image_list) > 0:
            boxes_per_image_series = pd.Series(boxes_per_image_list)
            boxes_per_image_stats = boxes_per_image_series.describe().to_dict()

        if len(box_widths) > 0:
            box_width_series = pd.Series(box_widths)
            box_width_stats = box_width_series.describe().to_dict()

        if len(box_heights) > 0:
            box_height_series = pd.Series(box_heights)
            box_height_stats = box_height_series.describe().to_dict()

        if len(box_areas) > 0:
            box_area_series = pd.Series(box_areas)
            box_area_stats = box_area_series.describe().to_dict()

        negative_positive_ratio = (
            num_negative_samples / (num_samples - num_negative_samples)
            if (num_samples - num_negative_samples > 0)
            else 0
        )

        return ObjectDetectionStatistics(
            num_negative_samples=num_negative_samples,
            num_positive_samples=num_samples - num_negative_samples,
            negative_positive_ratio=negative_positive_ratio,
            num_boxes=num_boxes_total,
            boxes_per_image=boxes_per_image_stats,
            box_width_stats=box_width_stats,
            box_height_stats=box_height_stats,
            box_area_stats=box_area_stats,
        )


class TabularDataStatistics(BaseDataStatistics):
    def __init__(
        self,
        train_data: TabularDataset,
        test_data: Optional[TabularDataset] = None,
    ):
        assert isinstance(train_data, TabularDataset), (
            f"train_data must be an instance of {type(TabularDataset)}, got {type(train_data)}"
        )
        self.train_data = train_data

        if test_data is not None:
            assert isinstance(test_data, TabularDataset), (
                f"test_data must be an instance of {type(TabularDataset)}, got {type(test_data)}"
            )
            self.test_data = test_data
        else:
            self.test_data = None
        self.task_type = (
            TaskType.TABULAR_CLASSIFICATION
            if isinstance(train_data, TabularDataset)
            else TaskType.TABULAR_REGRESSION
        )

    def get_train_statistics(self) -> TabularStatistics:
        return self._compute_statistics(
            self.train_data.get_data(),
            self.train_data.cat_features,
            self.train_data.num_features,
        )

    def get_test_statistics(self) -> TabularStatistics:
        return self._compute_statistics(
            self.test_data.get_data(),
            self.test_data.cat_features,
            self.test_data.num_features,
        )

    def _compute_statistics(
        self,
        dataset: pd.DataFrame,
        categorical_features: List[str],
        numerical_features: List[str],
    ) -> TabularStatistics:
        # TODO: implement feature statistics
        feature_stats = None  # dataset.describe().to_dict()
        number_unique_values = dataset.nunique().to_dict()
        percentage_unique_values = (
            (dataset.nunique() * 100 / len(dataset)).round(2).to_dict()
        )

        return TabularStatistics(
            feature_statistics=feature_stats,
            number_unique_values=number_unique_values,
            percentage_unique_values=percentage_unique_values,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
        )


class NLPDataStatistics(BaseDataStatistics):
    def __init__(
        self,
        train_data: NLPDataset,
        test_data: Optional[NLPDataset] = None,
    ):
        assert isinstance(train_data, NLPDataset), (
            f"train_data must be an instance of {type(NLPDataset)}, got {type(train_data)}"
        )
        self.train_data = train_data

        if test_data is not None:
            assert isinstance(test_data, NLPDataset), (
                f"test_data must be an instance of {type(NLPDataset)}, got {type(test_data)}"
            )
            self.test_data = test_data
        else:
            self.test_data = None
        self.task_type = TaskType.TEXT_CLASSIFICATION

    def get_train_statistics(self) -> NLPStatistics:
        return self._compute_statistics(self.train_data.dataset)

    def get_test_statistics(self) -> NLPStatistics:
        if self.test_data is None:
            raise ValueError("test_data is None, cannot compute test statistics")
        return self._compute_statistics(self.test_data.dataset)

    def _compute_statistics(self, dataset: TextData) -> NLPStatistics:
        """Compute comprehensive statistics for a TextData dataset."""
        # Basic dataset information
        num_samples = dataset.n_samples
        task_type = dataset.task_type.value if dataset.task_type else None

        # Text statistics
        text_stats = self._compute_text_statistics(dataset)
        text_statistics = TextStatistics(**text_stats) if text_stats else None

        # Label statistics (if labels exist)
        label_statistics = None
        if dataset.has_label():
            label_stats = self._compute_label_statistics(dataset)
            if label_stats:
                label_statistics = LabelStatistics(**label_stats)

        # Properties statistics (if properties exist)
        properties_statistics = None
        categorical_properties = []
        numerical_properties = []
        try:
            if dataset.properties is not None:
                props_stats = self._compute_properties_statistics(dataset)
                if props_stats:
                    properties_statistics = PropertiesStatistics(**props_stats)
                categorical_properties = dataset.categorical_properties or []
                numerical_properties = dataset.numerical_properties or []
        except (AttributeError, ValueError, DeepchecksNotSupportedError):
            pass

        # Metadata information (if metadata exists)
        categorical_metadata = []
        numerical_metadata = []
        try:
            if dataset.metadata is not None:
                categorical_metadata = dataset.categorical_metadata or []
                numerical_metadata = dataset.numerical_metadata or []
        except (AttributeError, ValueError, DeepchecksNotSupportedError):
            pass

        return NLPStatistics(
            num_samples=num_samples,
            task_type=task_type,
            text_statistics=text_statistics,
            label_statistics=label_statistics,
            properties_statistics=properties_statistics,
            categorical_properties=categorical_properties
            if categorical_properties
            else None,
            numerical_properties=numerical_properties if numerical_properties else None,
            categorical_metadata=categorical_metadata if categorical_metadata else None,
            numerical_metadata=numerical_metadata if numerical_metadata else None,
        )

    def _compute_text_statistics(self, dataset: TextData) -> Dict[str, Any]:
        """Compute statistics about the text content."""
        text_lengths = [len(text) for text in dataset.text]
        word_counts = [len(text.split()) for text in dataset.text]

        # Vocabulary size (unique words)
        all_words = []
        for text in dataset.text:
            all_words.extend(text.lower().split())
        vocabulary_size = len(set(all_words))

        # Character and word count statistics
        text_length_series = pd.Series(text_lengths)
        word_count_series = pd.Series(word_counts)

        return {
            "character_length": text_length_series.describe().to_dict(),
            "word_count": word_count_series.describe().to_dict(),
            "vocabulary_size": vocabulary_size,
            "avg_chars_per_word": float(np.mean([len(word) for word in all_words]))
            if all_words
            else 0,
        }

    def _compute_label_statistics(self, dataset: TextData) -> Dict[str, Any]:
        """Compute statistics about the labels."""
        stats = {}

        if not dataset.has_label():
            return stats

        # Check if multi-label
        is_multi_label = dataset.is_multi_label_classification()
        stats["is_multi_label"] = is_multi_label

        if is_multi_label:
            # Multi-label case: count labels per sample
            labels_per_sample = [
                sum(label)
                if hasattr(label, "__iter__") and not isinstance(label, str)
                else 1
                for label in dataset.label
            ]
            stats["labels_per_sample"] = (
                pd.Series(labels_per_sample).describe().to_dict()
            )
        else:
            # Single-label case: class distribution
            if isinstance(dataset.label, np.ndarray):
                unique_labels, counts = np.unique(dataset.label, return_counts=True)
                class_distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
            else:
                # Handle list format
                from collections import Counter

                class_distribution = dict(Counter(dataset.label))

            stats["class_distribution"] = class_distribution
            stats["num_classes"] = len(class_distribution)

            # Count label frequency
            label_series = pd.Series(list(class_distribution.values()))
            stats["label_distribution_stats"] = label_series.describe().to_dict()

        return stats

    def _compute_properties_statistics(
        self, dataset: TextData
    ) -> Optional[Dict[str, Any]]:
        """Compute statistics about text properties (similar to TabularDataStatistics)."""
        properties = dataset.properties

        if properties is None or len(properties) == 0:
            return None

        # Compute describe statistics
        feature_statistics = properties.describe().to_dict()

        # Add unique value counts
        number_unique_values = properties.nunique().to_dict()

        # Add percentage of unique values
        percentage_unique_values = (
            (properties.nunique() * 100 / len(properties)).round(2).to_dict()
        )

        return {
            "feature_statistics": feature_statistics,
            "number_unique_values": number_unique_values,
            "percentage_unique_values": percentage_unique_values,
        }


class IRDataStatistics(BaseDataStatistics):
    def __init__(
        self,
        train_data: InformationRetrievalDataset,
        test_data: Optional[InformationRetrievalDataset] = None,
    ):
        assert isinstance(train_data, InformationRetrievalDataset), (
            f"train_data must be an instance of InformationRetrievalDataset, got {type(train_data)}"
        )
        self.train_data = train_data

        if test_data is not None:
            assert isinstance(test_data, InformationRetrievalDataset), (
                f"test_data must be an instance of InformationRetrievalDataset, got {type(test_data)}"
            )
            self.test_data = test_data
        else:
            self.test_data = None
        self.task_type = TaskType.INFORMATION_RETRIEVAL

    def get_train_statistics(self) -> IRStatistics:
        return self._compute_statistics(self.train_data)

    def get_test_statistics(self) -> IRStatistics:
        if self.test_data is None:
            raise ValueError("test_data is None, cannot compute test statistics")
        return self._compute_statistics(self.test_data)

    def _compute_statistics(self, dataset: InformationRetrievalDataset) -> IRStatistics:
        """Compute composite statistics for an IR dataset (NLP + Tabular)."""
        # 1. Get NLP statistics (wrap as NLPDataset since IR no longer inherits from it)
        nlp_dataset = dataset.to_nlp_dataset()
        nlp_data_stats = NLPDataStatistics(train_data=nlp_dataset)
        nlp_stats = nlp_data_stats.get_train_statistics()

        # 2. Get Tabular statistics
        tabular_dataset = dataset.to_tabular()
        tabular_data_stats = TabularDataStatistics(train_data=tabular_dataset)
        tabular_stats = tabular_data_stats.get_train_statistics()

        # 3. Compute IR-specific fields
        num_queries = int(dataset.qrels["query_id"].nunique())
        num_relevant_docs = int(
            dataset.qrels[dataset.qrels["relevance"].astype(int) >= 1][
                "doc_id"
            ].nunique()
        )
        num_non_relevant_docs = int(
            dataset.qrels[dataset.qrels["relevance"].astype(int) < 1][
                "doc_id"
            ].nunique()
        )

        return IRStatistics(
            num_samples=len(dataset),
            num_queries=num_queries,
            num_relevant_docs=num_relevant_docs,
            nlp_statistics=nlp_stats,
            tabular_statistics=tabular_stats,
        )
