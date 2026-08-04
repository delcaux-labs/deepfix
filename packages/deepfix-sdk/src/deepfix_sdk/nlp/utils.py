from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from deepfix_core.models import NLPStatistics, TaskType
from deepfix_core.models.artifacts import (
    LabelStatistics,
    PropertiesStatistics,
    TextStatistics,
)

from ..data.base import BaseDataStatistics
from .dataset import NLPDataset


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
        self._task_type = TaskType.TEXT_CLASSIFICATION

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    def get_train_statistics(self) -> NLPStatistics:
        return self._compute_statistics(self.train_data.dataset)

    def get_test_statistics(self) -> NLPStatistics:
        if self.test_data is None:
            raise ValueError("test_data is None, cannot compute test statistics")
        return self._compute_statistics(self.test_data.dataset)

    def _compute_statistics(self, dataset) -> NLPStatistics:
        from deepchecks.core.errors import DeepchecksNotSupportedError

        num_samples = dataset.n_samples
        task_type = dataset.task_type.value if dataset.task_type else None

        text_stats = self._compute_text_statistics(dataset)
        text_statistics = TextStatistics(**text_stats) if text_stats else None

        label_statistics = None
        if dataset.has_label():
            label_stats = self._compute_label_statistics(dataset)
            if label_stats:
                label_statistics = LabelStatistics(**label_stats)

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

    def _compute_text_statistics(self, dataset) -> Dict[str, Any]:
        text_lengths = [len(text) for text in dataset.text]
        word_counts = [len(text.split()) for text in dataset.text]

        all_words = []
        for text in dataset.text:
            all_words.extend(text.lower().split())
        vocabulary_size = len(set(all_words))

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

    def _compute_label_statistics(self, dataset) -> Dict[str, Any]:
        stats = {}

        if not dataset.has_label():
            return stats

        is_multi_label = dataset.is_multi_label_classification()
        stats["is_multi_label"] = is_multi_label

        if is_multi_label:
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
            if isinstance(dataset.label, np.ndarray):
                unique_labels, counts = np.unique(dataset.label, return_counts=True)
                class_distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
            else:
                from collections import Counter
                class_distribution = dict(Counter(dataset.label))

            stats["class_distribution"] = class_distribution
            stats["num_classes"] = len(class_distribution)

            label_series = pd.Series(list(class_distribution.values()))
            stats["label_distribution_stats"] = label_series.describe().to_dict()

        return stats

    def _compute_properties_statistics(
        self, dataset
    ) -> Optional[Dict[str, Any]]:
        properties = dataset.properties

        if properties is None or len(properties) == 0:
            return None

        feature_statistics = properties.describe().to_dict()
        number_unique_values = properties.nunique().to_dict()
        percentage_unique_values = (
            (properties.nunique() * 100 / len(properties)).round(2).to_dict()
        )

        return {
            "feature_statistics": feature_statistics,
            "number_unique_values": number_unique_values,
            "percentage_unique_values": percentage_unique_values,
        }
