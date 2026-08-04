from typing import Any, Dict, List, Optional
import pandas as pd

from deepfix_core.models import TabularStatistics, TaskType
from ..data.base import BaseDataStatistics
from .dataset import TabularDataset


class TabularDataStatistics(BaseDataStatistics):
    def __init__(
        self,
        train_data: TabularDataset,
        test_data: Optional[TabularDataset] = None,
    ):
        assert isinstance(train_data, TabularDataset), (
            f"train_data must be an instance of {type(TabularDataset)}, got {type(train_data)}"
        )
        if test_data is not None:
            assert isinstance(test_data, TabularDataset), (
                f"test_data must be an instance of {type(TabularDataset)}, got {type(test_data)}"
            )
        super().__init__(train_data=train_data, test_data=test_data)
        self._task_type = (
            TaskType.TABULAR_CLASSIFICATION
            if isinstance(train_data, TabularDataset)
            else TaskType.TABULAR_REGRESSION
        )

    @property
    def task_type(self) -> TaskType:
        return self._task_type

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
        feature_stats = None
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
