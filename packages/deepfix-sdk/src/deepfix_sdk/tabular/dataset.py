from typing import List, Optional

import pandas as pd
from deepfix_core.models import DataType

from ..data.base import BaseDataset


class TabularDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset: pd.DataFrame,
        label: Optional[str] = None,
        cat_features: Optional[List[str]] = None,
    ):
        try:
            from deepchecks.tabular import Dataset as DeepchecksTabularDataset
        except ImportError:
            raise ImportError(
                "Tabular dependencies are required for this module. "
                "Install with: pip install deepfix-sdk[tabular]"
            ) from None

        if isinstance(dataset, pd.DataFrame):
            assert label is not None, "Label column is required"
            self.dataset = DeepchecksTabularDataset(
                dataset, label=label, cat_features=cat_features or []
            )

        else:
            raise ValueError(f"Invalid dataset type: {type(dataset)}")

        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.data.iloc[idx], self.dataset.label_col.iloc[idx]

    def __iter__(self):
        return iter(self.dataset)

    def get_data(self) -> pd.DataFrame:
        return self.dataset.data.copy()

    @property
    def data_type(self) -> DataType:
        return DataType.TABULAR

    @property
    def data(self) -> pd.DataFrame:
        return self.get_data()

    @property
    def name(self) -> str:
        return self.dataset_name

    @property
    def X(self) -> pd.DataFrame:
        x = self.get_data().drop(columns=[self.dataset.label_name])
        x[self.cat_features] = x[self.cat_features].astype("category")
        return x

    @property
    def y(self) -> pd.Series:
        return self.dataset.label_col.copy()

    @property
    def cat_features(self) -> List[str]:
        return self.dataset.cat_features

    @property
    def num_features(self) -> List[str]:
        return self.dataset.numerical_features

    def to_loader(self, *args, **kwargs) -> "TabularDataset":
        return self
