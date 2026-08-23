from typing import Sequence

import numpy as np
from deepfix_core.models import DataType

from ..data.base import BaseDataset


class NLPDataset(BaseDataset):
    def __init__(self, dataset_name: str, dataset):
        self.dataset = dataset
        self.dataset_name = dataset_name

    def to_loader(self, *args, **kwargs) -> "NLPDataset":
        """Compute properties of the dataset and return it."""
        self.dataset.calculate_builtin_properties()
        return self

    def __len__(self):
        return len(self.dataset)

    @property
    def data_type(self) -> DataType:
        return DataType.NLP

    @property
    def data(self):
        return self.dataset

    @property
    def embeddings(self) -> np.ndarray:
        return self.dataset.embeddings

    @property
    def X(self) -> Sequence[str]:
        return self.dataset.text

    @property
    def y(self):
        return self.dataset.label

    @property
    def name(self) -> str:
        return self.dataset_name
