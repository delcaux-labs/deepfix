from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    Tuple,
)
from functools import cached_property

import re
import string
import numpy as np
import pandas as pd
from deepfix_core.models import DataType
from typing_extensions import runtime_checkable
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

@runtime_checkable
class BaseDataset(Protocol):
    def to_loader(self, model: Optional[Callable] = None, batch_size: int = 8) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def data_type(self) -> DataType:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def name(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")



class BaseDataStatistics(Protocol):
    def __init__(
        self, train_data: BaseDataset, test_data: Optional[BaseDataset] = None
    ):
        self.train_data = train_data
        self.test_data = test_data

    @property
    def task_type(self) -> TaskType:
        raise NotImplementedError("Subclasses must implement this property")

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


