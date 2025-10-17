from torch.utils.data import Dataset
from typing import Optional, Callable
import torch

from .base import Step
from ..data import ClassificationVisionDataLoader


class DataIngestor(Step):
    def __init__(
        self,
        batch_size: int = 8,
        model: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.batch_size = batch_size
        self.model = model

    def run(
        self,
        context: dict,
        train_data: Optional[Dataset] = None,
        test_data: Optional[Dataset] = None,
        **kwargs,
    ) -> dict:
        train_data = ClassificationVisionDataLoader.load_from_dataset(
            train_data or context.get("train_data"),
            batch_size=self.batch_size,
            model=self.model or context.get("model"),
        )
        if context.get("test_data") is not None:
            test_data = ClassificationVisionDataLoader.load_from_dataset(
                test_data or context.get("test_data"),
                batch_size=self.batch_size,
                model=self.model or context.get("model"),
            )
        context["train_data"] = train_data
        context["test_data"] = test_data
        return context
