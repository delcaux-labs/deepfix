from typing import Optional, Callable
from ..data.datasets import BaseDataset
from .base import Step

class DataIngestor(Step):
    def __init__(
        self,
        batch_size: int = 8,
        model: Optional[Callable] = None,
    ):
        self.batch_size = batch_size
        self.model = model

    def run(
        self,
        context: dict,
        train_data: Optional[BaseDataset] = None,
        test_data: Optional[BaseDataset] = None,
        **kwargs,
    ) -> dict:

        if train_data is None:
            train_data = context.get("train_data")
        if test_data is None:
            test_data = context.get("test_data")

        if train_data is None:
            raise ValueError("train_data is required")
        train_data = train_data.to_loader(model=self.model, batch_size=self.batch_size)

        if test_data is not None:
            test_data = test_data.to_loader(model=self.model, batch_size=self.batch_size)
        
        context["train_data"] = train_data
        context["test_data"] = test_data
        return context
