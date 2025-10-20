from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Optional, Any
from typing import Callable

from ..data.loader import ClassificationVisionDataLoader

class BaseDataset(ABC):

    @abstractmethod
    def to_loader(self, model: Optional[Callable] = None, batch_size: int = 8) -> Any:
        pass

class VisionDataset(BaseDataset):
    pass

class ImageClassificationDataset(VisionDataset):

    def __init__(self, dataset_name: str,dataset:Dataset):
        self.dataset = dataset
        self.dataset_name = dataset_name

    def to_loader(self, model: Optional[Callable] = None, batch_size: int = 8) -> ClassificationVisionDataLoader:
        return ClassificationVisionDataLoader.load_from_dataset(
            self.dataset,
            batch_size=batch_size,
            model=model,
        )

class ObjectDetectionDataset(VisionDataset):
    def __init__(self, dataset_name: str,dataset):
        pass


class TabularDataset(BaseDataset):

    def __init__(self, dataset_name: str,dataset):
        pass


    
