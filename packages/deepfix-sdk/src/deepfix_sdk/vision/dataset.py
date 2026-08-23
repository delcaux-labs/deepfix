from typing import Callable, Dict, Iterable, Optional, Union

import numpy as np
from deepchecks.vision import VisionData
from deepfix_core.models import DataType
from supervision.dataset.core import DetectionDataset

from ..data.base import BaseDataset
from .loader import (
    ClassificationVisionDataLoader,
    DetectionVisionDataLoader,
    SegmentationVisionDataLoader,
)


class VisionDataset(BaseDataset):
    def __init__(self, dataset_name: str, dataset: VisionData | Iterable):
        self.dataset = dataset
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raise NotImplementedError("should be implemented by subclass")

    def __iter__(self):
        return iter(self.dataset)

    @property
    def data_type(self) -> DataType:
        return DataType.VISION

    def to_loader(self, **kwargs) -> VisionData:
        raise NotImplementedError("should be implemented by subclass")

    @property
    def name(self) -> str:
        return self.dataset_name


class ImageClassificationDataset(VisionDataset):
    def __init__(self, dataset_name: str, dataset: VisionData | Iterable):
        super().__init__(dataset_name=dataset_name, dataset=dataset)

    def to_loader(
        self, model: Optional[Callable] = None, batch_size: int = 8
    ) -> VisionData:

        if isinstance(self.dataset, VisionData):
            return self.dataset

        return ClassificationVisionDataLoader.load_from_dataset(
            self.dataset,
            batch_size=batch_size,
            model=model,
        )

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return dict(image=image, label=label)


class ObjectDetectionDataset(VisionDataset):
    def __init__(self, dataset_name: str, dataset: VisionData | Iterable):
        super().__init__(dataset_name=dataset_name, dataset=dataset)

    @classmethod
    def from_coco(
        cls,
        dataset_name: str,
        images_directory_path: str,
        annotations_path: str,
        force_masks: bool = False,
    ):

        data = DetectionDataset.from_coco(
            images_directory_path=images_directory_path,
            annotations_path=annotations_path,
            force_masks=force_masks,
        )
        return cls(dataset_name=dataset_name, dataset=data)

    @classmethod
    def from_yolo(
        cls,
        dataset_name: str,
        images_directory_path: str,
        data_yaml_path: str,
        annotations_directory_path: str,
        is_obb: bool = False,
        force_masks: bool = False,
    ):
        data = DetectionDataset.from_yolo(
            images_directory_path=images_directory_path,
            data_yaml_path=data_yaml_path,
            annotations_directory_path=annotations_directory_path,
            is_obb=is_obb,
            force_masks=force_masks,
        )
        return cls(dataset_name=dataset_name, dataset=data)

    def get_label_map(self) -> Dict[int, str]:
        labels = list(range(len(self.dataset.classes)))
        return dict(zip(labels, self.dataset.classes))

    def get_annotations(self):

        return self.dataset.annotations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, image, annotation = self.dataset[idx]
        return dict(image_path=image_path, image=image, label=annotation)

    def __iter__(self):
        return iter(self.dataset)

    def to_loader(
        self, batch_size: int = 8, shuffle: bool = False, **kwargs
    ) -> VisionData:
        if isinstance(self.dataset, VisionData):
            return self.dataset
        return DetectionVisionDataLoader.load_from_dataset(
            self.dataset,
            label_map=self.get_label_map(),
            batch_size=batch_size,
            shuffle=shuffle,
        )


class SemanticSegmentationDataset(VisionDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset: VisionData | Iterable,
        label_map: Optional[Dict[int, str]] = None,
    ):
        super().__init__(dataset_name=dataset_name, dataset=dataset)
        self.label_map = label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Union[np.ndarray, np.ndarray]]:
        image, annotation = self.dataset[idx]
        assert isinstance(image,np.ndarray), f'image must be numpy array but got {type(image)}'
        assert isinstance(annotation,np.ndarray), f'annotation must be numpy array but got {type(annotation)}'
        c = image.shape[0]
        if c in [1, 3]:
            image = image.transpose(1, 2, 0)  # (c,h,w) -> (h,w,c)
        return dict(image=image, label=annotation)

    def __iter__(self):
        return iter(self.dataset)

    def get_label_map(self) -> Dict[int, str]:
        if self.label_map is None:
            return {i: f"class_{i}" for i in range(len(self.dataset))}
        self.label_map = self._build_label_map()
        return self.label_map

    def _build_label_map(self) -> Dict[int, str]:
        label_map = set()
        for idx in range(self.__len__()):
            label = self.dataset[idx]["label"]
            assert isinstance(label,np.ndarray), f'annotation must be numpy array but got {type(label)}'
            label_map = label_map.union(set(label.flatten()))
        return {int(i): f"class_{i}" for i in label_map}

    def to_loader(
        self,
        model: Optional[Callable] = None,
        batch_size: int = 8,
        shuffle: bool = False,
    ) -> VisionData:

        if isinstance(self.dataset, VisionData):
            return self.dataset
        else:
            return SegmentationVisionDataLoader.load_from_dataset(
                self.dataset,
                label_map=self.get_label_map(),
                batch_size=batch_size,
                shuffle=shuffle,
            )

