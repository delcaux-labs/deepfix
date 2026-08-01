from functools import partial
from typing import Callable, Dict, Optional

import torch
from deepchecks.vision import BatchOutputFormat, VisionData
from supervision.dataset.core import DetectionDataset
from torch.utils.data import DataLoader, Dataset


def classification_collate(data) -> BatchOutputFormat:
    images = torch.stack([torch.Tensor(x[0]) for x in data])
    images = images.cpu().numpy()
    labels = [x[1] for x in data]
    return BatchOutputFormat(images=images, labels=labels)


def classification_collate_with_model(
    data, model: Callable[[torch.Tensor], torch.Tensor]
) -> BatchOutputFormat:
    images = torch.stack([torch.Tensor(x[0]) for x in data])
    with torch.inference_mode():
        predictions = model(images)
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
    images = images.cpu().numpy()
    labels = [x[1] for x in data]
    return BatchOutputFormat(images=images, labels=labels, predictions=predictions)


def detection_collate_without_model(data) -> BatchOutputFormat:
    images = []
    labels = []
    for item in data:
        # item may be (path, image, detections) or (image, detections)
        if len(item) == 3:
            _, image, detections = item
        elif len(item) == 2:
            image, detections = item
        else:
            raise ValueError(f"Invalid item length: {len(item)}")

        # Ensure images are numpy arrays in HWC as expected by deepchecks
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        images.append(image)

        if (
            detections is None
            or getattr(detections, "xyxy", None) is None
            or len(detections.xyxy) == 0
        ):
            labels.append(torch.zeros((0, 5), dtype=torch.float32))
            continue

        x1y1x2y2 = torch.from_numpy(detections.xyxy).float()
        if getattr(detections, "class_id", None) is not None:
            cls = torch.from_numpy(detections.class_id.astype("float32"))
        else:
            cls = torch.full((x1y1x2y2.shape[0],), -1.0, dtype=torch.float32)

        wh = x1y1x2y2[:, 2:4] - x1y1x2y2[:, 0:2]
        xywh = torch.cat([x1y1x2y2[:, 0:2], wh], dim=1)
        label = torch.cat([cls.view(-1, 1), xywh], dim=1)
        labels.append(label)

    return BatchOutputFormat(images=images, labels=labels)


def segmentation_collate_without_model(data) -> BatchOutputFormat:
    images = []
    labels = []
    for item in data:
        image, mask = item
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()  # HWC to CHW
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().long().numpy()
        images.append(image)
        labels.append(mask)
    return BatchOutputFormat(images=images, labels=labels)


class ClassificationVisionDataLoader:
    def __init__(
        self,
    ):
        pass

    @classmethod
    def load_from_dataset(
        cls,
        dataset: Dataset,
        batch_size: int = 8,
        shuffle: bool = True,
        model: Optional[Callable] = None,
    ) -> VisionData:
        assert isinstance(dataset, Dataset), (
            "dataset must be an instance of torch.utils.data.Dataset. Received: {}".format(
                type(dataset)
            )
        )
        collate_fn = (
            partial(classification_collate_with_model, model=model)
            if model
            else classification_collate
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )
        return cls.load_from_dataloader(dataloader)

    @classmethod
    def load_from_dataloader(
        cls, dataloader: DataLoader, label_map: Optional[Dict[int, str]] = None
    ) -> VisionData:
        assert isinstance(dataloader, DataLoader), (
            "dataloader must be an instance of torch.utils.data.DataLoader. Received: {}".format(
                type(dataloader)
            )
        )
        vision_data = VisionData(
            dataloader, task_type="classification", label_map=label_map
        )
        vision_data.head()
        return vision_data


class DetectionVisionDataLoader:
    def __init__(
        self,
    ):
        pass

    @classmethod
    def load_from_dataset(
        cls,
        dataset: DetectionDataset,
        label_map: Dict[int, str],
        batch_size: int = 8,
        shuffle: bool = True,
    ) -> VisionData:
        assert isinstance(dataset, DetectionDataset), (
            "dataset must be an instance of supervision.dataset.core.DetectionDataset. Received: {}".format(
                type(dataset)
            )
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=detection_collate_without_model,
        )

        return cls.load_from_dataloader(dataloader, label_map=label_map)

    @classmethod
    def load_from_dataloader(
        cls, dataloader: DataLoader, label_map: Dict[int, str]
    ) -> VisionData:
        assert isinstance(dataloader, DataLoader), (
            "dataloader must be an instance of torch.utils.data.DataLoader. Received: {}".format(
                type(dataloader)
            )
        )
        vision_data = VisionData(
            dataloader, task_type="object_detection", label_map=label_map
        )
        vision_data.head()
        return vision_data


class SegmentationVisionDataLoader:
    def __init__(
        self,
    ):
        pass

    @classmethod
    def load_from_dataset(
        cls,
        dataset: Dataset,
        label_map: Dict[int, str],
        batch_size: int = 8,
        shuffle: bool = False,
    ) -> VisionData:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=segmentation_collate_without_model,
        )

        return cls.load_from_dataloader(dataloader, label_map=label_map)

    @classmethod
    def load_from_dataloader(
        cls, dataloader: DataLoader, label_map: Dict[int, str]
    ) -> VisionData:
        assert isinstance(dataloader, DataLoader), (
            "dataloader must be an instance of torch.utils.data.DataLoader. Received: {}".format(
                type(dataloader)
            )
        )
        vision_data = VisionData(
            dataloader, task_type="semantic_segmentation", label_map=label_map
        )
        vision_data.head()
        return vision_data
