from functools import partial
from typing import Callable, Dict, Optional
import numpy as np

try:
    from deepchecks.vision import VisionData
    from deepchecks.vision import BatchOutputFormat
    from torch.utils.data import DataLoader, Dataset
    from supervision.dataset.core import DetectionDataset
except ImportError:
        raise ImportError(
            "Vision dependencies are required for this module. "
            "Install with: pip install deepfix-sdk[vision]"
        ) from None

def classification_collate(data):
    images = np.stack([np.array(x[0]) for x in data])
    labels = [x[1] for x in data]
    return BatchOutputFormat(images=images, labels=labels)


def classification_collate_with_model(data, model):
    images = np.stack([np.array(x[0]) for x in data])
    predictions = model(images)
    assert isinstance(predictions, np.ndarray), "Model must return numpy array"
    labels = [x[1] for x in data]
    return BatchOutputFormat(images=images, labels=labels, predictions=predictions)


def detection_collate_without_model(data):
    
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
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.shape[0] in [1, 3] and image.ndim == 3:
            image = image.transpose(1, 2, 0)
        images.append(image)

        if (
            detections is None
            or getattr(detections, "xyxy", None) is None
            or len(detections.xyxy) == 0
        ):
            labels.append(np.zeros((0, 5), dtype=np.float32))
            continue

        x1y1x2y2 = np.array(detections.xyxy, dtype=np.float32)
        if getattr(detections, "class_id", None) is not None:
            cls = np.array(detections.class_id, dtype=np.float32)
        else:
            cls = np.full((x1y1x2y2.shape[0],), -1.0, dtype=np.float32)

        wh = x1y1x2y2[:, 2:4] - x1y1x2y2[:, 0:2]
        xywh = np.concatenate([x1y1x2y2[:, 0:2], wh], axis=1)
        label = np.concatenate([cls.reshape(-1, 1), xywh], axis=1)
        labels.append(label)

    return BatchOutputFormat(images=images, labels=labels)


def segmentation_collate_without_model(data):
   
    images = []
    labels = []
    for item in data:
        image, mask = item
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.shape[0] in [1, 3] and image.ndim == 3:
            image = image.transpose(1, 2, 0)
            
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask, dtype=np.int64)
            
        images.append(image)
        labels.append(mask)
    return BatchOutputFormat(images=images, labels=labels)


class ClassificationVisionDataLoader:
    
    @classmethod
    def load_from_dataset(
        cls,
        dataset,
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
        cls, dataloader, label_map: Optional[Dict[int, str]] = None
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
    
    @classmethod
    def load_from_dataset(
        cls,
        dataset,
        label_map: Dict[int, str],
        batch_size: int = 8,
        shuffle: bool = True,
    ) -> VisionData:
        from torch.utils.data import DataLoader

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
        cls, dataloader, label_map: Dict[int, str]
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
    

    @classmethod
    def load_from_dataset(
        cls,
        dataset,
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
        cls, dataloader, label_map: Dict[int, str]
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
