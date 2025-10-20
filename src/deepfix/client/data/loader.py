from deepchecks.vision import VisionData, BatchOutputFormat
from torch.utils.data import DataLoader, Dataset
import torch
from typing import Dict, Optional, Callable
from functools import partial


def classification_collate(data) -> BatchOutputFormat:
    images = torch.stack([x[0] for x in data])
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # (N, C, H, W) into (N, H, W, C)
    labels = [x[1] for x in data]
    return BatchOutputFormat(images=images, labels=labels)


def classification_collate_with_model(
    data, model: Callable[[torch.Tensor], torch.Tensor]
) -> BatchOutputFormat:
    images = torch.stack([x[0] for x in data])
    with torch.inference_mode():
        predictions = model(images)
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # (N, C, H, W) into (N, H, W, C)
    labels = [x[1] for x in data]
    return BatchOutputFormat(images=images, labels=labels, predictions=predictions)


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
