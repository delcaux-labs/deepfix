from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from deepfix_core.models import (
    ObjectDetectionStatistics,
    TaskType,
    VisionStatistics,
)
from tqdm import tqdm

from ..data.base import BaseDataStatistics
from .dataset import (
    ImageClassificationDataset,
    ObjectDetectionDataset,
    SemanticSegmentationDataset,
    VisionDataset,
)


class VisionDataStatistics(BaseDataStatistics):
    def __init__(
        self,
        train_data: VisionDataset,
        test_data: Optional[VisionDataset] = None,
    ):
        assert isinstance(train_data, VisionDataset), (
            f"train_data must be an instance of {type(VisionDataset)}, got {type(train_data)}"
        )
        if test_data is not None:
            assert isinstance(test_data, VisionDataset), (
                f"test_data must be an instance of {type(VisionDataset)}, got {type(test_data)}"
            )
        super().__init__(train_data=train_data, test_data=test_data)
        self._task_type = self._get_task_type(train_data)

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    def _get_task_type(self, dataset: VisionDataset) -> TaskType:
        if isinstance(dataset, ImageClassificationDataset):
            return TaskType.IMAGE_CLASSIFICATION
        elif isinstance(dataset, ObjectDetectionDataset):
            return TaskType.OBJECT_DETECTION
        elif isinstance(dataset, SemanticSegmentationDataset):
            return TaskType.IMAGE_SEGMENTATION
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()
        return stats

    def compute_t_statistics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        if "test_stats" not in stats:
            return {}

        train_color_means = stats["train_stats"]["mean"]
        train_color_stds = stats["train_stats"]["std"]
        train_n = stats["num_train_samples"]

        test_color_means = stats["test_stats"]["mean"]
        test_color_stds = stats["test_stats"]["std"]
        test_n = stats["num_test_samples"]

        t = np.abs(np.array(train_color_means) - np.array(test_color_means)) / np.sqrt(
            np.array(train_color_stds) ** 2 / train_n
            + np.array(test_color_stds) ** 2 / test_n
        )
        t = map(float, t)

        return dict(
            zip([f"color_channel_{i}" for i in range(len(train_color_means))], t)
        )

    def get_train_statistics(
        self,
    ) -> VisionStatistics:
        return self._compute_statistics(self.train_data)

    def get_test_statistics(
        self,
    ) -> VisionStatistics:
        return self._compute_statistics(self.test_data)

    def _compute_statistics(self, dataset: VisionDataset) -> VisionStatistics:
        stats = self._compute_base_statistics(dataset)
        if isinstance(dataset, ObjectDetectionDataset):
            stats.object_detection_statistics = self._compute_box_statistics(dataset)
        return stats

    def _compute_base_statistics(self, dataset: VisionDataset) -> VisionStatistics:
        import torch

        assert isinstance(dataset, VisionDataset), (
            f"dataset must be an instance of VisionDataset. Received: {type(dataset)}"
        )

        num_samples = len(dataset)
        if num_samples == 0:
            return VisionStatistics(num_samples=0)

        first_image = dataset[0]["image"]

        H, W, C = first_image.shape
        assert C in [1, 3], (
            f"Expected image of shape H*W*1 or H*W*3. But got {H}*{W}*{C}."
        )

        sum_pixels = torch.zeros(C, dtype=torch.float64)
        sum_squared_pixels = torch.zeros(C, dtype=torch.float64)
        count = 0
        class_counts = {}
        pixel_class_ratio = dict()

        for idx in tqdm(range(len(dataset)), desc="Computing dataset base statistics"):
            image = dataset[idx]["image"]
            label = dataset[idx]["label"]
            assert image.shape[2] == C, (
                f"Expected image of shape H*W*{C}. But got {image.shape[0]}*{image.shape[1]}*{image.shape[2]}."
            )

            if isinstance(dataset, ImageClassificationDataset):
                if isinstance(label, int):
                    pass
                elif isinstance(label, torch.Tensor):
                    label = int(label.cpu().item())
                else:
                    label = int(label)
                class_counts[label] = class_counts.get(label, 0) + 1
            elif isinstance(dataset, ObjectDetectionDataset):
                class_ids = label.class_id
                for class_id in map(int, class_ids):
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
            elif isinstance(dataset, SemanticSegmentationDataset):
                class_ids = set(label.flatten())
                for class_id in map(int, class_ids):
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                total = max(sum(class_counts.values()), 1)
                pixel_class_ratio = {
                    k: round(v / total, 3) for k, v in class_counts.items()
                }
            else:
                raise ValueError(f"Unsupported dataset type: {type(dataset)}")

            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            elif isinstance(image, torch.Tensor):
                pass
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            if image.dtype.is_floating_point:
                image = image.to(torch.float32)
            else:
                image = image.to(torch.float32)
                if torch.max(image) > 1.0 and torch.min(image) >= 0.0:
                    image = image / 255.0

            image_flat = image.permute(2, 0, 1).reshape(C, -1)

            sum_pixels += image_flat.sum(dim=1).to(
                torch.float64
            )
            count += image_flat.shape[1]
            sum_squared_pixels += (image_flat**2).sum(dim=1).to(torch.float64)

        mean = sum_pixels / count
        variance = (sum_squared_pixels / count) - (mean**2)
        std = torch.sqrt(variance)

        return VisionStatistics(
            image_color_means=mean.tolist(),
            image_color_stds=std.tolist(),
            class_distribution={str(k): v for k, v in class_counts.items()},
            pixel_class_ratio={str(k): float(v) for k, v in pixel_class_ratio.items()},
            num_samples=num_samples,
        )

    def _compute_box_statistics(
        self, dataset: ObjectDetectionDataset
    ) -> ObjectDetectionStatistics:
        assert isinstance(dataset, ObjectDetectionDataset), (
            f"dataset must be an ObjectDetectionDataset. Received: {type(dataset)}"
        )

        num_samples = len(dataset)
        if num_samples == 0:
            return ObjectDetectionStatistics(
                num_negative_samples=0,
                num_positive_samples=0,
                negative_positive_ratio=0,
                num_boxes=0,
            )

        num_negative_samples = 0
        num_boxes_total = 0
        boxes_per_image_list = []
        box_widths = []
        box_heights = []
        box_areas = []

        annotations_dict = dataset.get_annotations()
        for annotation in tqdm(
            annotations_dict.values(), desc="Computing base box statistics"
        ):
            num_boxes_in_image = len(annotation.xyxy)

            if num_boxes_in_image == 0:
                num_negative_samples += 1
                boxes_per_image_list.append(0)
                continue

            num_boxes_total += num_boxes_in_image
            boxes_per_image_list.append(num_boxes_in_image)

            boxes = annotation.xyxy

            width = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            areas = width * heights
            box_areas.extend(areas)
            box_widths.extend(width)
            box_heights.extend(heights)

        boxes_per_image_stats = {}
        box_width_stats = {}
        box_height_stats = {}
        box_area_stats = {}

        if len(boxes_per_image_list) > 0:
            boxes_per_image_series = pd.Series(boxes_per_image_list)
            boxes_per_image_stats = boxes_per_image_series.describe().to_dict()

        if len(box_widths) > 0:
            box_width_series = pd.Series(box_widths)
            box_width_stats = box_width_series.describe().to_dict()

        if len(box_heights) > 0:
            box_height_series = pd.Series(box_heights)
            box_height_stats = box_height_series.describe().to_dict()

        if len(box_areas) > 0:
            box_area_series = pd.Series(box_areas)
            box_area_stats = box_area_series.describe().to_dict()

        negative_positive_ratio = (
            num_negative_samples / (num_samples - num_negative_samples)
            if (num_samples - num_negative_samples > 0)
            else 0
        )

        return ObjectDetectionStatistics(
            num_negative_samples=num_negative_samples,
            num_positive_samples=num_samples - num_negative_samples,
            negative_positive_ratio=negative_positive_ratio,
            num_boxes=num_boxes_total,
            boxes_per_image=boxes_per_image_stats,
            box_width_stats=box_width_stats,
            box_height_stats=box_height_stats,
            box_area_stats=box_area_stats,
        )
