"""
Feature extraction for image filtering algorithms.

This module provides feature extraction capabilities for clustering
and filtering algorithms in object detection training data selection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
import timm
import torch.nn as nn
import torchvision.transforms as T

from contextlib import nullcontext


class FeatureExtractor(nn.Module):
    """
    Feature extractor.
    """

    def __init__(
        self,
        model_name: str = "timm/vit_base_patch14_reg4_dinov2.lvd142m",
        freeze: bool = True,
        to_torchscript: bool = False,
    ):
        """
        Initialize the feature extractor.
        Args:
            model_name: timm model name (default: 'timm/vit_small_patch16_224.dino')
            device: Device to run inference on ('cpu', 'cuda',)
        """
        super().__init__()

        self.backbone = model_name
        self.model = None
        self.transform = None
        self.pil_to_tensor = T.PILToTensor()
        self.freeze = freeze

        self._set_model_and_transform()

        self.context = torch.no_grad if self.freeze else nullcontext

        if to_torchscript:
            self.to_torchscript()

        with torch.no_grad():
            self.num_features = self.forward(torch.randn(1, 3, 224, 224)).shape[1]

    def _set_model_and_transform(self) -> str:
        global_pool = "" if "vit" in self.backbone else "avg"
        self.model = timm.create_model(
            self.backbone, pretrained=True, num_classes=0, global_pool=global_pool
        )
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)
        self.transform = nn.Sequential(
            *[t for t in transform.transforms if isinstance(t, (T.Normalize, T.Resize))]
        )

        if self.freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def feature_dim(self) -> int:
        """
        Return the dimension of the extracted feature vector.
        """
        return self.num_features

    def forward(self, images: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Extract features
        """
        images = self._load(images)
        return self._forward(images)

    def _load(self, images: Union[torch.Tensor, List[Image.Image]]):
        if isinstance(images, torch.Tensor):
            images = images.float()
            images = self.transform(images)
        else:
            for image in images:
                assert isinstance(image, Image.Image), (
                    f"Image must be a PIL Image. Received {type(image)}"
                )
            images = torch.stack(
                [self.pil_to_tensor(image.convert("RGB")) for image in images], dim=0
            )
            images = images.float()
            images = self.transform(images)
        return images

    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        with self.context():
            if "vit" in self.backbone:  # get CLS token for ViT models
                return self.model(images)[:, 0, :]
            else:
                return self.model(images)

    def to_torchscript(self) -> None:
        self.model = torch.jit.script(self.model)
