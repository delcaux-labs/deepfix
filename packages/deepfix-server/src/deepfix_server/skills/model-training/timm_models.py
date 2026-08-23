"""
Timm model wrappers for vision tasks.

Requires the ``[vision]`` extra: ``pip install deepfix-sdk[vision]``
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Union

import timm
import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from torchvision import transforms as T


def get_timm_model(
    model_name: str, pretrained: bool = True, num_classes: int = 10
) -> "torch.nn.Module":

    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model))
    trfs = [t for t in transform.transforms if isinstance(t, (T.Normalize, T.Resize))]
    model = torch.nn.Sequential(*trfs, model)
    return model


class ClassifierHead:
    """MLP classification head on top of a feature extractor."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):

        layers = []
        if num_layers > 1:
            for i in range(num_layers - 1):
                if i == 0:
                    layers.append(
                        nn.Sequential(
                            nn.Linear(input_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                        )
                    )
                else:
                    layers.append(
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(p=dropout),
                        )
                    )
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.fc = nn.Sequential(*layers)
        elif num_layers == 1:
            self.fc = nn.Linear(input_dim, num_classes)
        else:
            raise ValueError(
                f"Invalid number of layers: {num_layers}. Must be greater than 0."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class FeatureExtractor:
    """
    Feature extractor backed by a timm model.

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

    def forward(self, images: Union[torch.Tensor, List[Image]]) -> torch.Tensor:
        """
        Extract features
        """
        images = self._load(images)
        return self._forward(images)

    def _load(self, images: Union[torch.Tensor, List[Image]]):

        if isinstance(images, torch.Tensor):
            images = images.float()
            images = self.transform(images)
        else:
            for image in images:
                assert isinstance(image, Image), (
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


class TimmClassificationModel:
    """Timm-backed classification model with a trainable MLP head."""

    def __init__(
        self,
        model_name: str = "timm/vit_base_patch14_reg4_dinov2.lvd142m",
        num_classes: int = 10,
        freeze_backbone: bool = True,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
    ):
        """
        Timm classification model.
        """

        self.backbone = FeatureExtractor(
            model_name, freeze=freeze_backbone, to_torchscript=False
        )
        self.mlp = ClassifierHead(
            self.backbone.feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.backbone(x))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class CLIPModel:
    """CLIP model wrapper for zero-shot classification."""

    def __init__(
        self, timm_model_name: str, labels_list: List[str], device: str = "cpu"
    ):
        self.model, self.preprocess = create_model_from_pretrained(
            f"hf-hub:timm/{timm_model_name}"
        )
        self.tokenizer = get_tokenizer(f"hf-hub:timm/{timm_model_name}")
        self.labels_list = labels_list
        self.text = self.tokenizer(
            self.labels_list, context_length=self.model.context_length
        )
        self.transforms = [
            t
            for t in self.preprocess.transforms
            if isinstance(t, (T.Normalize, T.Resize))
        ]
        self.preprocess = torch.nn.Sequential(*self.transforms)
        self.device = device

        self.model.to(self.device)
        self.text.to(self.device)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.float().to(self.device)
        image = self.preprocess(x).to(self.device)
        with torch.inference_mode():
            image_features = self.model.encode_image(image, normalize=True)
            text_features = self.model.encode_text(self.text, normalize=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        text_probs = text_probs.cpu()
        return text_probs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        probs = self.forward(x)
        return probs.argmax(dim=-1)
