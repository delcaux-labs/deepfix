"""
Timm model wrappers for vision tasks.

Requires the ``[vision]`` extra: ``pip install deepfix-sdk[vision]``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch


def _require_vision():
    try:
        import timm  # noqa: F811
        import torch  # noqa: F811
    except ImportError:
        raise ImportError(
            "Vision dependencies are required for timm models. "
            "Install with: pip install deepfix-sdk[vision]"
        ) from None


def get_timm_model(
    model_name: str, pretrained: bool = True, num_classes: int = 10
) -> "torch.nn.Module":
    import timm
    import torch
    from torchvision import transforms as T

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
        _require_vision()

        import torch.nn as nn

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
        _require_vision()

        from ..utils.feature_extractor import FeatureExtractor

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
        _require_vision()

        import torch
        from open_clip import create_model_from_pretrained, get_tokenizer
        from torchvision import transforms as T

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
        import torch

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
