from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import OrderedDict
from typing import Tuple, Dict, Optional
from pathlib import Path
import json
import torch

from ...utils.feature_extractor import FeatureExtractor

ing2label = {
    "apple_pie": 0,
    "baby_back_ribs": 1,
    "baklava": 2,
    "beef_carpaccio": 3,
    "beef_tartare": 4,
    "beet_salad": 5,
    "beignets": 6,
    "bibimbap": 7,
    "bread_pudding": 8,
    "breakfast_burrito": 9,
    "bruschetta": 10,
    "caesar_salad": 11,
    "cannoli": 12,
    "caprese_salad": 13,
    "carrot_cake": 14,
    "ceviche": 15,
    "cheesecake": 16,
    "cheese_plate": 17,
    "chicken_curry": 18,
    "chicken_quesadilla": 19,
    "chicken_wings": 20,
    "chocolate_cake": 21,
    "chocolate_mousse": 22,
    "churros": 23,
    "clam_chowder": 24,
    "club_sandwich": 25,
    "crab_cakes": 26,
    "creme_brulee": 27,
    "croque_madame": 28,
    "cup_cakes": 29,
    "deviled_eggs": 30,
    "donuts": 31,
    "dumplings": 32,
    "edamame": 33,
    "eggs_benedict": 34,
    "escargots": 35,
    "falafel": 36,
    "filet_mignon": 37,
    "fish_and_chips": 38,
    "foie_gras": 39,
    "french_fries": 40,
    "french_onion_soup": 41,
    "french_toast": 42,
    "fried_calamari": 43,
    "fried_rice": 44,
    "frozen_yogurt": 45,
    "garlic_bread": 46,
    "gnocchi": 47,
    "greek_salad": 48,
    "grilled_cheese_sandwich": 49,
    "grilled_salmon": 50,
    "guacamole": 51,
    "gyoza": 52,
    "hamburger": 53,
    "hot_and_sour_soup": 54,
    "hot_dog": 55,
    "huevos_rancheros": 56,
    "hummus": 57,
    "ice_cream": 58,
    "lasagna": 59,
    "lobster_bisque": 60,
    "lobster_roll_sandwich": 61,
    "macaroni_and_cheese": 62,
    "macarons": 63,
    "miso_soup": 64,
    "mussels": 65,
    "nachos": 66,
    "omelette": 67,
    "onion_rings": 68,
    "oysters": 69,
    "pad_thai": 70,
    "paella": 71,
    "pancakes": 72,
    "panna_cotta": 73,
    "peking_duck": 74,
    "pho": 75,
    "pizza": 76,
    "pork_chop": 77,
    "poutine": 78,
    "prime_rib": 79,
    "pulled_pork_sandwich": 80,
    "ramen": 81,
    "ravioli": 82,
    "red_velvet_cake": 83,
    "risotto": 84,
    "samosa": 85,
    "sashimi": 86,
    "scallops": 87,
    "seaweed_salad": 88,
    "shrimp_and_grits": 89,
    "spaghetti_bolognese": 90,
    "spaghetti_carbonara": 91,
    "spring_rolls": 92,
    "steak": 93,
    "strawberry_shortcake": 94,
    "sushi": 95,
    "tacos": 96,
    "takoyaki": 97,
    "tiramisu": 98,
    "tuna_tartare": 99,
    "waffles": 100,
}
ing2label = OrderedDict(sorted(ing2label.items(), key=lambda x: x[1]))


class FoodDataset(Dataset):
    def __init__(self, split: str = "train[0:500]", image_size: int = 518):
        self.dataset = load_dataset("ethz/food101", split=split)
        self.transform = T.Compose(
            [
                T.PILToTensor(),
                T.Resize(
                    (image_size, image_size), interpolation=T.InterpolationMode.NEAREST
                ),
            ]
        )
        self.split = split
        self.image_size = image_size
        self.class_to_label_map = {v: k for k, v in ing2label.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        label = self.dataset[idx]["label"]
        return self.transform(image), label

    @property
    def num_classes(self):
        return len(ing2label)

    @property
    def label_to_class_map(self):
        return dict(zip(ing2label.values(), ing2label.keys()))


class FoodDatasetWithEmbeddings(FoodDataset):
    def __init__(
        self,
        embedding_model: Optional[FeatureExtractor] = None,
        split="train[0:500]",
        image_size: int = 1024,
    ):
        super().__init__(split=split, image_size=image_size)
        self.embedding_model = embedding_model
        self.dataset_embeddings = []
        self.dataset_labels = []
        self._embeddings_computed = False

    def compute_embeddings(
        self,
        num_workers: int = 4,
        batch_size: int = 8,
        pin_memory: bool = True,
        device: str = "cpu",
    ):
        self.embedding_model = self.embedding_model.to(device)

        loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for batch_image, batch_label in tqdm(
            loader,
            desc=f"Computing embeddings for {self.split} set",
            total=int(len(self) / batch_size),
        ):
            batch_image = batch_image.to(device)
            batch_embedding = self.embedding_model(batch_image)
            self.dataset_embeddings.append(batch_embedding.to("cpu", non_blocking=True))
            self.dataset_labels.append(batch_label.to("cpu", non_blocking=True))

        self.dataset_embeddings = torch.vstack(self.dataset_embeddings)
        self.dataset_labels = torch.hstack(self.dataset_labels).reshape(-1, 1)
        self._embeddings_computed = True
        return None

    def __getitem__(self, idx):
        if not self._embeddings_computed:
            return super().__getitem__(idx)
        return self.dataset_embeddings[idx], self.dataset_labels[idx]

    def __len__(self):
        if not self._embeddings_computed:
            return super().__len__()
        return len(self.dataset_embeddings)

    def save_embeddings(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        path = Path(path).with_suffix(".pt")
        torch.save(self.dataset_embeddings, path)
        torch.save(self.dataset_labels, path.with_suffix(".labels.pt"))

        with open(path.with_suffix(".metadata.json"), "w") as f:
            metadata = {"split": self.split, "image_size": self.image_size}
            json.dump(metadata, f, indent=2)

    @classmethod
    def from_embeddings(cls, path: str):
        path = Path(path)
        metadata_path = path.with_suffix(".metadata.json")
        embeddings_path = path.with_suffix(".pt")
        labels_path = path.with_suffix(".labels.pt")
        assert embeddings_path.exists(), "Embeddings file does not exist"
        assert labels_path.exists(), "Labels file does not exist"
        assert metadata_path.exists(), "Metadata file does not exist"

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        split = metadata["split"]
        image_size = metadata["image_size"]

        dataset = cls(embedding_model=None, split=split, image_size=image_size)
        dataset.dataset_embeddings = torch.load(
            embeddings_path, map_location="cpu", mmap=True
        )
        dataset.dataset_labels = torch.load(labels_path, map_location="cpu", mmap=True)
        dataset._embeddings_computed = True
        return dataset


def create_train_and_val_datasets(
    split: str = "train[0:500]",
    image_size: int = 518,
    embedding_model: Optional[FeatureExtractor] = None,
    device: str = "cpu",
    num_workers: int = 4,
    batch_size: int = 8,
    pin_memory: bool = False,
) -> Tuple[FoodDataset, FoodDataset]:
    if embedding_model is not None:
        dataset = FoodDatasetWithEmbeddings(
            split=split, image_size=image_size, embedding_model=embedding_model
        )
        dataset.compute_embeddings(
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
        )
        return dataset
    else:
        return FoodDataset(split=split, image_size=image_size)


def load_train_and_val_datasets(
    image_size: int = 518,
    embedding_model: Optional[FeatureExtractor] = None,
    device: str = "cpu",
    num_workers: int = 4,
    split_size: str = ":10%",
    batch_size: int = 8,
    pin_memory: bool = False,
) -> Tuple[FoodDataset, FoodDataset]:
    cfg = {
        "image_size": image_size,
        "embedding_model": embedding_model,
        "device": device,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "pin_memory": pin_memory,
    }
    train_dataset = create_train_and_val_datasets(split=f"train[{split_size}]", **cfg)
    val_dataset = create_train_and_val_datasets(
        split=f"validation[{split_size}]", **cfg
    )
    return train_dataset, val_dataset
