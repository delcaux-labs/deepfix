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

eng = [
    "dressing_portion",
    "sauerkraut",
    "iceberg lettuce",
    "bell pepper",
    "potato pancakes",
    "red cabbage",
    "zucchini",
    "mushrooms",
    "lentils",
    "apple sauce",
    "cream",
    "green beans",
    "cauliflower",
    "peas",
    "carrots",
    "rice",
    "potato cubes",
    "potatoes",
    "brown sauce",
    "light sauce",
    "savoy cabbage",
    "chicken",
    "chicken strips",
    "gravy",
    "onion",
    "bread dumplings",
    "roast beef",
    "roast pork neck",
    "ham_mettwurst",
    "grilled sausage",
    "schnitzel",
    "vegetable cream",
    "pollock",
    "egg noodles",
    "mashed potatoes",
    "meatballs",
    "breaded fish fillet",
    "lentil stew",
    "tomato curry sauce",
    "malt beer mustard sauce",
    "coleslaw",
]

german = [
    "dressing_portion",
    "sauerkraut",
    "eisbergsalat",
    "paprika",
    "reibekuchen",
    "rotkohl",
    "zucchini",
    "pilze",
    "linsen",
    "apfelmus",
    "sahne",
    "grüne_bohnen",
    "blumenkohl",
    "erbsen",
    "möhre",
    "reis",
    "kartoffelwürfel",
    "kartoffeln",
    "braune_sauce",
    "helle_sauce",
    "wirsing",
    "hähnchen",
    "hähnchenstreifen",
    "bratenjus",
    "zwiebel",
    "semmelknödel",
    "rinderbraten",
    "schweinenackenbraten",
    "schinken_mettwurst",
    "rostbratwurst",
    "schnitzel",
    "pflanzencreme",
    "seelachs",
    "eierspätzle",
    "kartoffelpüree",
    "fleischbällchen_gebrüht",
    "paniertes_fischfilet",
    "linseneintopf",
    "tomaten-curry-sauce",
    "malzbier-senf-sauce",
    "krautsalat",
]

translations_de_en = dict(zip(german, eng))


class FoodWasteDataset(Dataset):
    def __init__(
        self,
        ing2label: OrderedDict,
        ing2name: OrderedDict,
        split="train",
        image_size: int = 1024,
    ):
        self.dataset = load_dataset("AI-ServicesBB/food-waste-dataset")[split]
        self.transform = T.Compose(
            [
                T.PILToTensor(),
                T.Resize(
                    (image_size, image_size), interpolation=T.InterpolationMode.NEAREST
                ),
            ]
        )
        self.ing2label = ing2label
        self.ing2name = ing2name
        self.split = split
        self.image_size = image_size

    @property
    def num_classes(self):
        return len(self.ing2label)

    @property
    def label_to_class_map(self):
        assert list(self.ing2label.keys()) == list(self.ing2name.keys()), (
            "labels ordering and names ordering must be the same"
        )
        return dict(zip(self.ing2label.values(), self.ing2name.values()))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(sample["image"])
        ing_ids = sample["Artikelnummer"]
        weights = sample["Menge_Rückläufer"]
        i = np.argmax(weights)
        ing_id = ing_ids[i]
        label = self.ing2label[ing_id]
        return image, label


class FoodWasteDatasetWithEmbeddings(FoodWasteDataset):
    def __init__(
        self,
        ing2label: OrderedDict,
        ing2name: OrderedDict,
        embedding_model: Optional[FeatureExtractor] = None,
        split="train",
        image_size: int = 1024,
    ):
        super().__init__(
            ing2label=ing2label, ing2name=ing2name, split=split, image_size=image_size
        )
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
            metadata = {
                "ing2label": self.ing2label,
                "ing2name": self.ing2name,
                "split": self.split,
                "image_size": self.image_size,
            }
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

        ing2label = metadata["ing2label"]
        ing2name = metadata["ing2name"]
        split = metadata["split"]
        image_size = metadata["image_size"]

        dataset = cls(
            embedding_model=None,
            ing2label=ing2label,
            ing2name=ing2name,
            split=split,
            image_size=image_size,
        )
        dataset.dataset_embeddings = torch.load(
            embeddings_path, map_location="cpu", mmap=True
        )
        dataset.dataset_labels = torch.load(labels_path, map_location="cpu", mmap=True)
        dataset._embeddings_computed = True
        return dataset


def get_label_mapping(as_eng: bool = True) -> Tuple[Dict[int, str], Dict[int, int]]:
    ing2name = {}
    dataset = load_dataset("AI-ServicesBB/food-waste-dataset")
    for d in tqdm(
        chain(dataset["train"], dataset["test"]),
        desc="Getting label mapping",
        total=len(dataset["train"]) + len(dataset["test"]),
    ):
        for n, ing in zip(d["Artikelnummer"], d["Artikel"]):
            if n not in ing2name:
                ing2name[n] = str(ing).lower().strip().replace(" ", "_")
    if as_eng:
        ing2name = {k: translations_de_en[v] for k, v in ing2name.items()}
    ing2name = OrderedDict(sorted(ing2name.items()))
    ing2label = OrderedDict({k: i for i, k in enumerate(ing2name.keys())})
    return ing2name, ing2label


def create_classification_dataset(
    ing2label: OrderedDict,
    ing2name: OrderedDict,
    split="train",
    image_size: int = 1024,
    embedding_model: Optional[FeatureExtractor] = None,
    device: str = "cpu",
    num_workers: int = 4,
    batch_size: int = 8,
    pin_memory: bool = False,
):
    if embedding_model is not None:
        dataset = FoodWasteDatasetWithEmbeddings(
            ing2label=ing2label,
            ing2name=ing2name,
            split=split,
            image_size=image_size,
            embedding_model=embedding_model,
        )
        dataset.compute_embeddings(
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
        )
        return dataset
    else:
        return FoodWasteDataset(
            ing2label=ing2label, ing2name=ing2name, split=split, image_size=image_size
        )


def load_train_and_val_datasets(
    image_size: int = 1024,
    embedding_model: Optional[FeatureExtractor] = None,
    device: str = "cpu",
    num_workers: int = 4,
    batch_size: int = 8,
    pin_memory: bool = False,
) -> Tuple[FoodWasteDataset, FoodWasteDataset]:
    ing2name, ing2label = get_label_mapping()
    cfg = {
        "image_size": image_size,
        "embedding_model": embedding_model,
        "device": device,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "pin_memory": pin_memory,
    }
    train_dataset = create_classification_dataset(
        ing2label=ing2label, ing2name=ing2name, split="train", **cfg
    )
    val_dataset = create_classification_dataset(
        ing2label=ing2label, ing2name=ing2name, split="test", **cfg
    )
    return train_dataset, val_dataset
