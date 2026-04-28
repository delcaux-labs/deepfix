from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Union, Tuple

import numpy as np
import pandas as pd
from deepchecks.nlp import TextData
from deepchecks.nlp.task_type import TTextLabel
from deepchecks.tabular import Dataset as DeepchecksTabularDataset
from deepchecks.vision import VisionData
from deepfix_core.models import DataType
from supervision.dataset.core import DetectionDataset
from supervision.detection.core import Detections
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import runtime_checkable

from ..data.loader import (
    ClassificationVisionDataLoader,
    DetectionVisionDataLoader,
    SegmentationVisionDataLoader,
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class BaseDataset(Protocol):
    def to_loader(self, model: Optional[Callable] = None, batch_size: int = 8) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def data_type(self) -> DataType:
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def name(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")


class VisionDataset(BaseDataset):
    def __init__(self, dataset_name: str, dataset: Union[Dataset, DetectionDataset]):
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

    @property
    def name(self) -> str:
        return self.dataset_name


class ImageClassificationDataset(VisionDataset):
    def __init__(self, dataset_name: str, dataset: Dataset):
        super().__init__(dataset_name=dataset_name, dataset=dataset)

    def to_loader(
        self, model: Optional[Callable] = None, batch_size: int = 8
    ) -> ClassificationVisionDataLoader:
        return ClassificationVisionDataLoader.load_from_dataset(
            self.dataset,
            batch_size=batch_size,
            model=model,
        )

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return dict(image=image, label=label)


class ObjectDetectionDataset(VisionDataset):
    def __init__(self, dataset_name: str, dataset: DetectionDataset):
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

    def get_annotations(
        self,
    ) -> Dict[str, Detections]:
        return self.dataset.annotations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Union[str, np.ndarray, Detections]]:
        image_path, image, annotation = self.dataset[idx]
        return dict(image_path=image_path, image=image, label=annotation)

    def __iter__(self):
        return iter(self.dataset)

    def to_loader(
        self, batch_size: int = 8, shuffle: bool = False, **kwargs
    ) -> VisionData:
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
        dataset: Dataset,
        label_map: Optional[Dict[int, str]] = None,
    ):
        assert isinstance(dataset, Dataset), (
            f"dataset must be an instance of Dataset. Received: {type(dataset)}"
        )
        super().__init__(dataset_name=dataset_name, dataset=dataset)
        self.label_map = label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, Union[np.ndarray, np.ndarray]]:
        image, annotation = self.dataset[idx]
        c = image.shape[0]
        if isinstance(image, Tensor):
            image = image.cpu().numpy()
        if isinstance(annotation, Tensor):
            annotation = annotation.cpu().long().numpy()
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
            if isinstance(label, Tensor):
                label = label.view(-1)
            elif isinstance(label, np.ndarray):
                label = label.ravel()

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


class TabularDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset: pd.DataFrame,
        label: Optional[str] = None,
        cat_features: Optional[List[str]] = None,
    ):
        if isinstance(dataset, pd.DataFrame):
            assert label is not None, "Label column is required"
            self.dataset = DeepchecksTabularDataset(
                dataset, label=label, cat_features=cat_features or []
            )

        else:
            raise ValueError(f"Invalid dataset type: {type(dataset)}")

        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.data.iloc[idx], self.dataset.label_col.iloc[idx]

    def __iter__(self):
        return iter(self.dataset)

    def get_data(self) -> pd.DataFrame:
        return self.dataset.data.copy()

    @property
    def data_type(self) -> DataType:
        return DataType.TABULAR

    @property
    def data(self) -> pd.DataFrame:
        return self.get_data()

    @property
    def name(self) -> str:
        return self.dataset_name

    @property
    def X(self) -> pd.DataFrame:
        x = self.get_data().drop(columns=[self.dataset.label_name])
        x[self.cat_features] = x[self.cat_features].astype("category")
        return x

    @property
    def y(self) -> pd.Series:
        return self.dataset.label_col.copy()

    @property
    def cat_features(self) -> List[str]:
        return self.dataset.cat_features

    @property
    def num_features(self) -> List[str]:
        return self.dataset.numerical_features

    def to_loader(self, *args, **kwargs) -> "TabularDataset":
        return self


class NLPDataset(BaseDataset):
    def __init__(self, dataset_name: str, dataset: TextData):
        self.dataset = dataset
        self.dataset_name = dataset_name

    def to_loader(self, *args, **kwargs) -> TextData:
        return self.dataset

    def __len__(self):
        return len(self.dataset)

    @property
    def data_type(self) -> DataType:
        return DataType.NLP

    @property
    def data(self) -> TextData:
        return self.dataset

    @property
    def embeddings(self) -> np.ndarray:
        return self.dataset.embeddings

    @property
    def X(self) -> Sequence[str]:
        return self.dataset.text

    @property
    def y(self) -> TTextLabel:
        return self.dataset.label

    @property
    def name(self) -> str:
        return self.dataset_name


class InformationRetrievalDataset(NLPDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset: TextData,
        qrels: Union[List[Dict[str, Any]], pd.DataFrame],
    ):
        super().__init__(dataset_name, dataset)
        self.qrels = pd.DataFrame(qrels) if isinstance(qrels, list) else qrels
        self.predictions = None
        self.probabilities = None

        # Classes are strictly binary relevance "0" and "1"
        self.model_classes = ["0", "1"]
        self.fp_probabilities = [1.0, 0.0]

    @staticmethod
    def get_cosine_similarity(q_emb: np.ndarray, e_emb: np.ndarray) -> float:
        dot_product = np.dot(q_emb, e_emb)
        norm_q = np.linalg.norm(q_emb)
        norm_e = np.linalg.norm(e_emb)
        if norm_q > 0 and norm_e > 0:
            sim = dot_product / (norm_q * norm_e)
        else:
            sim = 0.0
        return float(sim)

    @classmethod
    def from_ir_data(
        cls,
        dataset_name: str,
        queries: Dict[str, Dict[str, Any]],
        corpus: Dict[str, Dict[str, Any]],
        qrels: List[Dict[str, Any]],
        query_embeddings: Optional[Dict[str, np.ndarray]] = None,
        corpus_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> "InformationRetrievalDataset":
        pairs, labels, rows = [], [], []

        for qrel in qrels:
            q_id = qrel["query_id"]
            e_id = qrel["entity_id"]
            relevance = qrel["relevance"]
            rank = qrel["rank"]
            q = queries[q_id]
            e = corpus[e_id]

            text = f"{q['query']} [SEP] {e['text']}"
            pairs.append(text)
            labels.append(str(relevance))
            row = {
                "desc_length": len(e.get("text", "").split()),
                "query_length": len(q.get("query", "").split()),
                "query_id": q_id,
                "entity_id": e_id,
                "gt_rank": rank,
            }
            if query_embeddings is not None and corpus_embeddings is not None:
                q_emb = query_embeddings.get(q_id)
                e_emb = corpus_embeddings.get(e_id)
                if q_emb is not None and e_emb is not None:
                    row["cosine_similarity"] = cls.get_cosine_similarity(q_emb, e_emb)
            rows.append(row)

        metadata_df = pd.DataFrame(rows)
        categorical_metadata = ["desc_length", "query_length", "query_id", "entity_id"]

        text_data = TextData(
            raw_text=pairs,
            label=labels,
            name=dataset_name,
            task_type="text_classification",
            metadata=metadata_df,
            categorical_metadata=categorical_metadata,
        )

        return cls(dataset_name=dataset_name, dataset=text_data, qrels=qrels)

    def set_predictions(
        self,
        retrievals: pd.DataFrame,
        rank_to_grade: Optional[Callable[[float], int]] = None,
    ) -> None:

        results_df = retrievals.copy()
        qrels_df = self.qrels

        assert np.array([a for a in results_df["score"]]).shape[1] == len(
            self.model_classes
        ), "score should be a vector of size equal to number of classes"

        # Left join: unranked entities get score=NaN, rank=NaN
        pairs_df = qrels_df.merge(
            results_df[["query_id", "entity_id", "score", "rank", "relevance"]],
            on=["query_id", "entity_id"],
            how="left",
            suffixes=("_gt", ""),
        )
        # Predictions are binary relevance from the model
        self.predictions = (
            pairs_df["relevance"].fillna(0).astype(int).astype(str).tolist()
        )

        # Probabilities: Use retrieved score or default to [1.0, 0.0] for false positives
        self.probabilities = (
            pairs_df["score"]
            .apply(lambda x: self.fp_probabilities if pd.isna(x) is True else x)
            .tolist()
        )

        return None
