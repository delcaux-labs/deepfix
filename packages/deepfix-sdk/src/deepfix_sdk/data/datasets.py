from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    Tuple,
)
from functools import cached_property

import re
import string
import numpy as np
import pandas as pd
from deepchecks.nlp import TextData
from deepchecks.nlp.task_type import TTextLabel
from deepchecks.tabular import Dataset as DeepchecksTabularDataset
from deepchecks.vision import VisionData
from deepfix_core.models import DataType
from supervision.dataset.core import DetectionDataset
from supervision.detection.core import Detections
import tiktoken
import pyterrier as pt
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

    def to_loader(self, *args, **kwargs) -> "NLPDataset":
        """Compute properties of the dataset and return it."""
        self.dataset.calculate_builtin_properties()
        return self

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


class InformationRetrievalDataset(pt.datasets.Dataset, BaseDataset):
    """An IR dataset backed by PyTerrier's Dataset interface.

    Stores topics, qrels, and a corpus source lazily. The Deepchecks TextData
    representation is only materialized on first access to ``.dataset``.

    Can be constructed:
    - Directly via ``__init__`` with DataFrames and a corpus iterator factory.
    - From a PyTerrier dataset via ``from_pyterrier(pt_dataset)``.
    - From raw dicts via ``from_ir_data(...)`` (backward compatible).
    """

    def __init__(
        self,
        dataset_name: str,
        topics: pd.DataFrame,
        qrels: pd.DataFrame,
        corpus_iter: Callable[[], Iterable[Dict[str, Any]]],
    ):
        """
        Args:
            dataset_name: Human-readable name for the dataset.
            topics: DataFrame with columns ``qid`` and ``query`` or ``text``.
            qrels: DataFrame with columns ``qid``, ``docno``, and ``label``.
            corpus_iter: A callable that returns an iterable of dicts,
                each with at least ``docno`` and ``text`` keys.
        """
        self.dataset_name = dataset_name
        self._topics = topics
        self._qrels = qrels
        self._corpus_iter = corpus_iter

        self.predictions: Optional[list[int]] = None
        self.probabilities: Optional[list[float]] = None
        self._embeddings: Optional[np.ndarray] = None
        self._l1_norm: Optional[np.ndarray] = None
        self._cosine_sim: Optional[np.ndarray] = None
        self._metadata: Optional[pd.DataFrame] = None
        self._categorical_metadata: Optional[list[str]] = None

        self._text_data: Optional[TextData] = None

        # Classes are strictly binary relevance "0" and "1"
        self.model_classes = ["0", "1"]
        self.fp_probabilities = [1.0, 0.0]

        self.tokenizer = None

    # ------------------------------------------------------------------
    # pt.datasets.Dataset interface
    # ------------------------------------------------------------------

    def get_topics(self, variant: Optional[str] = None) -> pd.DataFrame:
        """Return topics as a DataFrame with ``qid`` and ``query``/``text`` columns."""
        return self._topics

    def get_qrels(self, variant: Optional[str] = None) -> pd.DataFrame:
        """Return qrels as a DataFrame with ``qid``, ``docno``, and ``label`` columns."""
        return self._qrels

    def get_corpus_iter(self, *, verbose: bool = True) -> Iterable[Dict[str, Any]]:
        """Yield dicts with ``docno`` and ``text`` keys."""
        return self._corpus_iter()

    # ------------------------------------------------------------------
    # BaseDataset protocol
    # ------------------------------------------------------------------

    @property
    def data_type(self) -> DataType:
        return DataType.IR

    @property
    def name(self) -> str:
        return self.dataset_name

    def to_loader(self, *args, **kwargs) -> "InformationRetrievalDataset":
        return self

    # ------------------------------------------------------------------
    # Lazy Deepchecks materialisation
    # ------------------------------------------------------------------

    @property
    def dataset(self) -> TextData:
        """Lazily materialise the Deepchecks ``TextData`` from topics + qrels + corpus."""

        if self._text_data is not None:
            return self._text_data
            
        logger.info("Materialising TextData for '%s' …", self.dataset_name)

        # Build lookup dicts
        q = "query" if "query" in self._topics.columns else "text"
        queries = {str(row["qid"]): row[q] for _, row in self._topics.iterrows()}

        needed_docnos = set(self._qrels["docno"].astype(str).unique())
        corpus: Dict[str, str] = {}
        try:
            for doc in self._corpus_iter():
                docno = str(doc["docno"])
                if docno in needed_docnos:
                    corpus[docno] = doc.get("text", doc.get("body", None))
                    if corpus[docno] is None:
                        raise ValueError(
                            f"Document {docno} has no text or body. Found {list(doc.keys())}."
                        )
                    if len(corpus) == len(needed_docnos):
                        break
        except (ValueError, EOFError, OSError) as e:
            logger.warning(
                "Corpus iteration stopped early for '%s' due to error: %s. "
                "Found %d / %d needed documents. This may happen with corrupted/truncated caches.",
                self.dataset_name,
                e,
                len(corpus),
                len(needed_docnos),
            )
            if len(corpus) == 0:
                raise

        pairs, labels, rows = [], [], []
        for _, row in self._qrels.iterrows():
            q_id = str(row["qid"])
            e_id = str(row["docno"])
            relevance = int(row["label"])

            q_text = queries.get(q_id, "")
            e_text = corpus.get(e_id, "")

            text = self.format_pair(q_text, e_text)
            pairs.append(text)
            labels.append(relevance)

            rows.append(
                {
                    "query_token_count": len(self.get_tokens(q_text)),
                    "doc_token_count": len(self.get_tokens(e_text)),
                    "query_id": q_id,
                    "doc_id": e_id,
                }
            )

        self._metadata = pd.DataFrame(rows)
        self._categorical_metadata=["query_token_count", "doc_token_count","query_id","doc_id"]
        # create dataset instance from deepchecks
        self._text_data = TextData(
            raw_text=pairs,
            label=labels,
            name=self.dataset_name,
            task_type="text_classification",
            metadata=self._metadata.copy(),
            categorical_metadata=self._categorical_metadata[:]
        )
        self.calculate_text_properties()
        return self._text_data
    
    def calculate_text_properties(self):
        self._text_data.calculate_builtin_properties()
        if self._text_data.properties is not None:
            self._categorical_metadata.extend(self._text_data.categorical_properties)
            self._metadata = pd.concat([self._metadata, self._text_data.properties], axis=1).reset_index(drop=True)


    @property
    def qrels(self) -> pd.DataFrame:
        """Return qrels in the internal format (query_id, doc_id, relevance)."""
        return self._qrels.rename(
            columns={"qid": "query_id", "docno": "doc_id", "label": "relevance"}
        )
    
    @property
    def metadata(self) -> pd.DataFrame:
        return self._metadata

    @property
    def categorical_metadata(self) -> list[str]:
        return list(self._categorical_metadata)

    @property
    def X(self) -> Sequence[str]:
        return self.dataset.text

    @property
    def y(self) -> TTextLabel:
        return self.dataset.label

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        return self._embeddings

    def __len__(self):
        return len(self._qrels)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pyterrier(
        cls,
        pt_dataset: Any,
        dataset_name: Optional[str] = None,
    ) -> "InformationRetrievalDataset":
        """Create from an existing PyTerrier dataset (e.g. ``pt.get_dataset(...)``).

        Data is NOT loaded into memory at construction time; the corpus is
        streamed lazily on first access to ``.dataset``.

        Args:
            pt_dataset: A PyTerrier dataset object.
            dataset_name: Optional override name.
        """
        if dataset_name is None:
            dataset_name = (
                getattr(pt_dataset, "info_url", lambda: None)() or "pyterrier_dataset"
            )

        return cls(
            dataset_name=dataset_name,
            topics=pt_dataset.get_topics(),
            qrels=pt_dataset.get_qrels(),
            corpus_iter=lambda: pt_dataset.get_corpus_iter(),
        )

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
        """Backward-compatible constructor from raw dicts.

        Converts the raw dict-based IR data into the PyTerrier-native format
        (topics, qrels, corpus DataFrames / iterators).
        """
        # Build topics DataFrame
        topics_rows = [
            {"qid": qid, "query": q_data.get("query", "")}
            for qid, q_data in queries.items()
        ]
        topics_df = pd.DataFrame(topics_rows)

        # Build qrels DataFrame
        qrels_rows = [
            {"qid": q["query_id"], "docno": q["doc_id"], "label": q["relevance"]}
            for q in qrels
        ]
        qrels_df = pd.DataFrame(qrels_rows)

        # Build corpus iterator factory from in-memory dict
        def _corpus_iter():
            for docno, doc_data in corpus.items():
                yield {"docno": docno, "text": doc_data.get("text", "")}

        return cls(
            dataset_name=dataset_name,
            topics=topics_df,
            qrels=qrels_df,
            corpus_iter=_corpus_iter,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def split(
        self,
        train_size: float = 0.8,
        random_state: int = 42,
    ) -> Tuple["InformationRetrievalDataset", "InformationRetrievalDataset"]:
        """Split the dataset into two non-overlapping train/test partitions.

        Uses stratified sampling on the relevance labels so that the label
        distribution is preserved in both partitions.

        Args:
            train_size: Fraction of qrel rows to include in the train split.
            random_state: Random seed for reproducibility.

        Returns:
            A ``(train_dataset, test_dataset)`` tuple, each sharing the same
            corpus iterator factory as the original dataset.
        """
        from sklearn.model_selection import train_test_split

        qrels_df = self._qrels.copy()

        train_df, test_df = train_test_split(
            qrels_df,
            train_size=train_size,
            random_state=random_state,
            stratify=qrels_df["label"],
        )

        parent_corpus_iter = self._corpus_iter
        train_docnos = set(train_df["docno"].astype(str).unique())
        test_docnos = set(test_df["docno"].astype(str).unique())

        def _filtered_corpus_iter(docnos):
            def _iter():
                for doc in parent_corpus_iter():
                    if str(doc["docno"]) in docnos:
                        yield doc

            return _iter

        train_ds = InformationRetrievalDataset(
            dataset_name=f"{self.dataset_name}_train",
            topics=self._topics,
            qrels=train_df.reset_index(drop=True),
            corpus_iter=_filtered_corpus_iter(train_docnos),
        )
        test_ds = InformationRetrievalDataset(
            dataset_name=f"{self.dataset_name}_test",
            topics=self._topics,
            qrels=test_df.reset_index(drop=True),
            corpus_iter=_filtered_corpus_iter(test_docnos),
        )

        return train_ds, test_ds

    @staticmethod
    def format_pair(query: str, document: str) -> str:
        """Combine query and document into a formatted string."""
        return f"<query> {query} </query> <document> {document} </document>"

    @staticmethod
    def parse_pair(text: str) -> Tuple[str, str]:
        """Extract query and document from a formatted string."""
        query_match = re.search(r"<query>(.*?)</query>", text, re.DOTALL)
        doc_match = re.search(r"<document>(.*?)</document>", text, re.DOTALL)
        q_text = query_match.group(1).strip() if query_match else ""
        d_text = doc_match.group(1).strip() if doc_match else ""
        return q_text, d_text

    @staticmethod
    def get_tokens(
        text: str, tokenizer: Optional[tiktoken.Encoding] = None
    ) -> np.ndarray:
        if tokenizer is None:
            tokenizer = tiktoken.get_encoding("o200k_base")
        return tokenizer.encode(text)

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
            results_df[["query_id", "doc_id", "score", "rank", "relevance"]],
            on=["query_id", "doc_id"],
            how="left",
            suffixes=("_gt", ""),
        )
        # Predictions are binary relevance from the model
        self.predictions = (
            pairs_df["relevance"].fillna(0).astype(int).tolist()
        )

        # Probabilities: Use retrieved score or default to [1.0, 0.0] for false positives
        self.probabilities = (
            pairs_df["score"]
            .apply(lambda x: self.fp_probabilities if pd.isna(x) is True else x)
            .tolist()
        )

        return None

    def set_embeddings(self, embedder: Callable[[str], np.ndarray]) -> None:
        """Compute and set embeddings for the dataset using the provided embedder.

        The embedder is applied to queries and documents separately, and the
        resulting pair embedding is the difference (query - document).
        """
        logger.info("Computing embeddings for '%s' ...", self.dataset_name)

        self.dataset # ensure properties are computed

        cache: Dict[str, np.ndarray] = {}

        def get_embedding(text: str) -> np.ndarray:
            if text not in cache:
                cache[text] = embedder(text)
            return cache[text]

        embeddings = []
        l1_norm = []
        cosine_sim = []
        for text in self._text_data.text:
            q_text, d_text = self.parse_pair(text)
            q_emb = get_embedding(q_text)
            d_emb = get_embedding(d_text)
            embeddings.append(q_emb - d_emb)
            l1_norm.append(np.linalg.norm(q_emb - d_emb, ord=1))
            cosine_sim.append(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb)))
        
        self._l1_norm = np.array(l1_norm)
        self._cosine_sim = np.array(cosine_sim)
        self._embeddings = np.array(embeddings)

        # Update Deepchecks TextData with the computed embeddings
        self._text_data.set_embeddings(self._embeddings)

        return None

    @property
    def l1_norm(self) -> np.ndarray:
        return self._l1_norm

    @property
    def cosine_sim(self) -> np.ndarray:
        return self._cosine_sim

    def to_tabular(self) -> TabularDataset:
        """Convert the IR dataset to a TabularDataset view for Deepchecks Tabular suites."""
        self.dataset # ensure properties are computed

        df = self.metadata.copy()
        label_name = "relevance"
        df[label_name] = self.y
        df[label_name] = df[label_name].apply(int)

        if self.l1_norm is not None:
            df["l1_norm"] = self.l1_norm
        if self.cosine_sim is not None:
            df["cosine_sim"] = self.cosine_sim

        # Remove id columns from categorical features
        cat_features = list(self.categorical_metadata)        
        if "query_id" in df.columns:
            df.drop(columns=["query_id",], inplace=True)
            cat_features.remove('query_id')
        if "doc_id" in df.columns:
            df.drop(columns=["doc_id",], inplace=True)
            cat_features.remove('doc_id')

        # TODO: provide embeddings as features for tabular dataset?
        
        return TabularDataset(
            dataset_name=f"{self.dataset_name}_tabular",
            dataset=df,
            label=label_name,
            cat_features=cat_features,
        )

    def to_nlp_dataset(self) -> NLPDataset:
        """Wrap the materialised TextData as an NLPDataset for Deepchecks NLP suites."""
        return NLPDataset(dataset_name=self.dataset_name, dataset=self.dataset)
