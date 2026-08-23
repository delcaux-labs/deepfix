from typing import Any, Dict, Optional, Sequence, Union

from datasets import Dataset as HFDataset
from deepfix_core.models import DataType

from ..data.base import BaseDataset


class NLPDataset(BaseDataset):
    """DeepFix NLP Dataset backed by Hugging Face Datasets and Deepchecks TextData."""

    def __init__(
        self,
        dataset_name: str,
        dataset: Union[HFDataset, Dict[str, Any], Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        self._hf_dataset: Optional[HFDataset] = None

        text_data_cls = None
        try:
            from deepchecks.nlp import TextData
            text_data_cls = TextData
        except (ImportError, Exception):
            pass

        if isinstance(dataset, HFDataset):
            self._hf_dataset = dataset
            text = dataset["text"] if "text" in dataset.column_names else []
            label = dataset["label"] if "label" in dataset.column_names else None
            if text_data_cls is not None:
                self.dataset = text_data_cls(text=text, label=label)
            else:
                self.dataset = dataset
        elif isinstance(dataset, dict):
            text = dataset.get("text", [])
            label = dataset.get("label")
            if text_data_cls is not None:
                self.dataset = text_data_cls(text=text, label=label)
            else:
                self.dataset = dataset
        else:
            self.dataset = dataset

        self._metadata: Dict[str, Any] = {
            "dataset_name": self.dataset_name,
            **(metadata or {}),
        }

    def to_loader(self, *args, **kwargs) -> "NLPDataset":
        """Compute properties of the dataset and return it."""
        if hasattr(self.dataset, "calculate_builtin_properties"):
            self.dataset.calculate_builtin_properties()
        return self

    def __len__(self) -> int:
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)
        elif self._hf_dataset is not None:
            return len(self._hf_dataset)
        return 0

    @property
    def data_type(self) -> DataType:
        return DataType.NLP

    @property
    def data(self):
        return self.dataset

    @property
    def embeddings(self):
        return getattr(self.dataset, "embeddings", None)

    @property
    def X(self) -> Sequence[str]:
        if hasattr(self.dataset, "text"):
            return self.dataset.text
        elif self._hf_dataset is not None and "text" in self._hf_dataset.column_names:
            return self._hf_dataset["text"]
        elif isinstance(self.dataset, dict):
            return self.dataset.get("text", [])
        return []

    @property
    def y(self):
        if hasattr(self.dataset, "label"):
            return self.dataset.label
        elif self._hf_dataset is not None and "label" in self._hf_dataset.column_names:
            return self._hf_dataset["label"]
        elif isinstance(self.dataset, dict):
            return self.dataset.get("label")
        return None

    @property
    def name(self) -> str:
        return self.dataset_name

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def to_hf_dataset(self) -> HFDataset:
        """Convert or export to a Hugging Face Dataset instance."""
        if self._hf_dataset is not None:
            return self._hf_dataset

        data_dict: Dict[str, Any] = {
            "text": list(self.X) if self.X is not None else [],
        }
        if self.y is not None:
            data_dict["label"] = list(self.y)

        hf_ds = HFDataset.from_dict(data_dict)
        hf_ds.info.custom_attributes = dict(self._metadata)
        hf_ds.info.description = f"DeepFix NLP Dataset: {self.dataset_name}"
        self._hf_dataset = hf_ds
        return hf_ds

    @classmethod
    def from_hf_dataset(
        cls,
        dataset: HFDataset,
        dataset_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "NLPDataset":
        """Instantiate an NLPDataset from a Hugging Face Dataset."""
        custom_attrs = getattr(dataset.info, "custom_attributes", {}) or {}
        name = dataset_name or custom_attrs.get("dataset_name") or "hf_nlp_dataset"
        return cls(
            dataset_name=name,
            dataset=dataset,
            metadata=metadata or custom_attrs,
        )

    def push_to_s3(
        self,
        s3_bucket: str,
        s3_prefix: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Push NLP dataset to S3 bucket as Parquet with embedded metadata and return canonical S3 URI."""
        import io
        import json
        import os

        import boto3
        import pyarrow as pa
        import pyarrow.parquet as pq

        prefix = s3_prefix.strip("/") if s3_prefix else f"datasets/{self.dataset_name}"
        filename = f"{self.dataset_name}.parquet"
        s3_key = f"{prefix}/{filename}" if prefix else filename

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key
            or os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        s3_client = session.client(
            "s3", endpoint_url=endpoint_url or os.getenv("AWS_ENDPOINT_URL")
        )

        hf_ds = self.to_hf_dataset()
        df = hf_ds.to_pandas()
        table = pa.Table.from_pandas(df)
        metadata_dict = dict(self._metadata)
        existing_meta = table.schema.metadata or {}
        merged_meta = {
            **existing_meta,
            b"deepfix_metadata": json.dumps(metadata_dict).encode("utf-8"),
        }
        table = table.replace_schema_metadata(merged_meta)

        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)

        s3_client.upload_fileobj(buffer, s3_bucket, s3_key)

        return f"s3://{s3_bucket}/{s3_key}"

    @classmethod
    def from_s3(
        cls,
        s3_uri: str,
        dataset_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "NLPDataset":
        """Load NLP dataset from an S3 URI."""
        import io
        import json
        import os
        from urllib.parse import urlparse

        import boto3
        import pyarrow.parquet as pq

        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        session = boto3.Session(
            aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key
            or os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        s3_client = session.client(
            "s3", endpoint_url=endpoint_url or os.getenv("AWS_ENDPOINT_URL")
        )

        buffer = io.BytesIO()
        s3_client.download_fileobj(bucket, key, buffer)
        buffer.seek(0)

        if key.endswith(".json"):
            data = json.loads(buffer.getvalue().decode("utf-8"))
            derived_name = (
                dataset_name
                or data.get("dataset_name")
                or os.path.splitext(os.path.basename(key))[0]
            )
            return cls(dataset_name=derived_name, dataset=data)

        table = pq.read_table(buffer)
        df = table.to_pandas()

        meta_bytes = (table.schema.metadata or {}).get(b"deepfix_metadata")
        meta = json.loads(meta_bytes.decode("utf-8")) if meta_bytes else {}
        derived_name = (
            dataset_name
            or meta.get("dataset_name")
            or os.path.splitext(os.path.basename(key))[0]
        )

        hf_ds = HFDataset.from_pandas(df)
        hf_ds.info.custom_attributes = meta

        return cls.from_hf_dataset(hf_ds, dataset_name=derived_name, metadata=meta)
