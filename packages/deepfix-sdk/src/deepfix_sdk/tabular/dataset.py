from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset as HFDataset
from deepfix_core.models import DataType

from ..data.base import BaseDataset


class TabularDataset(BaseDataset):
    """DeepFix Tabular Dataset backed by Hugging Face Datasets and Deepchecks."""

    def __init__(
        self,
        dataset_name: str,
        dataset: pd.DataFrame,
        label: Optional[str] = None,
        cat_features: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            from deepchecks.tabular import Dataset as DeepchecksTabularDataset
        except ImportError:
            raise ImportError(
                "Tabular dependencies are required for this module. "
                "Install with: pip install deepfix-sdk[tabular]"
            ) from None

        self.dataset_name = dataset_name
        self._hf_dataset: Optional[HFDataset] = None

        if isinstance(dataset, pd.DataFrame):
            df = dataset
        else:
            raise ValueError(f"Invalid dataset type: {type(dataset)}")

        assert label is not None, "Label column is required"
        self.dataset = DeepchecksTabularDataset(
            df, label=label, cat_features=cat_features or []
        )

        self._metadata: Dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "label": self.dataset.label_name,
            "cat_features": self.dataset.cat_features,
            "num_features": self.dataset.numerical_features,
            **(metadata or {}),
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
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
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

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

    @property
    def label(self) -> Optional[str]:
        return self.dataset.label_name

    def to_loader(self, *args, **kwargs) -> "TabularDataset":
        return self

    def to_hf_dataset(self) -> HFDataset:
        """Convert or export to a Hugging Face Dataset instance with embedded metadata."""
        if self._hf_dataset is not None:
            return self._hf_dataset

        df = self.get_data()
        hf_ds = HFDataset.from_pandas(df)
        hf_ds.info.custom_attributes = dict(self._metadata)
        hf_ds.info.description = f"DeepFix Tabular Dataset: {self.dataset_name}"
        self._hf_dataset = hf_ds
        return hf_ds

    @classmethod
    def from_hf_dataset(
        cls,
        dataset: HFDataset,
        dataset_name: Optional[str] = None,
        label: Optional[str] = None,
        cat_features: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TabularDataset":
        """Instantiate a TabularDataset from a Hugging Face Dataset."""
        custom_attrs = getattr(dataset.info, "custom_attributes", {}) or {}
        name = dataset_name or custom_attrs.get("dataset_name") or "hf_tabular_dataset"
        inferred_label = label or custom_attrs.get("label")
        inferred_cats = cat_features or custom_attrs.get("cat_features")
        df = dataset.to_pandas()

        return cls(
            dataset_name=name,
            dataset=df,
            label=inferred_label,
            cat_features=inferred_cats,
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
        """Push tabular dataset to S3 bucket as Parquet with embedded metadata and return canonical S3 URI."""
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

        df = self.get_data()
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
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "TabularDataset":
        """Load tabular dataset from an S3 URI (extracting metadata and DataFrame from Parquet)."""
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

        table = pq.read_table(buffer)
        df = table.to_pandas()

        meta_bytes = (table.schema.metadata or {}).get(b"deepfix_metadata")
        if meta_bytes:
            meta = json.loads(meta_bytes.decode("utf-8"))
            dataset_name = (
                kwargs.get("dataset_name")
                or meta.get("dataset_name")
                or os.path.splitext(os.path.basename(key))[0]
            )
            label = kwargs.get("label") or meta.get("label")
            cat_features = kwargs.get("cat_features") or meta.get("cat_features")
        else:
            meta = {}
            dataset_name = (
                kwargs.get("dataset_name")
                or os.path.splitext(os.path.basename(key))[0]
            )
            label = (
                kwargs.get("label")
                or kwargs.get("label_column")
                or (df.columns[-1] if len(df.columns) > 0 else None)
            )
            cat_features = kwargs.get("cat_features")

        return cls(
            dataset_name=dataset_name,
            dataset=df,
            label=label,
            cat_features=cat_features,
            metadata=meta,
        )

