from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Protocol,
)

from deepfix_core.models import (
    BaseDatasetStatistics,
    DataType,
    TaskType,
)
from typing_extensions import runtime_checkable


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

    def to_hf_dataset(self) -> Any:
        """Convert or export to a Hugging Face Dataset instance."""
        raise NotImplementedError("Subclasses must implement this method")

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
        """Push dataset artifacts to an S3 bucket and return the canonical S3 URI.

        Args:
            s3_bucket (str): Target S3 bucket name.
            s3_prefix (str, optional): Optional S3 prefix / folder path.
            aws_access_key_id (str, optional): Optional AWS access key ID.
            aws_secret_access_key (str, optional): Optional AWS secret access key.
            endpoint_url (str, optional): Optional S3 endpoint URL.
            region_name (str, optional): Optional AWS region name.
            **kwargs: Additional dataset-specific upload arguments.

        Returns:
            str: Canonical S3 URI (e.g. ``s3://bucket/datasets/name.parquet``).
        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def from_s3(
        cls,
        s3_uri: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Load dataset from an S3 URI.

        Args:
            s3_uri (str): Canonical S3 URI (e.g. ``s3://bucket/path/data.parquet``).
            aws_access_key_id (str, optional): Optional AWS access key ID.
            aws_secret_access_key (str, optional): Optional AWS secret access key.
            endpoint_url (str, optional): Optional S3 endpoint URL.
            region_name (str, optional): Optional AWS region name.
            **kwargs: Additional dataset-specific loader arguments.

        Returns:
            BaseDataset: Initialized dataset instance.
        """
        raise NotImplementedError("Subclasses must implement this method")


class BaseDataStatistics(Protocol):
    def __init__(
        self, train_data: BaseDataset, test_data: Optional[BaseDataset] = None
    ):
        self.train_data = train_data
        self.test_data = test_data

    @property
    def task_type(self) -> TaskType:
        raise NotImplementedError("Subclasses must implement this property")

    def get_statistics(self) -> Dict[str, Any]:
        stats = {"train": self.get_train_statistics()}
        if self.test_data is not None:
            stats["test"] = self.get_test_statistics()
        stats["task_type"] = self.task_type
        return stats

    def get_train_statistics(self) -> BaseDatasetStatistics:
        raise NotImplementedError(
            "get_train_statistics method must be implemented in the subclass"
        )

    def get_test_statistics(self) -> BaseDatasetStatistics:
        raise NotImplementedError(
            "get_test_statistics method must be implemented in the subclass"
        )

    def _compute_statistics(self, dataset: BaseDataset) -> BaseDatasetStatistics:
        raise NotImplementedError(
            "_compute_statistics method must be implemented in the subclass"
        )
