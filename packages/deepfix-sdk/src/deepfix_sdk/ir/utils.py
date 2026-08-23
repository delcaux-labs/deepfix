from typing import Optional

from deepfix_core.models import IRStatistics, TaskType

from ..data.base import BaseDataStatistics
from ..nlp.utils import NLPDataStatistics
from ..tabular.utils import TabularDataStatistics
from .dataset import InformationRetrievalDataset


class IRDataStatistics(BaseDataStatistics):
    def __init__(
        self,
        train_data: InformationRetrievalDataset,
        test_data: Optional[InformationRetrievalDataset] = None,
    ):
        assert isinstance(train_data, InformationRetrievalDataset), (
            f"train_data must be an instance of InformationRetrievalDataset, got {type(train_data)}"
        )
        self.train_data = train_data

        if test_data is not None:
            assert isinstance(test_data, InformationRetrievalDataset), (
                f"test_data must be an instance of InformationRetrievalDataset, got {type(test_data)}"
            )
            self.test_data = test_data
        else:
            self.test_data = None
        self._task_type = TaskType.INFORMATION_RETRIEVAL

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    def get_train_statistics(self) -> IRStatistics:
        return self._compute_statistics(self.train_data)

    def get_test_statistics(self) -> IRStatistics:
        if self.test_data is None:
            raise ValueError("test_data is None, cannot compute test statistics")
        return self._compute_statistics(self.test_data)

    def _compute_statistics(self, dataset: InformationRetrievalDataset) -> IRStatistics:
        nlp_dataset = dataset.to_nlp_dataset()
        nlp_data_stats = NLPDataStatistics(train_data=nlp_dataset)
        nlp_stats = nlp_data_stats.get_train_statistics()

        tabular_dataset = dataset.to_tabular()
        tabular_data_stats = TabularDataStatistics(train_data=tabular_dataset)
        tabular_stats = tabular_data_stats.get_train_statistics()

        num_queries = int(dataset.qrels["query_id"].nunique())
        num_relevant_docs = int(
            dataset.qrels[dataset.qrels["relevance"].astype(int) >= 1][
                "doc_id"
            ].nunique()
        )
        int(
            dataset.qrels[dataset.qrels["relevance"].astype(int) < 1][
                "doc_id"
            ].nunique()
        )

        return IRStatistics(
            num_samples=len(dataset),
            num_queries=num_queries,
            num_relevant_docs=num_relevant_docs,
            nlp_statistics=nlp_stats,
            tabular_statistics=tabular_stats,
        )
