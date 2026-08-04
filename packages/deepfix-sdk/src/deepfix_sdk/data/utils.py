from typing import Optional, Union
from deepfix_core.models import DataType
from .base import BaseDataset

def get_data_statistics(
    data_type: Union[str, DataType],
    train_data: BaseDataset,
    test_data: Optional[BaseDataset] = None,
):
    if data_type == DataType.VISION:
        from ..vision.utils import VisionDataStatistics
        return VisionDataStatistics(
            train_data=train_data, test_data=test_data
        ).get_statistics()
    elif data_type == DataType.TABULAR:
        from ..tabular.utils import TabularDataStatistics
        return TabularDataStatistics(
            train_data=train_data, test_data=test_data
        ).get_statistics()
    elif data_type == DataType.NLP:
        from ..nlp.utils import NLPDataStatistics
        return NLPDataStatistics(
            train_data=train_data, test_data=test_data
        ).get_statistics()
    elif data_type == DataType.IR:
        from ..ir.utils import IRDataStatistics
        return IRDataStatistics(
            train_data=train_data, test_data=test_data
        ).get_statistics()
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
