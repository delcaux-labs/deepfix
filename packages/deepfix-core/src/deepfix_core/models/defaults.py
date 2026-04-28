from enum import StrEnum


class DataType(StrEnum):
    """Types of data supported by the system."""

    VISION = "vision"
    TABULAR = "tabular"
    NLP = "nlp"
    IR = "ir"


class TaskType(StrEnum):
    """Types of machine learning tasks supported by the system."""

    # tabular tasks
    TABULAR_CLASSIFICATION = "tabular_classification"
    TABULAR_REGRESSION = "tabular_regression"
    # vision tasks
    IMAGE_SEGMENTATION = "image_segmentation"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    # NLP tasks
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_TOKEN_CLASSIFICATION = "text_token_classification"
    INFORMATION_RETRIEVAL = "information_retrieval"
