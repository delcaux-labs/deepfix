"""
DeepChecks vision datasets utilities.

This module provides convenient loaders for pre-built vision datasets from the DeepChecks library,
supporting classification, object detection, and segmentation tasks.
"""

from typing import Optional, Tuple, Union
import logging
from deepchecks.vision import VisionData
from deepchecks.vision.datasets import classification, detection
import torch

try:
    from deepchecks.vision.datasets.segmentation import segmentation_coco
except ImportError:
    segmentation_coco = None

LOGGER = logging.getLogger(__name__)


# Classification Datasets
def load_mnist_classification(
    train: bool = True,
    n_samples: Optional[int] = None,
    batch_size: int = 8,
    shuffle: bool = False,
    pin_memory: bool = True,
    object_type: str = 'VisionData',
    use_iterable_dataset: bool = False,
    device: Union[str, torch.device] = 'cpu',
) -> Union[VisionData, 'torch.utils.data.DataLoader', Tuple[VisionData, VisionData]]:
    """
    Load MNIST classification dataset from DeepChecks.
    
    Args:
        train: Whether to load training or test dataset. Default is True.
        n_samples: Only relevant for VisionData. Number of samples to load. 
                  Returns the first n_samples if shuffle is False, otherwise selects 
                  n_samples at random. If None, returns all samples.
        batch_size: How many samples per batch to load. Default is 8.
        shuffle: If True, reshuffles data at every epoch. Cannot work with 
                use_iterable_dataset=True. Default is False.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory 
                   before returning them. Default is True.
        object_type: Object type to return. If 'VisionData' then 
                    deepchecks.vision.VisionData will be returned, if 'DataLoader' then 
                    torch.utils.data.DataLoader. Default is 'VisionData'.
        use_iterable_dataset: If True, will use IterableTorchMnistDataset instead of 
                             TorchMnistDataset. Default is False.
        device: Device to use in tensor calculations. Default is 'cpu'.
    
    Returns:
        Depending on the object_type parameter:
        - 'VisionData': deepchecks.vision.VisionData object
        - 'DataLoader': torch.utils.data.DataLoader object
        - 'both': Tuple of (train_vision_data, test_vision_data) if train=True
    """
    LOGGER.info("Loading MNIST classification dataset (train=%s, object_type=%s)", train, object_type)
    
    
    try:
        if train and object_type == 'VisionData':
            # Load both train and test for VisionData
            train_data = classification.mnist_torch.load_dataset(  # type: ignore
                train=True,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                device=device,
                object_type=object_type,
                use_iterable_dataset=use_iterable_dataset,
                n_samples=n_samples,
            )
            test_data = classification.mnist_torch.load_dataset(  # type: ignore
                train=False,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                device=device,
                object_type=object_type,
                use_iterable_dataset=use_iterable_dataset,
                n_samples=n_samples,
            )
            return train_data, test_data
        else:
            return classification.mnist_torch.load_dataset(  # type: ignore
                train=train,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                device=device,
                object_type=object_type,
                use_iterable_dataset=use_iterable_dataset,
                n_samples=n_samples,
            )
    except Exception as e:
        LOGGER.error("Failed to load MNIST classification dataset: %s", str(e))
        raise e


# Detection Datasets
def load_coco_detection(
    train: bool = True,
    n_samples: Optional[int] = None,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    object_type: str = 'VisionData',
    device: Union[str, torch.device] = 'cpu',
) -> Union[VisionData, 'torch.utils.data.DataLoader', Tuple[VisionData, VisionData]]:
    """
    Load COCO object detection dataset from DeepChecks.
    
    Args:
        train: Whether to load training or validation dataset. Default is True.
        n_samples: Only relevant for VisionData. Number of samples to load. 
                  Returns the first n_samples if shuffle is False, otherwise selects 
                  n_samples at random. If None, returns all samples.
        batch_size: How many samples per batch to load. Default is 8.
        shuffle: If True, reshuffles data at every epoch. Default is False.
        num_workers: Number of workers to use for data loading. Default is 0.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory 
                   before returning them. Default is True.
        object_type: Object type to return. If 'VisionData' then 
                    deepchecks.vision.VisionData will be returned, if 'DataLoader' then 
                    torch.utils.data.DataLoader. Default is 'VisionData'.
        device: Device to use in tensor calculations. Default is 'cpu'.
    
    Returns:
        Depending on the object_type parameter:
        - 'VisionData': deepchecks.vision.VisionData object
        - 'DataLoader': torch.utils.data.DataLoader object
    """
    LOGGER.info("Loading COCO detection dataset (train=%s, object_type=%s)", train, object_type)
    
    if detection is None:
        raise ImportError("DeepChecks vision module not installed")
    
    try:
        if train and object_type == 'VisionData':
            # Load both train and val for VisionData
            train_data = detection.coco_torch.load_dataset(  # type: ignore
                train=True,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                device=device,
                object_type=object_type,
                n_samples=n_samples,
            )
            val_data = detection.coco_torch.load_dataset(  # type: ignore
                train=False,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                device=device,
                object_type=object_type,
                n_samples=n_samples,
            )
            return train_data, val_data
        else:
            return detection.coco_torch.load_dataset(  # type: ignore
                train=train,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                device=device,
                object_type=object_type,
                n_samples=n_samples,
            )
    except Exception as e:
        LOGGER.error("Failed to load COCO detection dataset: %s", str(e))
        raise


def load_mask_detection(
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    object_type: str = 'VisionData',
    day_index: int = 0,
) -> Union[VisionData, 'torch.utils.data.DataLoader']:
    """
    Load mask face detection dataset from DeepChecks.
    
    The mask dataset contains images of people wearing masks, people not wearing masks, 
    and people wearing masks incorrectly. Dataset source: 
    https://www.kaggle.com/datasets/andrewmvd/face-mask-detection (CC0 license)
    
    Args:
        batch_size: How many samples per batch to load. Default is 8.
        shuffle: If True, reshuffles data at every epoch. Default is False.
        num_workers: Number of workers to use for data loading. Default is 0.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory 
                   before returning them. Default is True.
        object_type: Object type to return. If 'VisionData' then 
                    deepchecks.vision.VisionData will be returned, if 'DataLoader' then 
                    torch.utils.data.DataLoader. Default is 'VisionData'.
        day_index: Optional day index for selecting specific day in production data. 
                  Default is 0. 0 is the training set, and each subsequent number is a 
                  subsequent day in the production dataset. Last day index is 59.
    
    Returns:
        Depending on the object_type parameter:
        - 'VisionData': deepchecks.vision.VisionData object
        - 'DataLoader': torch.utils.data.DataLoader object
    
    Raises:
        ImportError: If DeepChecks vision module is not installed
    """
    LOGGER.info("Loading mask detection dataset (object_type=%s, day_index=%s)", object_type, day_index)
    
    if detection is None:
        raise ImportError("DeepChecks vision module not installed")
    
    try:
        kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'object_type': object_type,
        }
                
        return detection.mask.load_dataset(**kwargs)  # type: ignore
    except Exception as e:
        LOGGER.error("Failed to load mask detection dataset: %s", str(e))
        raise e


# Segmentation Datasets
def load_segmentation_dataset(
    train: bool = True,
    batch_size: int = 8,
    shuffle: bool = False,
    pin_memory: bool = True,
    object_type: str = 'VisionData',
) -> Union[VisionData, 'torch.utils.data.DataLoader', Tuple[VisionData, VisionData]]:
    """
    Load a segmentation dataset from DeepChecks.
    
    Supported datasets: 'coco'
    
    Args:
        train: Whether to load training or validation dataset. Default is True.
        batch_size: How many samples per batch to load. Default is 8.
        shuffle: If True, reshuffles data at every epoch. Default is False.
        pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory 
                   before returning them. Default is True.
        object_type: Object type to return. If 'VisionData' then 
                    deepchecks.vision.VisionData will be returned, if 'DataLoader' then 
                    torch.utils.data.DataLoader. Default is 'VisionData'.
    
    Returns:
        Depending on the object_type parameter:
        - 'VisionData': deepchecks.vision.VisionData object
        - 'DataLoader': torch.utils.data.DataLoader object
    
    Raises:
        ImportError: If DeepChecks vision module is not installed
    """
    LOGGER.info("Loading segmentation dataset (train=%s, object_type=%s)", train, object_type)
    
    if segmentation_coco is None:
        raise ImportError("DeepChecks vision module not installed")
    
    try:
                
        if train and object_type == 'VisionData':
            # Load both train and val for VisionData
            train_data = segmentation_coco.load_dataset(
                train=True,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                object_type=object_type,
            )
            val_data = segmentation_coco.load_dataset(
                train=False,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                object_type=object_type,
            )
            return train_data, val_data
        else:
            return segmentation_coco.load_dataset(
                train=train,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=pin_memory,
                object_type=object_type,
            )
    except ValueError:
        raise
    except Exception as e:
        LOGGER.error("Failed to load segmentation dataset: %s", str(e))
        raise
