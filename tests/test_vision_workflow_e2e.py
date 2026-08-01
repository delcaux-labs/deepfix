import os
import pytest
import sys

# Ensure UTF-8 encoding for Windows terminal
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ["PYTHONIOENCODING"] = "utf-8"

from deepfix_sdk import DeepFixClient
from deepfix_sdk.data.datasets import (
    ImageClassificationDataset,
    ObjectDetectionDataset,
    SemanticSegmentationDataset,
    VisionDataset,
)
from deepfix_sdk.zoo.datasets.deepchecks_vision import (
    load_mnist_classification,
    load_coco_detection,
    load_segmentation_dataset,
)
from deepfix_sdk.zoo.datasets.foodwaste import load_train_and_val_datasets
from deepfix_core.models import APIResponse


@pytest.fixture(autouse=True)
def setup_env():
    """Setup environment variables for the test."""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    yield


import torch
from torch.utils.data import Dataset


class TransposeMNIST(Dataset):
    """Wrapper to transpose MNIST images from (C, H, W) to (H, W, C)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # MNIST images are usually (1, 28, 28) tensors
        if hasattr(image, "permute"):
            image = image.permute(1, 2, 0).numpy()
        return image, label


class TestVisionWorkflowE2E:
    """End-to-end tests for the Computer Vision workflow, reproducing the tutorial."""

    def test_foodwaste_classification_workflow(
        self, api_url: str, check_response: callable, deepfix_timeout: int
    ):
        """
        Test the full diagnosis workflow for the foodwaste dataset.
        Reproduces logic from tutorials/computer-vision.ipynb.
        """
        # 1. Initialize Client
        print("1. Initializing client...")
        client = DeepFixClient(api_url=api_url, timeout=deepfix_timeout)
        print("2. Client initialized.")

        # 2. Load and Prepare Data
        print("3. Loading and preparing data...")
        dataset_name = "cafetaria-foodwaste-lstroetmann"
        train_data_raw, val_data_raw = load_train_and_val_datasets(
            image_size=448,
            batch_size=8,
            num_workers=0,
            pin_memory=False,
        )

        train_data = ImageClassificationDataset(
            dataset_name=dataset_name, dataset=train_data_raw
        )
        val_data = ImageClassificationDataset(
            dataset_name=dataset_name, dataset=val_data_raw
        )
        print("4. Data loaded and prepared.")

        # 3. Run Diagnosis
        print("5. Running diagnosis...")
        response = client.get_diagnosis(
            train_data=train_data,
            test_data=val_data,
            language="english",
        )

        # 4. Verify Response
        print("6. Verifying response...")
        assert check_response(response)

    def test_semantic_segmentation_workflow(
        self, api_url: str, check_response: callable, deepfix_timeout: int
    ):
        """Test the semantic segmentation workflow using COCO segmentation dataset."""
        print("\nRunning Semantic Segmentation workflow...")
        client = DeepFixClient(api_url=api_url, timeout=deepfix_timeout)

        dataset_name = "coco_segmentation"
        train_data, val_data = load_segmentation_dataset(
            batch_size=8,
            shuffle=False,
            pin_memory=False,
        )
        train_data = SemanticSegmentationDataset(
            dataset_name=dataset_name, dataset=train_data.dataset
        )
        val_data = SemanticSegmentationDataset(
            dataset_name=dataset_name, dataset=val_data.dataset
        )
        print(f"Executing diagnosis for {dataset_name}...")
        result = client.get_diagnosis(
            train_data=train_data,
            test_data=val_data,
            language="english",
        )

        assert check_response(result)

    def test_object_detection_workflow(
        self, api_url: str, coco_detection_paths: dict, check_response: callable, deepfix_timeout: int
    ):
        """Test the object detection workflow using COCO detection dataset."""
        print("\nRunning Object Detection workflow...")
        client = DeepFixClient(api_url=api_url, timeout=deepfix_timeout)

        print("Coco detection paths: ", coco_detection_paths)

        dataset_name = "coco_detection_dataset"
        train_data = ObjectDetectionDataset.from_coco(
            dataset_name=dataset_name,
            images_directory_path=coco_detection_paths["tr_images"],
            annotations_path=coco_detection_paths["tr_annotations"],
        )
        val_data = ObjectDetectionDataset.from_coco(
            dataset_name=dataset_name,
            images_directory_path=coco_detection_paths["val_images"],
            annotations_path=coco_detection_paths["val_annotations"],
        )
        print(f"Executing diagnosis for {dataset_name}...")
        result = client.get_diagnosis(
            train_data=train_data,
            test_data=val_data,
            language="english",
        )

        assert check_response(result)
