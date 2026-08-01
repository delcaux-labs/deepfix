"""
End-to-end tests for Deepchecks dataloaders.

Tests verify that dataloader functions execute successfully and return
expected object types without errors.
"""

from deepchecks.nlp import TextData
from deepchecks.tabular import Dataset as TabularDataset
from deepchecks.vision import VisionData
from deepfix_sdk.zoo.datasets.deepchecks_nlp import load_tweet_emotion_classification
from deepfix_sdk.zoo.datasets.deepchecks_tabular import load_adult_classification
from deepfix_sdk.zoo.datasets.deepchecks_vision import load_mnist_classification
from torch.utils.data import DataLoader

class TestVisionDataloaders:
    """Tests for vision dataset loaders."""

    def test_load_mnist_classification_train_and_test(self):
        """Test loading MNIST classification dataset (train=True returns both train and test)."""
        result = load_mnist_classification(
            n_samples=10,
            batch_size=4,
            device="cpu",
        )

        assert isinstance(result, tuple), "Expected tuple of (train_data, test_data)"
        assert len(result) == 2, "Expected exactly 2 elements in tuple"

        train_data, test_data = result
        assert isinstance(train_data, DataLoader), "First element should be DataLoader"
        assert isinstance(test_data, DataLoader), "Second element should be DataLoader"
        assert len(train_data) > 0, "Training data should not be empty"
        assert len(test_data) > 0, "Test data should not be empty"


class TestTabularDataloaders:
    """Tests for tabular dataset loaders."""

    def test_load_adult_classification(self):
        """Test loading Adult classification dataset."""
        result = load_adult_classification()

        # The result should be a tuple of (train_data, test_data) or a single Dataset
        if isinstance(result, tuple):
            assert len(result) == 2, "Expected tuple of (train, test)"
            train_data, test_data = result
            assert hasattr(train_data, "data"), (
                "Training data should have 'data' attribute"
            )
            assert len(train_data.data) > 0, "Training data should contain samples"
            assert hasattr(test_data, "data"), "Test data should have 'data' attribute"
            assert len(test_data.data) > 0, "Test data should contain samples"
        else:
            # Single dataset
            assert isinstance(result, TabularDataset), (
                f"Expected TabularDataset object, got {type(result)}"
            )
            assert len(result) > 0, "Dataset should not be empty"

            # Verify dataset has required attributes
            assert hasattr(result, "data"), "Dataset should have 'data' attribute"
            assert result.data is not None, "Dataset data should not be None"
            assert len(result.data) > 0, "Dataset data should contain samples"


class TestNLPDataloaders:
    """Tests for NLP dataset loaders."""

    def test_load_tweet_emotion_classification(self):
        """Test loading Tweet Emotion classification dataset."""
        result = load_tweet_emotion_classification(
            include_embeddings=False,
        )

        assert isinstance(result, TextData), "Expected TextData object"
        assert len(result) > 0, "Dataset should not be empty"

        # Verify TextData has texts
        assert hasattr(result, "text"), "TextData should have 'texts' attribute"
        assert len(result.text) > 0, "TextData should contain texts"
