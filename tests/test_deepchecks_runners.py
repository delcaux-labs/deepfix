"""
End-to-end tests for Deepchecks runners.

Tests verify that runner classes execute successfully with representative datasets
and return valid DeepchecksArtifacts without errors.
"""

import pytest
from deepfix.client.zoo.datasets.deepchecks_vision import load_mnist_classification
from deepfix.client.zoo.datasets.deepchecks_tabular import load_adult_classification
from deepfix.client.zoo.datasets.deepchecks_nlp import load_tweet_emotion_classification
from deepfix.shared.models import DeepchecksConfig, DeepchecksArtifacts

from deepfix.client.integrations.deepchecks import DeepchecksRunnerForVision, DeepchecksRunnerForTabular


class TestDeepchecksRunnerForVision:
    """Tests for Deepchecks Vision runner."""

    def test_run_suites_with_mnist(self, minimal_deepchecks_config: DeepchecksConfig):
        """Test running Deepchecks suites on MNIST vision dataset."""
        # Load MNIST dataset
        train_data, test_data = load_mnist_classification(
            train=True,
            n_samples=10,
            batch_size=4,
            object_type="VisionData",
            device="cpu",
        )
        
        # Initialize runner with minimal config
        runner = DeepchecksRunnerForVision(config=minimal_deepchecks_config)
        
        # Run suites
        artifact = runner.run_suites(
            train_data=train_data,
            dataset_name="test_mnist",
            test_data=test_data,
        )
        
        # Verify artifact structure
        assert isinstance(artifact, DeepchecksArtifacts), "Expected DeepchecksArtifacts"
        assert artifact.dataset_name == "test_mnist", "Dataset name should match"
        assert artifact.results is not None, "Results should not be None"
        assert isinstance(artifact.results, dict), "Results should be a dictionary"
        assert len(artifact.results) > 0, "Results should contain suite results"
        assert "train_test_validation" in artifact.results, "Should contain train_test_validation results"


class TestDeepchecksRunnerForTabular:
    """Tests for Deepchecks Tabular runner."""

    def test_run_suites_with_adult(self, minimal_deepchecks_config: DeepchecksConfig):
        """Test running Deepchecks suites on Adult tabular dataset."""
        # Load Adult dataset
        train_data,test_data = load_adult_classification(as_train_test=True)
        
        # Initialize runner with minimal config
        runner = DeepchecksRunnerForTabular(config=minimal_deepchecks_config)
        
        # Run suites (no test_data for this simple case)
        artifact = runner.run_suites(
            train_data=train_data,
            dataset_name="test_adult",
            test_data=test_data,
        )
        
        # Verify artifact structure
        assert isinstance(artifact, DeepchecksArtifacts), "Expected DeepchecksArtifacts"
        assert artifact.dataset_name == "test_adult", "Dataset name should match"
        assert artifact.results is not None, "Results should not be None"
        assert isinstance(artifact.results, dict), "Results should be a dictionary"
        assert len(artifact.results) > 0, "Results should contain suite results"
        assert "train_test_validation" in artifact.results, "Should contain train_test_validation results"


class TestDeepchecksRunnerForText:
    """Tests for Deepchecks NLP Text runner."""

    def test_run_suites_with_tweet_emotion(self, minimal_deepchecks_config: DeepchecksConfig):
        """Test running Deepchecks suites on Tweet Emotion NLP dataset."""
        # Load Tweet Emotion dataset
        train_data,test_data = load_tweet_emotion_classification(include_embeddings=False,as_train_test=True)
        
        # Initialize runner with minimal config
        runner = DeepchecksRunnerForNLP(config=minimal_deepchecks_config)
        
        # Run suites
        artifact = runner.run_suites(
            train_data=train_data,
            dataset_name="test_tweet_emotion",
            test_data=test_data,
        )
        
        # Verify artifact structure
        assert isinstance(artifact, DeepchecksArtifacts), "Expected DeepchecksArtifacts"
        assert artifact.dataset_name == "test_tweet_emotion", "Dataset name should match"
        assert artifact.results is not None, "Results should not be None"
        assert isinstance(artifact.results, dict), "Results should be a dictionary"
        assert len(artifact.results) > 0, "Results should contain suite results"
        assert "train_test_validation" in artifact.results, "Should contain train_test_validation results"

    def test_run_suite_data_integrity_not_implemented(self, minimal_deepchecks_config: DeepchecksConfig):
        """Test that data_integrity suite raises NotImplementedError for NLP runner."""
        train_data,test_data = load_tweet_emotion_classification(include_embeddings=False,as_train_test=True)
        runner = DeepchecksRunnerForNLP(config=minimal_deepchecks_config)
        
        # Verify that run_suite_data_integrity raises NotImplementedError
        with pytest.raises(NotImplementedError):
            runner.run_suite_data_integrity(train_data=train_data, test_data=test_data)
