"""
End-to-end tests for Deepchecks runners.

Tests verify that runner classes execute successfully with representative datasets
and return valid DeepchecksArtifacts without errors.
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(__file__))
from deepfix_sdk.integrations.deepchecks import (
    DeepchecksRunnerForNLP,
    DeepchecksRunnerForTabular,
    DeepchecksRunnerForVision,
    DeepchecksRunnerForIR,
)
from deepfix_sdk.zoo.datasets.deepchecks_nlp import load_tweet_emotion_classification
from deepfix_sdk.zoo.datasets.deepchecks_tabular import load_adult_classification
from deepfix_sdk.zoo.datasets.deepchecks_vision import load_mnist_classification
from deepfix_core.models import DeepchecksArtifacts, DeepchecksConfig
from deepfix_sdk.models import IRLookupModel

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
        assert "train_test_validation" in artifact.results, (
            "Should contain train_test_validation results"
        )


class TestDeepchecksRunnerForTabular:
    """Tests for Deepchecks Tabular runner."""

    def test_run_suites_with_adult(self, minimal_deepchecks_config: DeepchecksConfig):
        """Test running Deepchecks suites on Adult tabular dataset."""
        # Load Adult dataset
        train_data, test_data = load_adult_classification(as_train_test=True)

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
        assert "train_test_validation" in artifact.results, (
            "Should contain train_test_validation results"
        )


class TestDeepchecksRunnerForText:
    """Tests for Deepchecks NLP Text runner."""

    def test_run_suites_with_tweet_emotion(
        self, minimal_deepchecks_config: DeepchecksConfig
    ):
        """Test running Deepchecks suites on Tweet Emotion NLP dataset."""
        # Load Tweet Emotion dataset
        train_data, test_data = load_tweet_emotion_classification(
            include_embeddings=True, as_train_test=True
        )

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
        assert artifact.dataset_name == "test_tweet_emotion", (
            "Dataset name should match"
        )
        assert artifact.results is not None, "Results should not be None"
        assert isinstance(artifact.results, dict), "Results should be a dictionary"
        assert len(artifact.results) > 0, "Results should contain suite results"
        assert "train_test_validation" in artifact.results, (
            "Should contain train_test_validation results"
        )


class TestDeepchecksRunnerForIR:
    """Tests for Deepchecks IR runner."""

    def test_run_suites_with_ir_data(self, minimal_deepchecks_config: DeepchecksConfig):
        """Test running Deepchecks suites on IR dataset."""
        from test_ir_workflow_e2e import load_ir_data
        

        # Load small IR dataset for fast testing
        train_data, test_data = load_ir_data()
        model = IRLookupModel(train_dataset=train_data, test_dataset=test_data)

        # Initialize runner with minimal config
        runner = DeepchecksRunnerForIR(config=minimal_deepchecks_config)

        # Run suites
        artifact = runner.run_suites(
            train_data=train_data,
            dataset_name="test_ir",
            test_data=test_data,
            model=model,
            model_name="random-predictions",
            train_predictions=train_data.predictions,
            test_predictions=test_data.predictions,
            train_probabilities=train_data.probabilities,
            test_probabilities=test_data.probabilities,
        )

        # Verify artifact structure
        assert isinstance(artifact, DeepchecksArtifacts), "Expected DeepchecksArtifacts"
        assert artifact.dataset_name == "test_ir", "Dataset name should match"
        assert artifact.results is not None, "Results should not be None"
        assert isinstance(artifact.results, dict), "Results should be a dictionary"
        assert len(artifact.results) > 0, "Results should contain suite results"
        
        # Verify it contains both NLP and Tabular results
        keys = list(artifact.results.keys())
        has_nlp = any(k.startswith("nlp_") for k in keys)
        has_tabular = any(k.startswith("tabular_") for k in keys)
        assert has_nlp, "Should contain NLP results"
        assert has_tabular, "Should contain Tabular results"

    
    def test_run_tabular_suite(self, minimal_deepchecks_config: DeepchecksConfig):
        from test_ir_workflow_e2e import load_ir_data
        
        # Load small IR dataset for fast testing
        train_data, test_data = load_ir_data()
        model = IRLookupModel(train_dataset=train_data, test_dataset=test_data)

        # Initialize runner with minimal config
        runner = DeepchecksRunnerForTabular(config=minimal_deepchecks_config)

        print(f"columns:{train_data.to_tabular().X.columns}")
        
        # Run suites (no test_data for this simple case)
        artifact = runner.run_suites(
            train_data=train_data.to_tabular(),
            dataset_name="test_ir",
            test_data=test_data.to_tabular(),
            model=model
        )

        # Verify artifact structure
        assert isinstance(artifact, DeepchecksArtifacts), "Expected DeepchecksArtifacts"
        assert artifact.dataset_name == "test_ir", "Dataset name should match"
        assert artifact.results is not None, "Results should not be None"
        assert isinstance(artifact.results, dict), "Results should be a dictionary"
        assert len(artifact.results) > 0, "Results should contain suite results"
        assert "train_test_validation" in artifact.results, (
            "Should contain train_test_validation results"
        )
