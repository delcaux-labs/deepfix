import os
import pytest
from deepfix_sdk import DeepFixClient
from deepfix_sdk.data.datasets import NLPDataset
from deepfix_sdk.zoo.datasets import load_tweet_emotion_classification
from deepfix_core.models import APIResponse


@pytest.fixture(autouse=True)
def setup_env():
    """Setup environment variables for the test."""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    yield


class TestNLPWorkflowE2E:
    """End-to-end tests for the NLP workflow, reproducing the tutorial."""

    def test_nlp_diagnosis_workflow(self, api_url: str, check_response: callable,deepfix_timeout: int):
        """
        Test the full diagnosis workflow for an NLP dataset.
        Reproduces logic from tutorials/nlp.ipynb.
        """
        # 1. Initialize Client
        print("1. Initializing client...")
        client = DeepFixClient(api_url=api_url, timeout=deepfix_timeout)
        print("2. Client initialized.")

        # 2. Load and Prepare Data
        print("3. Loading and preparing data...")
        train_text_data, val_text_data = load_tweet_emotion_classification(
            as_train_test=True, include_embeddings=True
        )

        dataset_name = "tweet_emotion_classification_e2e"

        train_data = NLPDataset(dataset_name=dataset_name, dataset=train_text_data)
        val_data = NLPDataset(dataset_name=dataset_name, dataset=val_text_data)
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
