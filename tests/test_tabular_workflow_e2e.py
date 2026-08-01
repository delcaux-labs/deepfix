import os
import pytest
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from deepfix_sdk import DeepFixClient
from deepfix_sdk.data.datasets import TabularDataset
from deepfix_core.models import APIResponse


@pytest.fixture(autouse=True)
def setup_env():
    """Setup environment variables for the test."""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    yield
    # Cleanup if needed


class TestTabularWorkflowE2E:
    """End-to-end tests for the tabular workflow, reproducing the tutorial."""

    def test_tabular_diagnosis_workflow(self, api_url: str,deepfix_timeout: int, check_response: callable):
        """
        Test the full diagnosis workflow for a tabular dataset.
        Reproduces logic from tutorials/tabular.ipynb.
        """
        # 1. Initialize Client
        print("1. Initializing client...")
        client = DeepFixClient(api_url=api_url, timeout=deepfix_timeout)
        print("2. Client initialized.")

        # 2. Load and Prepare Data
        print("3. Loading and preparing data...")
        X, y = load_breast_cancer(as_frame=True, return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        dataset_name = "breast_cancer_classification_e2e"
        label = "target"

        train = X_train.copy()
        train[label] = y_train
        cat_features = X_train.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

        test = X_test.copy()
        test[label] = y_test

        train_data = TabularDataset(
            dataset=train,
            dataset_name=dataset_name,
            label=label,
            cat_features=cat_features,
        )
        val_data = TabularDataset(
            dataset=test,
            dataset_name=dataset_name,
            label=label,
            cat_features=cat_features,
        )
        print("4. Data loaded and prepared.")

        # 3. Fit Model
        print("5. Fitting model...")
        model_name = "HistGradientBoostingClassifier"
        clf = HistGradientBoostingClassifier(max_depth=3, random_state=42)
        clf.fit(train_data.X, train_data.y)
        print("6. Model fitted.")

        # 4. Run Diagnosis
        # This calls ingest() and then diagnose() internally
        print("7. Running diagnosis...")
        response = client.get_diagnosis(
            train_data=train_data,
            test_data=val_data,
            model_name=model_name,
            model=clf,
            language="english",
        )

        # 5. Verify Response
        print("8. Verifying response...")
        assert check_response(response)

    def test_tabular_diagnosis_without_model(
        self, api_url: str,deepfix_timeout: int, check_response: callable
    ):
        """
        Test the diagnosis workflow without providing a fitted model.
        The SDK should handle model fitting internally.
        """
        # 1. Initialize Client
        print("1. Initializing client...")
        client = DeepFixClient(api_url=api_url, timeout=deepfix_timeout)
        print("2. Client initialized.")

        # 2. Load and Prepare Data
        print("3. Loading and preparing data...")
        X, y = load_breast_cancer(as_frame=True, return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        dataset_name = "breast_cancer_classification_e2e_no_model"
        label = "target"

        train = X_train.copy()
        train[label] = y_train
        cat_features = X_train.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

        test = X_test.copy()
        test[label] = y_test

        train_data = TabularDataset(
            dataset=train,
            dataset_name=dataset_name,
            label=label,
            cat_features=cat_features,
        )
        val_data = TabularDataset(
            dataset=test,
            dataset_name=dataset_name,
            label=label,
            cat_features=cat_features,
        )
        print("4. Data loaded and prepared.")

        # 3. Run Diagnosis without providing a model
        print("5. Running diagnosis without model...")
        response = client.get_diagnosis(
            train_data=train_data,
            test_data=val_data,
            language="english",
        )

        # 4. Verify Response
        print("6. Verifying response...")
        assert check_response(response)
