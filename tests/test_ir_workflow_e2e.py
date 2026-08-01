import os
import random
import pytest
import pandas as pd
import numpy as np
import pyterrier as pt
import hashlib
from pathlib import Path
from deepfix_sdk import DeepFixClient
from deepfix_sdk.data.datasets import InformationRetrievalDataset
from deepfix_core.models import APIResponse
from deepfix_sdk.config import DeepchecksConfig
from deepfix_sdk.models import IRLookupModel


def simulate_retrievals(
    qrels_df: pd.DataFrame, retrieval_rate: float = 0.8, seed: int = 42
) -> pd.DataFrame:
    """Simulate model retrievals from a qrels DataFrame.

    Args:
        qrels_df: DataFrame with ``query_id``, ``doc_id``, ``relevance`` columns.
        retrieval_rate: Fraction of qrel pairs "found" by the model.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n = len(qrels_df)

    # Only a fraction of items are "found" by the model
    retrieved_mask = rng.random(n) < retrieval_rate
    df = qrels_df[retrieved_mask].copy()
    m = len(df)

    prob = rng.random(m)
    is_relevant = df["relevance"].values == 1
    # 80% of relevant docs get a high class-1 score, 20% get a low one
    high_class1 = rng.random(m) > 0.2

    score_class0 = np.where(
        is_relevant,
        np.where(high_class1, prob * 0.5, 1 - prob * 0.5),
        1 - prob * 0.2,
    )
    score_class1 = np.where(
        is_relevant,
        np.where(high_class1, 1 - prob * 0.5, prob * 0.5),
        prob * 0.2,
    )

    return pd.DataFrame(
        {
            "query_id": df["query_id"].values,
            "doc_id": df["doc_id"].values,
            "score": [[s0, s1] for s0, s1 in zip(score_class0, score_class1)],
            "relevance": (score_class1 > score_class0).astype(int),
            "rank": rng.integers(1, 101, size=m),
        }
    )


def random_embedder(text: str, dim: int = 10) -> np.ndarray:
    """Compute a deterministic random embedding for a given text.

    Args:
        text: Input string.
        dim: Embedding dimension.
    """
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random(dim)


def load_ir_data(subset_queries: int = 50):
    """Load BEIR dbpedia-entity data using PyTerrier and prepare IR datasets."""
    name = "irds:beir/scidocs"
    dataset = pt.get_dataset(name)

    # 1. Get all topics and qrels, subset for fast testing
    all_topics = dataset.get_topics()
    all_qrels = dataset.get_qrels()

    qid_subset = all_topics["qid"].unique()[:subset_queries]
    topics_df = all_topics[all_topics["qid"].isin(qid_subset)]
    qrels_df = all_qrels[all_qrels["qid"].isin(qid_subset)]

    # 2. Build a single dataset, then split using stratified sampling on labels
    ir_ds = InformationRetrievalDataset(
        dataset_name=name,
        topics=topics_df,
        qrels=qrels_df,
        corpus_iter=dataset.get_corpus_iter,
    )

    train_ir_ds, test_ir_ds = ir_ds.split(train_size=0.7, random_state=42)

    # 3. Simulate retrievals and set predictions
    train_ir_ds.set_predictions(simulate_retrievals(train_ir_ds.qrels))
    test_ir_ds.set_predictions(simulate_retrievals(test_ir_ds.qrels))

    # 4. Set embeddings for diagnostic suites
    train_ir_ds.set_embeddings(random_embedder)
    test_ir_ds.set_embeddings(random_embedder)

    return train_ir_ds, test_ir_ds


class TestIRWorkflowE2E:
    """End-to-end tests for the Information Retrieval (IR) workflow using real CISI data."""
    def test_ir_model(self):
        """Verify that IRLookupModel correctly retrieves predictions and probabilities."""
        # 1. Load data
        train_data, test_data = load_ir_data(subset_queries=10)

        # 2. Initialize model
        model = IRLookupModel(train_dataset=train_data, test_dataset=test_data)

        # 3. Get metadata from test_data (where Deepchecks will look for X)
        X = test_data.metadata

        # 4. Predict
        preds = model.predict(X)
        assert len(preds) == len(test_data.predictions)
        # IRLookupModel returns strings to match Deepchecks expectations
        assert all(p == expected for p, expected in zip(preds, test_data.predictions))

        # 5. Predict Proba
        probas = model.predict_proba(X)
        assert probas.shape == (len(test_data.probabilities), 2)
        # Check if probabilities match
        for i, expected_proba in enumerate(test_data.probabilities):
            np.testing.assert_array_almost_equal(probas[i], expected_proba)

        # 6. Test KeyError for missing pairs
        missing_X = pd.DataFrame({"query_id": ["missing_q"], "doc_id": ["missing_d"]})
        with pytest.raises(KeyError, match="not found in lookup table"):
            model.predict(missing_X)

    def test_ir_diagnosis_workflow(self, api_url: str, deepfix_timeout: int, check_response: callable):
        """
        Test the full diagnosis workflow for an IR dataset using real CISI data.
        """
        # 1. Initialize Client
        print("1. Initializing client...")
        client = DeepFixClient(api_url=api_url, timeout=deepfix_timeout)
        print("2. Client initialized.")

        # 2. Prepare Data
        print("3. Preparing PyTerrier IR data...")
        train_data, test_data = load_ir_data()
        print(
            f"4. IR data prepared. Train samples: {len(train_data)}, Test samples: {len(test_data)}"
        )

        model = IRLookupModel(train_dataset=train_data, test_dataset=test_data)

        # 3. Run Diagnosis
        print("5. Running diagnosis...")
        response = client.get_diagnosis(
            train_data=train_data,
            test_data=test_data,
            model=model,
            model_name="random",
            language="english",
        )

        # 4. Verify Response
        print("6. Verifying response...")
        assert check_response(response)
