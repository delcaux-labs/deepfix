import tempfile
import numpy as np
import pandas as pd
import pytest
from deepfix_core.models import APIResponse
from deepfix_sdk import DeepFixClient
from deepfix_sdk.ir import (
    InformationRetrievalDataset,
    IRLookupModel,
    LlamaindexModel,
    RetrievalResult,
    RetrievalWorkflow,
)


class TestIRWorkflowE2E:
    """End-to-end tests for Information Retrieval (IR) workflows using real scidocs data."""

    @pytest.mark.asyncio
    async def test_retrieval_workflow_dense(self, ir_data):
        train_data, _, lancedb_index_dir = ir_data
        ir_dataset = train_data
        topics = ir_dataset.get_topics()
        q_col = "query" if "query" in topics.columns else "title" if "title" in topics.columns else "text"
        sample_query = str(topics.iloc[0][q_col])

        workflow = RetrievalWorkflow(
            dataset=ir_dataset,
            load_if_exists=True,
            lancedb_index_dir=lancedb_index_dir,
            top_k=1,
            retrieval_mode="dense",
        )

        # Ingest corpus documents into LanceDB index
        index = await workflow.run(ingest=True)
        assert index is not None

        # Retrieve candidate results
        results = await workflow.run(query=sample_query, index=index)
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)
        assert results[0].doc_id is not None
        assert len(results[0].text) > 0
        assert results[0].embedding is not None and len(results[0].embedding) > 0

        print("\n Results from Dense retrieval: \n", results)

    @pytest.mark.asyncio
    async def test_retrieval_workflow_hybrid(self, ir_data):
        train_data, _, lancedb_index_dir = ir_data
        ir_dataset = train_data
        topics = ir_dataset.get_topics()
        q_col = "query" if "query" in topics.columns else "title" if "title" in topics.columns else "text"
        sample_query = str(topics.iloc[0][q_col])

        workflow = RetrievalWorkflow(
            dataset=ir_dataset,
            load_if_exists=True,
            lancedb_index_dir=lancedb_index_dir,
            top_k=1,
            retrieval_mode="hybrid",
            dense_weight=0.5,
            bm25_weight=0.5,
        )

        index = await workflow.run(ingest=True)
        assert index is not None

        results = await workflow.run(query=sample_query, index=index)
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], RetrievalResult)
        assert results[0].doc_id is not None
        assert len(results[0].text) > 0

        print("\n Results from Hybrid retrieval: \n", results)

    def test_llamaindex_model_sklearn_interface(self, ir_data):
        ir_dataset, _, lancedb_index_dir = ir_data
        model = LlamaindexModel(
            dataset=ir_dataset,
            load_if_exists=True,
            lancedb_index_dir=lancedb_index_dir,
            top_k=1,
            retrieval_mode="dense",
        )
        model.fit()
        assert model.is_fitted_

        # 1. Test retrieve method
        topics = ir_dataset.get_topics()
        q_col = "query" if "query" in topics.columns else "title" if "title" in topics.columns else "text"
        sample_query = str(topics.iloc[0][q_col])
        retrievals = model.retrieve(sample_query)
        assert len(retrievals) > 0
        assert isinstance(retrievals[0], RetrievalResult)
        assert retrievals[0].embedding is not None and len(retrievals[0].embedding) > 0

        # 2. Test predict and predict_proba on metadata (query_id, doc_id pairs)
        X = ir_dataset.metadata.copy()
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert set(np.unique(preds)).issubset({0, 1})

        probas = model.predict_proba(X)
        assert probas.shape == (len(X), 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)

        # 3. Test retrieve_dataframe and compatibility with set_predictions
        retrieval_df = model.retrieve_dataframe(ir_dataset, top_k=1)
        assert "query_id" in retrieval_df.columns
        assert "doc_id" in retrieval_df.columns
        assert "score" in retrieval_df.columns
        assert "rank" in retrieval_df.columns
        assert "relevance" in retrieval_df.columns
        assert len(retrieval_df) > 0

        # 4. Verify it integrates with InformationRetrievalDataset.set_predictions
        ir_dataset.set_predictions(retrieval_df)
        assert len(ir_dataset.predictions) == len(ir_dataset.qrels)
        assert len(ir_dataset.probabilities) == len(ir_dataset.qrels)
        print("\n Results from sklearn dense retrieval: ", retrievals)

    def test_ir_model(self, ir_data):
        """Verify that IRLookupModel correctly retrieves predictions and probabilities."""
        # 1. Load data
        train_data, test_data,_ = ir_data

        # 2. Initialize model
        model = IRLookupModel(train_dataset=train_data, test_dataset=test_data)

        # 3. Get metadata from test_data (where Deepchecks will look for X)
        X = test_data.metadata

        # 4. Predict
        preds = model.predict(X)
        assert len(preds) == len(test_data.predictions)
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

    def test_ir_diagnosis_workflow(
        self, ir_data, api_url: str, deepfix_timeout: int, check_response: callable
    ):
        """Test the full diagnosis workflow for an IR dataset."""
        # 1. Initialize Client
        print("1. Initializing client...")
        client = DeepFixClient(api_url=api_url, timeout=deepfix_timeout)
        print("2. Client initialized.")

        # 2. Prepare Data
        print("3. Preparing PyTerrier IR data...")
        train_data, test_data, _ = ir_data
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

    def test_ir_diagnosis_workflow_llamaindex(
        self, ir_data, api_url: str, deepfix_timeout: int, check_response: callable
    ):
        """Test the full diagnosis workflow for an IR dataset using LlamaindexModel."""
        # 1. Initialize Client
        print("1. Initializing client...")
        client = DeepFixClient(api_url=api_url, timeout=deepfix_timeout)
        print("2. Client initialized.")

        # 2. Prepare Data
        print("3. Preparing PyTerrier IR data...")
        train_data, test_data, lancedb_index_dir = ir_data
        print(
            f"4. IR data prepared. Train samples: {len(train_data)}, Test samples: {len(test_data)}"
        )

        model = LlamaindexModel(
            dataset=train_data,
            load_if_exists=True,
            lancedb_index_dir=lancedb_index_dir,
            top_k=1,
            retrieval_mode="dense",
        )
        model.fit()

        # 3. Run Diagnosis
        print("5. Running diagnosis with LlamaindexModel...")
        response = client.get_diagnosis(
            train_data=train_data,
            test_data=test_data,
            model=model,
            model_name="llamaindex",
            language="english",
        )

        # 4. Verify Response
        print("6. Verifying response...")
        assert check_response(response)


