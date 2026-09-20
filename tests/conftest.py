import sys
import os
from pathlib import Path
import pytest


# Patch numpy early
try:
    import numpy as np

    if not hasattr(np, "Inf"):
        np.Inf = np.inf
    if not hasattr(np, "NINF"):
        np.NINF = -np.inf
except Exception:
    pass

import tempfile
from deepfix_core.models import APIResponse, DeepchecksConfig

@pytest.fixture
def minimal_deepchecks_config() -> DeepchecksConfig:
    """
    Fixture providing minimal DeepchecksConfig for quick test execution.
    Only enables train_test_validation suite.
    """
    return DeepchecksConfig(
        save_results=False,
        train_test_validation=True,
        data_integrity=True,
        model_evaluation=True,
        random_state=42,
    )


@pytest.fixture
def api_url():
    """Fixture providing the DeepFix API URL for tests."""
    url = os.getenv("DEEPFIX_SERVER_URL")
    if url is None:
        raise ValueError("DEEPFIX_SERVER_URL is not set")
    return url


@pytest.fixture
def deepfix_timeout():
    return int(os.getenv("DEEPFIX_TIMEOUT", "600"))


@pytest.fixture
def coco_detection_paths() -> dict[str, str]:
    """Fixture providing COCO detection dataset paths, skipping if not set."""
    paths = {
        "tr_images": os.getenv("TR_IMAGES_DIR_PATH"),
        "tr_annotations": os.getenv("TR_ANNOTATIONS_PATH"),
        "val_images": os.getenv("VAL_IMAGES_DIR_PATH"),
        "val_annotations": os.getenv("VAL_ANNOTATIONS_PATH"),
    }

    if not all(paths.values()):
        pytest.skip(
            "Object detection dataset paths (TR_IMAGES_DIR_PATH, TR_ANNOTATIONS_PATH, "
            "VAL_IMAGES_DIR_PATH, VAL_ANNOTATIONS_PATH) not fully set"
        )

    return paths


@pytest.fixture
def check_response():

    def check(response):
        assert isinstance(response, APIResponse), (
            "Response should be an APIResponse instance"
        )
        assert response.summary is not None, "Response should have a summary"
        assert len(response.agent_results) > 0, (
            "Response should contain results from agents"
        )

        print("\nDeepFix Analysis Summary:")
        print(response.to_text())

        return True

    return check

@pytest.fixture
def ir_data(subset_queries: int = 10):

    """Load BEIR scidocs data using PyTerrier and prepare IR datasets with real LlamaindexModel retrievals."""
    from deepfix_sdk.ir import InformationRetrievalDataset, LlamaindexModel
    import pyterrier as pt
    
    name = "irds:beir/scidocs"
    dataset = pt.get_dataset(name)

    # 1. Get all topics and qrels, subset for fast testing    
    all_topics = dataset.get_topics(variant="text")
    all_qrels = dataset.get_qrels()

    qid_subset = all_topics["qid"].unique()[:subset_queries]
    topics_df = all_topics[all_topics["qid"].isin(qid_subset)].copy()
    qrels_df = all_qrels[all_qrels["qid"].isin(qid_subset)].copy()

    needed_docnos = set(qrels_df["docno"].astype(str).unique())
    parent_corpus_iter = dataset.get_corpus_iter

    def subset_corpus_iter():
        count = 0
        for doc in parent_corpus_iter():
            if str(doc["docno"]) in needed_docnos:
                yield doc
                count += 1
                if count >= len(needed_docnos):
                    break

    # 2. Build a single dataset, then split using stratified sampling on labels
    ir_ds = InformationRetrievalDataset(
        dataset_name=name,
        topics=topics_df,
        enable_embedding_pca=True,
        embedding_pca_components=200,
        qrels=qrels_df,
        corpus_iter=subset_corpus_iter,
    )

    train_ir_ds, test_ir_ds = ir_ds.split(train_size=0.7, random_state=42)

    # 3. Generate real retrievals and embeddings using LlamaindexModel
    lancedb_index_dir = str(Path(__file__).parent / '.tmp_lancedb')
    model = LlamaindexModel(
        dataset=ir_ds,
        load_if_exists=True,
        lancedb_index_dir=lancedb_index_dir,
        top_k=5,
        retrieval_mode="dense",
    )
    model.fit()
    train_ir_ds.set_predictions(model.retrieve_dataframe(train_ir_ds,))
    test_ir_ds.set_predictions(model.retrieve_dataframe(test_ir_ds,))

    # 4. Set embeddings for diagnostic suites using LlamaindexModel
    train_ir_ds.set_embeddings(model.get_embedding)
    test_ir_ds.set_embeddings(model.get_embedding)

    return train_ir_ds, test_ir_ds, lancedb_index_dir


