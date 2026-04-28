import os
import re
import random
import itertools
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from deepfix_sdk import DeepFixClient
from deepfix_sdk.data.datasets import InformationRetrievalDataset
from deepfix_core.models import APIResponse
from deepfix_sdk.config import DeepchecksConfig


@pytest.fixture(autouse=True)
def setup_env():
    """Setup environment variables for the test."""
    os.environ["PYTHONIOENCODING"] = "utf-8"
    yield


def parse_cisi_docs(filepath):
    """Parse CISI.ALL file. Returns dict of {doc_id: full_text}."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    docs = {}
    raw_docs = re.split(r"\n\.I\s+", content)
    for raw in raw_docs:
        if not raw.strip():
            continue
        raw = raw.lstrip(".I ").strip()
        parts = raw.split("\n", 1)
        if len(parts) < 2:
            continue
        doc_id = int(parts[0].strip())
        body = parts[1]
        title_match = re.search(r"\.T\n(.*?)(?=\n\.[A-Z])", body, re.DOTALL)
        abstract_match = re.search(r"\.W\n(.*?)(?=\n\.[A-Z]|$)", body, re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        abstract = abstract_match.group(1).strip() if abstract_match else ""
        docs[doc_id] = (title + " " + abstract).strip()
    return docs


def parse_cisi_queries(filepath):
    """Parse CISI.QRY file. Returns dict of {query_id: query_text}."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    queries = {}
    raw_queries = re.split(r"\n\.I\s+", content)
    for raw in raw_queries:
        if not raw.strip():
            continue
        raw = raw.lstrip(".I ").strip()
        parts = raw.split("\n", 1)
        if len(parts) < 2:
            continue
        qid = int(parts[0].strip())
        body = parts[1]
        text_match = re.search(r"\.W\n(.*?)(?=\n\.[A-Z]|$)", body, re.DOTALL)
        if text_match:
            queries[qid] = text_match.group(1).strip()
    return queries


def parse_cisi_qrels(filepath):
    """Parse CISI.REL file. Returns dict of {query_id: set of relevant doc_ids}."""
    qrels = defaultdict(set)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                qid = int(parts[0])
                doc_id = int(parts[1])
                qrels[qid].add(doc_id)
    return dict(qrels)


def simulate_retrievals(qrels_list):
    data = []
    for qrel in qrels_list:
        # Simulate retrieval: only 80% of items are "found" by the model
        if random.random() > 0.2:
            prob = random.random()
            # If ground truth relevance is 1, give higher prob to class 1
            if qrel["relevance"] == 1:
                score = (
                    [1 - prob * 0.5, prob * 0.5]
                    if random.random() > 0.8
                    else [prob * 0.5, 1 - prob * 0.5]
                )
            else:
                score = [1 - prob * 0.2, prob * 0.2]

            data.append(
                {
                    "query_id": qrel["query_id"],
                    "entity_id": qrel["entity_id"],
                    "score": score,
                    "relevance": int(np.argmax(score)),
                    "rank": random.randint(1, 100),
                }
            )
    return pd.DataFrame(data)


def load_cisi_ir_data(data_dir: str, subset_queries: int = 10):
    """Load real CISI data and prepare IR datasets."""
    data_path = Path(data_dir)
    documents = parse_cisi_docs(data_path / "CISI.ALL")
    queries = parse_cisi_queries(data_path / "CISI.QRY")
    qrels = parse_cisi_qrels(data_path / "CISI.REL")

    # Filter to a subset of queries for faster testing
    qid_subset = sorted(list(qrels.keys()))[:subset_queries]
    qrels_subset = {qid: qrels[qid] for qid in qid_subset}

    corpus_dict = {str(did): {"text": text} for did, text in documents.items()}
    queries_dict = {
        str(qid): {"query": text} for qid, text in queries.items() if qid in qid_subset
    }

    qrels_list = []
    for qid, rel_docs in qrels_subset.items():
        if str(qid) not in queries_dict:
            continue
        valid_pos = [did for did in rel_docs if str(did) in corpus_dict]
        for _, doc_id in itertools.product([qid], valid_pos):
            qrels_list.append(
                {
                    "query_id": str(qid),
                    "entity_id": str(doc_id),
                    "rank": 1,
                    "relevance": 1,
                }
            )
        neg_candidates = list(set(documents.keys()) - set(rel_docs))
        if neg_candidates:
            # Add some negatives
            sampled_negs = random.sample(neg_candidates, min(5, len(neg_candidates)))
            for neg_id in sampled_negs:
                qrels_list.append(
                    {
                        "query_id": str(qid),
                        "entity_id": str(neg_id),
                        "rank": random.randint(1, 100),
                        "relevance": 0,
                    }
                )

    train_qrels, test_qrels = InformationRetrievalDataset.split_by_query(
        qrels_list, train_ratio=0.7, random_seed=42
    )

    train_ir_ds = InformationRetrievalDataset.from_ir_data(
        dataset_name="CISI_Train_E2E",
        queries=queries_dict,
        corpus=corpus_dict,
        qrels=train_qrels,
    )

    test_ir_ds = InformationRetrievalDataset.from_ir_data(
        dataset_name="CISI_Test_E2E",
        queries=queries_dict,
        corpus=corpus_dict,
        qrels=test_qrels,
    )

    train_retrievals = simulate_retrievals(train_qrels)
    test_retrievals = simulate_retrievals(test_qrels)

    train_ir_ds.set_predictions(train_retrievals)
    test_ir_ds.set_predictions(test_retrievals)

    return train_ir_ds, test_ir_ds


class TestIRWorkflowE2E:
    """End-to-end tests for the Information Retrieval (IR) workflow using real CISI data."""

    def test_ir_diagnosis_workflow(self, api_url: str, check_response: callable):
        """
        Test the full diagnosis workflow for an IR dataset using real CISI data.
        """
        # 1. Initialize Client
        print("1. Initializing client...")
        client = DeepFixClient(api_url=api_url, timeout=300)
        print("2. Client initialized.")

        # 2. Prepare Data
        print("3. Preparing real CISI IR data...")
        # Use relative path from project root
        cisi_path = os.path.join(os.getcwd(), "examples", "CISI")
        if not os.path.exists(cisi_path):
            # Fallback for different CWDs if needed, but typically it should be root
            pytest.skip(f"CISI data not found at {cisi_path}")

        train_data, test_data = load_cisi_ir_data(cisi_path, subset_queries=5)
        print(
            f"4. IR data prepared. Train samples: {len(train_data)}, Test samples: {len(test_data)}"
        )

        # 3. Run Diagnosis
        print("5. Running diagnosis...")
        response = client.get_diagnosis(
            train_data=train_data,
            test_data=test_data,
            language="english",
        )

        # 4. Verify Response
        print("6. Verifying response...")
        assert check_response(response)
