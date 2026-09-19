import pandas as pd
import pytest
from deepfix_core.models import DeepchecksConfig, DeepchecksParsedResult
from deepfix_sdk.integrations.deepchecks import DeepchecksRunnerForIR
from deepfix_sdk.ir.metrics import compute_ir_ranking_metrics


def test_compute_ir_ranking_metrics_perfect():
    """Verify metrics when relevant document is ranked #1."""
    qrels = pd.DataFrame([
        {"qid": "q1", "docno": "d1", "label": 1},
        {"qid": "q1", "docno": "d2", "label": 0},
    ])
    retrievals = pd.DataFrame([
        {"qid": "q1", "docno": "d1", "score": [0.1, 0.9], "rank": 0},
        {"qid": "q1", "docno": "d2", "score": [0.8, 0.2], "rank": 1},
    ])

    metrics = compute_ir_ranking_metrics(qrels_df=qrels, retrievals_df=retrievals, k=5)
    assert "nDCG@5" in metrics
    assert "MRR" in metrics
    assert "P@5" in metrics
    assert "R@5" in metrics

    assert metrics["MRR"] == pytest.approx(1.0, rel=1e-3)
    assert metrics["nDCG@5"] == pytest.approx(1.0, rel=1e-3)
    assert metrics["P@5"] == pytest.approx(0.2, rel=1e-3)  # 1 relevant in top 5
    assert metrics["R@5"] == pytest.approx(1.0, rel=1e-3)  # 1 / 1 relevant recalled


def test_compute_ir_ranking_metrics_second_rank():
    """Verify MRR when relevant document is ranked #2."""
    qrels = pd.DataFrame([
        {"qid": "q1", "docno": "d1", "label": 1},
        {"qid": "q1", "docno": "d2", "label": 0},
    ])
    retrievals = pd.DataFrame([
        {"qid": "q1", "docno": "d2", "score": 2.0, "rank": 1},  # 1-indexed rank
        {"qid": "q1", "docno": "d1", "score": 1.0, "rank": 2},
    ])

    metrics = compute_ir_ranking_metrics(qrels_df=qrels, retrievals_df=retrievals, k=5)
    assert metrics["MRR"] == pytest.approx(0.5, rel=1e-3)
    assert metrics["R@5"] == pytest.approx(1.0, rel=1e-3)


def test_compute_ir_ranking_metrics_no_match():
    """Verify metrics when no retrieved docs match relevance."""
    qrels = pd.DataFrame([
        {"qid": "q1", "docno": "d1", "label": 1},
    ])
    retrievals = pd.DataFrame([
        {"qid": "q1", "docno": "d99", "score": 1.0, "rank": 0},
    ])

    metrics = compute_ir_ranking_metrics(qrels_df=qrels, retrievals_df=retrievals, k=5)
    assert metrics["MRR"] == 0.0
    assert metrics["nDCG@5"] == 0.0
    assert metrics["P@5"] == 0.0
    assert metrics["R@5"] == 0.0


def test_runner_ir_ranking_integration(ir_data):
    """Verify DeepchecksRunnerForIR populates ir_ranking in results."""
    train_data, test_data, _ = ir_data
    assert test_data.retrievals is not None

    runner = DeepchecksRunnerForIR(
        config=DeepchecksConfig(
            train_test_validation=False,
            data_integrity=False,
            model_evaluation=False,
        )
    )

    artifact = runner.run_suites(
        train_data=train_data,
        test_data=test_data,
        dataset_name="test_ir_ranking",
    )

    assert "ir_ranking" in artifact.results
    ranking_checks = artifact.results["ir_ranking"]
    assert len(ranking_checks) == 1
    parsed = ranking_checks[0]
    assert isinstance(parsed, DeepchecksParsedResult)
    assert parsed.result.check == "Ranking Performance"
    assert "nDCG@5" in parsed.result.value
    assert "MRR" in parsed.result.value
    assert "P@5" in parsed.result.value
    assert "R@5" in parsed.result.value
