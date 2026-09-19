"""Evaluation metrics for Information Retrieval tasks using PyTerrier."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pyterrier as pt
from pyterrier.measures import P, R, RR, nDCG

logger = logging.getLogger(__name__)


def compute_ir_ranking_metrics(
    qrels_df: pd.DataFrame,
    retrievals_df: pd.DataFrame,
    k: int = 5,
) -> Dict[str, float]:
    """Compute Learning-to-Rank (LTR) evaluation metrics using PyTerrier.

    Calculates:
        - nDCG@k: Normalized Discounted Cumulative Gain at rank cutoff k
        - MRR: Mean Reciprocal Rank (RR)
        - P@k: Precision at rank cutoff k
        - R@k: Recall at rank cutoff k

    Args:
        qrels_df: Ground truth DataFrame containing query-document relevance.
            Must contain ('qid' or 'query_id'), ('docno' or 'doc_id'),
            and ('label' or 'relevance').
        retrievals_df: Predicted retrieval DataFrame.
            Must contain ('qid' or 'query_id'), ('docno' or 'doc_id'),
            and optionally 'score' and 'rank'.
        k: Rank cutoff for evaluation (default: 5).

    Returns:
        Dictionary mapping metric names ("nDCG@k", "MRR", "P@k", "R@k")
        to float evaluation scores.
    """
    default_metrics = {
        f"nDCG@{k}": 0.0,
        "MRR": 0.0,
        f"P@{k}": 0.0,
        f"R@{k}": 0.0,
    }

    if qrels_df is None or qrels_df.empty or retrievals_df is None or retrievals_df.empty:
        logger.warning("Empty qrels or retrievals provided to compute_ir_ranking_metrics.")
        return default_metrics

    # 1. Standardize qrels DataFrame: qid, docno, label
    qrels = qrels_df.copy()
    qrels_rename = {}
    if "query_id" in qrels.columns and "qid" not in qrels.columns:
        qrels_rename["query_id"] = "qid"
    if "doc_id" in qrels.columns and "docno" not in qrels.columns:
        qrels_rename["doc_id"] = "docno"
    if "relevance" in qrels.columns and "label" not in qrels.columns:
        qrels_rename["relevance"] = "label"
    if qrels_rename:
        qrels = qrels.rename(columns=qrels_rename)

    qrels["qid"] = qrels["qid"].astype(str)
    qrels["docno"] = qrels["docno"].astype(str)
    qrels["label"] = qrels["label"].astype(int)
    qrels = qrels[["qid", "docno", "label"]]

    # 2. Standardize retrievals DataFrame: qid, docno, score, rank
    res = retrievals_df.copy()
    res_rename = {}
    if "query_id" in res.columns and "qid" not in res.columns:
        res_rename["query_id"] = "qid"
    if "doc_id" in res.columns and "docno" not in res.columns:
        res_rename["doc_id"] = "docno"
    if res_rename:
        res = res.rename(columns=res_rename)

    res["qid"] = res["qid"].astype(str)
    res["docno"] = res["docno"].astype(str)

    # Standardize score to a single float per row
    if "score" in res.columns:
        def _extract_score(val: Any) -> float:
            if isinstance(val, (list, tuple, np.ndarray)):
                if len(val) > 1:
                    return float(val[1])  # positive class probability
                elif len(val) == 1:
                    return float(val[0])
                return 0.0
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        res["score"] = res["score"].apply(_extract_score)
    else:
        # If no score column, derive descending scores from rank
        if "rank" in res.columns:
            res["score"] = -res["rank"].astype(float)
        else:
            res["score"] = 1.0

    # Ensure rank is 0-indexed integer per query
    if "rank" not in res.columns:
        res = res.sort_values(by=["qid", "score"], ascending=[True, False])
        res["rank"] = res.groupby("qid").cumcount()
    else:
        # Adjust rank if 1-indexed
        min_rank = res["rank"].min()
        if min_rank >= 1:
            res["rank"] = res["rank"].astype(int) - 1
        else:
            res["rank"] = res["rank"].astype(int)

    res = res[["qid", "docno", "score", "rank"]]

    measure_ndcg = nDCG @ k
    measure_rr = RR
    measure_p = P @ k
    measure_r = R @ k

    measures_list = [measure_ndcg, measure_rr, measure_p, measure_r]
    raw_results = pt.Evaluate(res, qrels, metrics=measures_list)

    def _get_metric_val(measure_obj: Any, aliases: list[str]) -> float:
        if measure_obj in raw_results:
            return float(raw_results[measure_obj])
        for alias in aliases:
            if alias in raw_results:
                return float(raw_results[alias])
        for k_key, v_val in raw_results.items():
            if str(k_key) == str(measure_obj) or str(k_key) in aliases:
                return float(v_val)
        return 0.0

    return {
        f"nDCG@{k}": _get_metric_val(measure_ndcg, [f"nDCG@{k}", f"ndcg_cut_{k}", f"NDCG@{k}"]),
        "MRR": _get_metric_val(measure_rr, ["RR", "MRR", "recip_rank"]),
        f"P@{k}": _get_metric_val(measure_p, [f"P@{k}", f"P_{k}", f"precision_{k}"]),
        f"R@{k}": _get_metric_val(measure_r, [f"R@{k}", f"recall_{k}", f"R_{k}"]),
    }
