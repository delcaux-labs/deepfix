import json
from pathlib import Path

import pytest
from deepfix_sdk.eval.loader import (
    filter_test_cases,
    load_benchmark_suite,
    load_manifest,
    normalize_tag,
)
from deepfix_sdk.eval.models import (
    BenchmarkManifest,
    BenchmarkTestCase,
    DefectVerificationResult,
)

MANIFESTS_DIR = Path(__file__).parent.parent / "packages" / "deepfix-sdk" / "benchmarks" / "manifests"


def test_benchmark_test_case_validation():
    """Verify BenchmarkTestCase schema validation and defaults."""
    tc = BenchmarkTestCase(
        id="test_01",
        name="Test Case 1",
        dataset_uri="s3://datasets/test.csv",
        expected_defects=["MULTICOLLINEARITY", "LEAKAGE"],
        target_metric="accuracy",
        target_value=0.95,
        max_iterations=4,
        timeout_seconds=180,
        tags=["tabular", "smoke"],
    )

    assert tc.id == "test_01"
    assert tc.name == "Test Case 1"
    assert tc.dataset_uri == "s3://datasets/test.csv"
    assert tc.model_uri is None
    assert tc.target_metric == "accuracy"
    assert tc.target_value == 0.95
    assert tc.max_iterations == 4
    assert tc.timeout_seconds == 180
    assert "tabular" in tc.tags
    assert len(tc.expected_defects) == 2


def test_defect_verification_logic():
    """Test defect verification matching against applied fixes."""
    tc = BenchmarkTestCase(
        id="defect_test",
        name="Defect Test",
        expected_defects=["MULTICOLLINEARITY", "LEAKAGE", "CLASS_IMBALANCE"],
    )

    # 1. Full match with various string formats
    applied_1 = [
        "Identified and removed multicollinearity via VIF thresholding",
        "LEAKAGE",
        "class_imbalance resolved by SMOTE oversampling",
    ]
    res_1 = tc.verify_addressed_defects(applied_1)
    assert isinstance(res_1, DefectVerificationResult)
    assert res_1.all_addressed is True
    assert set(res_1.addressed_defects) == {"MULTICOLLINEARITY", "LEAKAGE", "CLASS_IMBALANCE"}
    assert len(res_1.unaddressed_defects) == 0
    assert res_1.coverage_ratio == 1.0

    # 2. Partial match
    applied_2 = [
        "Removed multicollinear features",
    ]
    res_2 = tc.verify_addressed_defects(applied_2)
    assert res_2.all_addressed is False
    assert "MULTICOLLINEARITY" in res_2.addressed_defects
    assert "LEAKAGE" in res_2.unaddressed_defects
    assert "CLASS_IMBALANCE" in res_2.unaddressed_defects
    assert pytest.approx(res_2.coverage_ratio, 0.01) == 1 / 3

    # 3. Empty applied fixes
    res_3 = tc.verify_addressed_defects([])
    assert res_3.all_addressed is False
    assert len(res_3.addressed_defects) == 0
    assert len(res_3.unaddressed_defects) == 3
    assert res_3.coverage_ratio == 0.0

    # 4. Empty expected defects
    tc_no_defects = BenchmarkTestCase(id="no_defects", name="No defects")
    res_4 = tc_no_defects.verify_addressed_defects(["Fixed something"])
    assert res_4.all_addressed is True
    assert res_4.coverage_ratio == 1.0


def test_normalize_tag():
    """Test tag normalization."""
    assert normalize_tag("tabular") == "tabular"
    assert normalize_tag("Tag: Tabular ") == "tabular"
    assert normalize_tag("tag:LEAKAGE") == "leakage"
    assert normalize_tag("  smoke  ") == "smoke"


def test_filter_test_cases():
    """Test filtering test cases by tag expressions."""
    tc1 = BenchmarkTestCase(id="tc1", name="TC1", tags=["tabular", "smoke"], expected_defects=["MULTICOLLINEARITY"])
    tc2 = BenchmarkTestCase(id="tc2", name="TC2", tags=["tabular", "leakage"], expected_defects=["LEAKAGE"])
    tc3 = BenchmarkTestCase(id="tc3", name="TC3", tags=["vision"], expected_defects=["BLUR"])

    cases = [tc1, tc2, tc3]

    # Filter by direct tag
    assert filter_test_cases(cases, ["smoke"]) == [tc1]
    # Filter by prefix tag
    assert filter_test_cases(cases, ["tag: leakage"]) == [tc2]
    # Filter matching multiple
    assert len(filter_test_cases(cases, ["tabular"])) == 2
    # Filter matching defect tag
    assert filter_test_cases(cases, ["tag: MULTICOLLINEARITY"]) == [tc1]
    # Filter with no matching tags
    assert filter_test_cases(cases, ["nlp"]) == []
    # None or empty filter returns all
    assert filter_test_cases(cases, None) == cases
    assert filter_test_cases(cases, []) == cases


def test_load_canonical_manifests_exist_and_validate():
    """Verify all canonical benchmark manifests exist, parse, and validate."""
    expected_files = [
        "breast_cancer_multicollinearity.yaml",
        "breast_cancer_dataset_only.yaml",
        "synthetic_leakage.yaml",
        "imbalanced_credit.yaml",
        "canonical_suite.yaml",
    ]

    for fname in expected_files:
        manifest_path = MANIFESTS_DIR / fname
        assert manifest_path.exists(), f"Missing canonical manifest: {manifest_path}"
        manifest = load_manifest(manifest_path)
        assert isinstance(manifest, BenchmarkManifest)
        assert len(manifest.test_cases) >= 1
        for tc in manifest.test_cases:
            assert tc.id is not None
            assert tc.name is not None
            assert tc.target_metric is not None
            assert tc.target_value > 0.0
            assert tc.max_iterations > 0
            assert tc.timeout_seconds > 0


def test_load_single_case_yaml(tmp_path):
    """Test loading a standalone single test case manifest YAML."""
    yaml_content = """
id: "test_custom_manifest"
name: "Custom Test Manifest"
dataset_uri: "s3://bucket/data.csv"
target_metric: "f1_score"
target_value: 0.88
expected_defects:
  - "CLASS_IMBALANCE"
tags:
  - "finance"
"""
    test_file = tmp_path / "custom.yaml"
    test_file.write_text(yaml_content)

    manifest = load_manifest(test_file)
    assert manifest.suite_name == "test_custom_manifest"
    assert len(manifest.test_cases) == 1
    tc = manifest.test_cases[0]
    assert tc.id == "test_custom_manifest"
    assert tc.target_metric == "f1_score"
    assert tc.target_value == 0.88
    assert tc.expected_defects == ["CLASS_IMBALANCE"]


def test_load_json_manifest(tmp_path):
    """Test loading a manifest from JSON format."""
    data = {
        "version": "1.0",
        "suite_name": "json_suite",
        "test_cases": [
            {
                "id": "json_tc1",
                "name": "JSON TC1",
                "dataset_uri": "data/test.csv",
                "target_metric": "accuracy",
                "target_value": 0.90,
                "expected_defects": ["LEAKAGE"],
                "tags": ["smoke"],
            }
        ],
    }
    json_file = tmp_path / "suite.json"
    json_file.write_text(json.dumps(data))

    manifest = load_manifest(json_file)
    assert manifest.suite_name == "json_suite"
    assert len(manifest.test_cases) == 1
    assert manifest.test_cases[0].id == "json_tc1"


def test_load_manifest_errors(tmp_path):
    """Test error handling when loading non-existent or invalid manifest."""
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "non_existent.yaml")

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("invalid: - [")
    with pytest.raises(ValueError, match="Failed to parse benchmark manifest"):
        load_manifest(bad_yaml)

    not_a_dict = tmp_path / "list.yaml"
    not_a_dict.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="must be a dictionary"):
        load_manifest(not_a_dict)


def test_load_benchmark_suite_from_directory():
    """Test directory discovery and suite loading."""
    suite = load_benchmark_suite(MANIFESTS_DIR)
    assert len(suite.test_cases) >= 4
    tc_ids = {tc.id for tc in suite.test_cases}
    assert "breast_cancer_multicollinearity" in tc_ids
    assert "breast_cancer_dataset_only" in tc_ids
    assert "synthetic_leakage" in tc_ids
    assert "imbalanced_credit" in tc_ids

    # Filter directory load by tag
    leakage_suite = load_benchmark_suite(MANIFESTS_DIR, tags=["tag: leakage"])
    assert len(leakage_suite.test_cases) >= 1
    assert all("leakage" in [normalize_tag(t) for t in tc.tags] or "LEAKAGE" in tc.expected_defects for tc in leakage_suite.test_cases)
