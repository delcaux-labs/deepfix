from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import yaml

from .models import BenchmarkManifest, BenchmarkTestCase


def normalize_tag(tag: str) -> str:
    """Normalize a filter tag, stripping optional 'tag:' or 'tag: ' prefix."""
    t = tag.strip().lower()
    if t.startswith("tag:"):
        t = t[4:].strip()
    return t


def filter_test_cases(
    test_cases: List[BenchmarkTestCase],
    tags: Optional[List[str]] = None,
) -> List[BenchmarkTestCase]:
    """Filter a list of test cases by category/diagnostic tags."""
    if not tags:
        return list(test_cases)

    normalized_filter_tags = {
        normalize_tag(t) for t in tags if normalize_tag(t)
    }
    if not normalized_filter_tags:
        return list(test_cases)

    filtered: List[BenchmarkTestCase] = []
    for tc in test_cases:
        tc_tags = {normalize_tag(t) for t in tc.tags}
        for defect in tc.expected_defects:
            tc_tags.add(normalize_tag(defect))
        tc_tags.add(normalize_tag(tc.id))

        if any(f_tag in tc_tags for f_tag in normalized_filter_tags):
            filtered.append(tc)

    return filtered


def load_manifest(path: Union[str, Path]) -> BenchmarkManifest:
    """Load a benchmark manifest from a single YAML or JSON file."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Benchmark manifest file not found: {manifest_path}"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        content_str = f.read()

    try:
        if manifest_path.suffix.lower() == ".json":
            data = json.loads(content_str)
        else:
            data = yaml.safe_load(content_str)
    except Exception as exc:
        raise ValueError(
            f"Failed to parse benchmark manifest at {manifest_path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Manifest content in {manifest_path} must be a dictionary/mapping"
        )

    # Case 1: Full BenchmarkManifest with test_cases list
    if "test_cases" in data:
        return BenchmarkManifest.model_validate(data)

    # Case 2: Single BenchmarkTestCase
    if "id" in data:
        test_case = BenchmarkTestCase.model_validate(data)
        suite_name = data.get("suite_name") or test_case.id
        description = data.get("description") or test_case.description
        tags = data.get("tags") or test_case.tags
        return BenchmarkManifest(
            suite_name=suite_name,
            description=description,
            tags=tags,
            test_cases=[test_case],
            metadata={"source_file": str(manifest_path)},
        )

    raise ValueError(
        f"Unrecognized benchmark schema in {manifest_path}. "
        "Must contain either 'test_cases' or 'id'."
    )


def load_benchmark_suite(
    target: Union[str, Path],
    tags: Optional[List[str]] = None,
) -> BenchmarkManifest:
    """Load benchmark test cases from a file or directory of manifests."""
    target_path = Path(target)
    if not target_path.exists():
        raise FileNotFoundError(
            f"Benchmark target path not found: {target_path}"
        )

    all_test_cases: List[BenchmarkTestCase] = []
    seen_ids = set()
    suite_tags: List[str] = []
    suite_name = target_path.stem

    if target_path.is_file():
        manifest = load_manifest(target_path)
        all_test_cases.extend(manifest.test_cases)
        suite_tags.extend(manifest.tags)
        suite_name = manifest.suite_name
    elif target_path.is_dir():
        manifest_files = sorted(
            [
                p
                for p in target_path.rglob("*")
                if p.is_file()
                and p.suffix.lower() in {".yaml", ".yml", ".json"}
            ]
        )
        if not manifest_files:
            raise ValueError(
                f"No YAML/JSON manifests found in directory: {target_path}"
            )

        for m_file in manifest_files:
            manifest = load_manifest(m_file)
            for tc in manifest.test_cases:
                if tc.id not in seen_ids:
                    seen_ids.add(tc.id)
                    all_test_cases.append(tc)
            suite_tags.extend(manifest.tags)

    filtered_cases = filter_test_cases(all_test_cases, tags=tags)

    return BenchmarkManifest(
        suite_name=suite_name,
        description=f"Loaded from {target_path}",
        tags=list(set(suite_tags)),
        test_cases=filtered_cases,
        metadata={
            "target_path": str(target_path),
            "total_unfiltered": len(all_test_cases),
        },
    )
