from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DefectVerificationResult(BaseModel):
    """Outcome of verifying applied fixes against expected diagnostic defect tags."""

    addressed_defects: List[str] = Field(
        default_factory=list,
        description="Defect tags that were addressed by the applied fixes",
    )
    unaddressed_defects: List[str] = Field(
        default_factory=list,
        description="Expected defect tags that were not addressed",
    )
    all_addressed: bool = Field(
        default=True,
        description="Whether all expected defects were addressed",
    )
    coverage_ratio: float = Field(
        default=1.0,
        description="Proportion of expected defects addressed (0.0 to 1.0)",
    )


def _get_word_stem(word: str) -> str:
    """Extract linguistic stem from a word by stripping known suffixes."""
    w = word.lower()
    suffixes = (
        "linearity",
        "linear",
        "leakage",
        "leaking",
        "imbalance",
        "imbalanced",
        "preprocessing",
        "processing",
        "processed",
        "scaling",
        "scaled",
        "fitting",
        "overfitting",
        "correlation",
        "correlated",
        "sampling",
        "sampled",
        "weights",
        "weighted",
        "reduction",
        "ity",
        "tion",
        "ing",
        "ed",
        "es",
        "s",
    )
    for suffix in suffixes:
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _is_defect_addressed(
    defect: str,
    normalized_fixes: List[str],
    raw_fixes_joined: str,
    raw_stems: set[str],
) -> bool:
    """Check if a specific diagnostic defect tag is addressed by applied fixes."""
    defect_clean = re.sub(r"[\s_-]+", "", defect.lower())
    defect_words = [w.lower() for w in re.split(r"[\s_-]+", defect) if len(w) > 2]
    defect_stems = [_get_word_stem(w) for w in defect_words]

    for norm_fix in normalized_fixes:
        if defect_clean in norm_fix or norm_fix in defect_clean:
            return True

    if defect_words:
        if any(w in raw_fixes_joined for w in defect_words):
            return True
        if any(s in raw_fixes_joined or s in raw_stems for s in defect_stems):
            return True

    return False


class BenchmarkTestCase(BaseModel):
    """Specification of a single autonomous fix benchmark test case."""

    id: str = Field(
        ...,
        description="Unique identifier for the benchmark test case",
    )
    name: str = Field(
        ...,
        description="Human-readable title for the test case",
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed description of defect scenario and objectives",
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Dataset name registered in MLflow / DeepFix or catalog",
    )
    dataset_uri: Optional[str] = Field(
        default=None,
        description="URI or path to dataset (local, MLflow, S3, or HF dataset ID)",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Optional baseline model name or identifier",
    )
    model_uri: Optional[str] = Field(
        default=None,
        description="Optional baseline model URI or artifact location",
    )
    target_metric: str = Field(
        default="accuracy",
        description="Target metric key to optimize (accuracy, f1_score, roc_auc)",
    )
    target_value: float = Field(
        default=0.90,
        description="Target metric threshold value required for success",
    )
    max_iterations: int = Field(
        default=5,
        description="Maximum autonomous refinement iterations allowed",
    )
    expected_defects: List[str] = Field(
        default_factory=list,
        description="Diagnostic defect tags expected to be resolved",
    )
    timeout_seconds: int = Field(
        default=300,
        description="Execution timeout in seconds for this benchmark case",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Category and filtering tags (e.g., tabular, leakage, smoke)",
    )
    s3_bucket: Optional[str] = Field(
        default=None,
        description="Optional target S3 bucket for model weights or artifacts",
    )
    diagnosis: Optional[str] = Field(
        default=None,
        description="Pre-computed diagnostic findings injected into the fix agent",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra metadata and hyperparameters",
    )

    def verify_addressed_defects(
        self, applied_fixes: List[str]
    ) -> DefectVerificationResult:
        """Verify whether applied fixes addressed the expected diagnostic defects."""
        if not self.expected_defects:
            return DefectVerificationResult(
                addressed_defects=[],
                unaddressed_defects=[],
                all_addressed=True,
                coverage_ratio=1.0,
            )

        normalized_fixes = [
            re.sub(r"[\s_-]+", "", fix.lower())
            for fix in applied_fixes
            if isinstance(fix, str)
        ]
        raw_fixes_joined = " ".join(str(f).lower() for f in applied_fixes)
        raw_words = [
            w.lower()
            for w in re.split(r"[\s_\W]+", raw_fixes_joined)
            if len(w) > 2
        ]
        raw_stems = {_get_word_stem(w) for w in raw_words}

        addressed: List[str] = []
        unaddressed: List[str] = []

        for defect in self.expected_defects:
            if _is_defect_addressed(
                defect, normalized_fixes, raw_fixes_joined, raw_stems
            ):
                addressed.append(defect)
            else:
                unaddressed.append(defect)

        coverage = (
            len(addressed) / len(self.expected_defects)
            if self.expected_defects
            else 1.0
        )

        return DefectVerificationResult(
            addressed_defects=addressed,
            unaddressed_defects=unaddressed,
            all_addressed=len(unaddressed) == 0,
            coverage_ratio=coverage,
        )


class BenchmarkManifest(BaseModel):
    """Declarative benchmark suite manifest containing one or more test cases."""

    version: str = Field(
        default="1.0",
        description="Manifest schema version",
    )
    suite_name: str = Field(
        ...,
        description="Name of the benchmark suite",
    )
    description: Optional[str] = Field(
        default=None,
        description="Overview of the benchmark suite objectives and scenarios",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Suite-level tags for categorization and filtering",
    )
    test_cases: List[BenchmarkTestCase] = Field(
        default_factory=list,
        description="List of benchmark test cases in this suite",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra suite-level metadata",
    )


class DimensionScore(BaseModel):
    """Grading details for a single LLM-as-a-judge rubric dimension."""

    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Score on a 1-5 integer scale",
    )
    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weight of this dimension in the overall score calculation",
    )
    feedback: str = Field(
        ...,
        description="Qualitative justification and critique for this dimension",
    )


class JudgeVerdict(BaseModel):
    """Structured evaluation report returned by the LLM-as-a-Judge engine."""

    test_case_id: str = Field(
        ...,
        description="Identifier of the evaluated benchmark test case",
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite score normalized from 0.0 to 100.0 across dimensions",
    )
    pass_status: bool = Field(
        ...,
        description="Boolean indicating whether the fix passed evaluation criteria",
    )
    dimension_scores: Dict[str, DimensionScore] = Field(
        default_factory=dict,
        description="Breakdown of individual rubric dimension scores",
    )
    detected_failure_modes: List[str] = Field(
        default_factory=list,
        description="Identified flaws (e.g. DATA_LEAKAGE, THRASHING, SYNTAX_ERROR)",
    )
    rationale: str = Field(
        ...,
        description="Executive summary and rationale behind the judging verdict",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations for improvement",
    )
