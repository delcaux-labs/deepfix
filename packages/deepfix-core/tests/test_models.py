import pytest
from deepfix_core.models import Finding, Severity, Recommendation

def test_finding_creation():
    finding = Finding(
        description="High loss detected",
        evidence="Loss is > 10.0",
        severity=Severity.HIGH,
        confidence=0.9
    )
    assert finding.description == "High loss detected"
    assert finding.severity == Severity.HIGH
    assert finding.confidence == 0.9

def test_recommendation_creation():
    rec = Recommendation(
        action="Lower learning rate",
        rationale="Learning rate is likely too high",
        expected_impact="More stable training",
        confidence=0.9
    )
    assert rec.action == "Lower learning rate"
    assert rec.rationale == "Learning rate is likely too high"
    assert rec.confidence == 0.9
