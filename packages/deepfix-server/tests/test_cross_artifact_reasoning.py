"""Tests for CrossArtifactReasoningAgent — updated for Pydantic AI migration.

Previously tested dspy.ChainOfThought / dspy.ReAct / dspy.MultiChainComparison.
Now tests the Pydantic AI Agent with self-consistency.
"""

import pytest
from unittest.mock import MagicMock, patch

from deepfix_core.models import (
    AgentResult,
    Analysis,
    Finding,
    Recommendation,
    Severity,
)
from deepfix_server.agents.cross_artifact_reasoning import CrossArtifactReasoningAgent
from deepfix_server.config import LLMConfig, settings
from pydantic_ai import Agent as PydanticAgent


def test_agent_initialization_without_kb():
    """Test initialization when knowledge_bridge is None."""
    agent = CrossArtifactReasoningAgent(knowledge_bridge=None, num_attempts=5)
    assert agent.knowledge_bridge is None
    assert agent.num_attempts == 5
    # Agent should be a Pydantic AI Agent
    assert isinstance(agent.agent, PydanticAgent)


def test_agent_initialization_with_kb():
    """Test initialization when knowledge_bridge is provided."""
    mock_kb = MagicMock()
    agent = CrossArtifactReasoningAgent(knowledge_bridge=mock_kb, num_attempts=2)

    assert agent.knowledge_bridge == mock_kb
    assert agent.num_attempts == 2
    assert isinstance(agent.agent, PydanticAgent)


@pytest.mark.asyncio
async def test_agent_forward_execution_with_self_consistency():
    """Test the agent aforward execution with self-consistency."""
    num_attempts = 3
    agent = CrossArtifactReasoningAgent(knowledge_bridge=None, num_attempts=num_attempts)

    # Construct standard valid Analysis models
    dummy_analysis = Analysis(
        findings=Finding(
            description="Dummy description",
            evidence="Dummy evidence",
            severity=Severity.LOW,
            confidence=0.8,
        ),
        recommendations=Recommendation(
            action="Dummy action",
            rationale="Dummy rationale",
            confidence=0.9,
        ),
    )

    # Mock the run method to return our data
    from deepfix_server.agent_models import CrossArtifactReasoningResult

    mock_completion = CrossArtifactReasoningResult(
        analysis=[dummy_analysis], summary="Intermediate Summary"
    )
    mock_final = CrossArtifactReasoningResult(
        analysis=[dummy_analysis], summary="Consolidated Summary"
    )

    # We need to mock agent.run to return different values each call
    call_count = [0]

    async def mock_run(_user_message):
        call_count[0] += 1
        if call_count[0] <= num_attempts:
            # Return completion result for self-consistency passes
            result = MagicMock()
            result.output = mock_completion
            return result
        else:
            # Return final consolidation result
            result = MagicMock()
            result.output = mock_final
            return result

    with patch.object(agent.agent, "run", side_effect=mock_run):
        previous_analyses = {
            "analyzer_1": AgentResult(
                agent_name="analyzer_1",
                analysis=[],
                analyzed_artifacts=["DatasetArtifacts"],
                retrieved_knowledge=["Key Knowledge"],
            )
        }

        result = await agent.aforward(previous_analyses, output_language="english")

        assert result.agent_name == "CrossArtifactReasoningAgent"
        assert result.analysis == [dummy_analysis]
        assert result.analyzed_artifacts == ["DatasetArtifacts"]
        assert result.retrieved_knowledge == ["Key Knowledge"]
        assert result.additional_outputs == {"summary": "Consolidated Summary"}

        # Verify agent.run was called num_attempts + 1 times (passes + consolidation)
        assert call_count[0] == num_attempts + 1


@pytest.mark.asyncio
async def test_agent_forward_execution_live():
    """Test the agent aforward execution with a live/configured LLM (no mocking).

    Only runs if an LLM API key is configured.
    """
    llm_config = settings.get_llm_config()
    if not llm_config.api_key:
        pytest.skip(
            "No DEEPFIX_LLM_API_KEY configured in the environment or .env file."
        )

    agent = CrossArtifactReasoningAgent(llm_config=llm_config, num_attempts=1)

    previous_analyses = {
        "data_quality_agent": AgentResult(
            agent_name="DataQualityAgent",
            analysis=[
                Analysis(
                    findings=Finding(
                        description="Label noise detected in dataset labels.",
                        evidence="About 12% of the labels for class 'dog' are mismatched.",
                        severity=Severity.HIGH,
                        confidence=0.85,
                    ),
                    recommendations=Recommendation(
                        action="Relabel the misclassified samples or use label smoothing.",
                        rationale="Correcting label errors directly improves target alignment.",
                        confidence=0.9,
                    ),
                )
            ],
            analyzed_artifacts=["DatasetArtifacts"],
            retrieved_knowledge=[],
        ),
        "training_dynamics_agent": AgentResult(
            agent_name="TrainingDynamicsAgent",
            analysis=[
                Analysis(
                    findings=Finding(
                        description="Training loss continues to fall but validation loss plateaus.",
                        evidence="Validation loss variance increases after epoch 15.",
                        severity=Severity.MEDIUM,
                        confidence=0.9,
                    ),
                    recommendations=Recommendation(
                        action="Apply L2 regularization and implement early stopping.",
                        rationale="L2 regularization limits weight sizes, mitigating overfitting on noisy data.",
                        confidence=0.8,
                    ),
                )
            ],
            analyzed_artifacts=["TrainingArtifacts"],
            retrieved_knowledge=[],
        ),
    }

    try:
        result = await agent.aforward(previous_analyses, output_language="english")
    except Exception as exc:
        if "403" in str(exc) or "PermissionDeniedError" in type(exc).__name__:
            pytest.skip(f"API access denied (403): {exc}")
        raise

    assert result.agent_name == "CrossArtifactReasoningAgent"
    assert len(result.analysis) > 0
    assert "DatasetArtifacts" in result.analyzed_artifacts
    assert "TrainingArtifacts" in result.analyzed_artifacts
    assert result.additional_outputs.get("summary") is not None

    first_analysis = result.analysis[0]
    assert first_analysis.findings.description is not None
    assert first_analysis.recommendations.action is not None