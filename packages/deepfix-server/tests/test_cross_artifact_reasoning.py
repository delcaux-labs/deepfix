import pytest
from unittest.mock import MagicMock, patch
import dspy
from deepfix_kb import KnowledgeBridge
from deepfix_kb.tools import create_knowledge_tools
from deepfix_core.models import (
    AgentResult,
    Analysis,
    Finding,
    Recommendation,
    Severity,
)
from deepfix_server.agents.cross_artifact_reasoning import CrossArtifactReasoningAgent
from deepfix_server.config import LLMConfig

def test_agent_initialization_without_kb():
    """Test initialization when knowledge_bridge is None."""
    agent = CrossArtifactReasoningAgent(knowledge_bridge=None, num_attempts=5)
    assert agent.knowledge_bridge is None
    assert agent.num_attempts == 5
    # If no knowledge_bridge is provided, it should use ChainOfThought as predictor
    assert isinstance(agent.predict, dspy.ChainOfThought)
    # The compare module should be MultiChainComparison
    assert isinstance(agent.compare, dspy.MultiChainComparison)
    assert agent.compare.M == 5


def test_agent_initialization_with_kb():
    """Test initialization when knowledge_bridge is provided."""
    mock_kb = MagicMock(spec=KnowledgeBridge)
    agent = CrossArtifactReasoningAgent(knowledge_bridge=mock_kb, num_attempts=2)
    
    assert agent.knowledge_bridge == mock_kb
    assert agent.num_attempts == 2
    # If knowledge_bridge is provided, it should use ReAct as predictor
    assert isinstance(agent.predict, dspy.ReAct)
    assert isinstance(agent.compare, dspy.MultiChainComparison)
    assert agent.compare.M == 2
    
    # Verify create_knowledge_tools itself returns 3 tools (without hybrid)
    tools = create_knowledge_tools(mock_kb, include_hybrid=False)
    assert len(tools) == 3


@pytest.mark.asyncio
async def test_agent_forward_execution_with_self_consistency():
    """Test the agent forward/aforward execution with self-consistency."""
    num_attempts = 3
    agent = CrossArtifactReasoningAgent(knowledge_bridge=None, num_attempts=num_attempts)

    # Construct standard valid Analysis models
    dummy_analysis = Analysis(
        findings=Finding(
            description="Dummy description",
            evidence="Dummy evidence",
            severity=Severity.LOW,
            confidence=0.8
        ),
        recommendations=Recommendation(
            action="Dummy action",
            rationale="Dummy rationale",
            confidence=0.9
        )
    )

    # Mock responses for individual completions
    mock_completion_1 = MagicMock()
    mock_completion_2 = MagicMock()
    mock_completion_3 = MagicMock()

    # Mock response for the final comparison/aggregation
    mock_final_prediction = MagicMock()
    mock_final_prediction.analysis = [dummy_analysis]
    mock_final_prediction.summary = "Consolidated Summary"

    # We mock both predict.acall and compare.acall
    with patch.object(agent.predict, "acall", side_effect=[mock_completion_1, mock_completion_2, mock_completion_3]) as mock_predict_acall:
        with patch.object(agent.compare, "acall", return_value=mock_final_prediction) as mock_compare_acall:
            previous_analyses = {
                "analyzer_1": AgentResult(
                    agent_name="analyzer_1",
                    analysis=[],
                    analyzed_artifacts=["DatasetArtifacts"],
                    retrieved_knowledge=["Key Knowledge"]
                )
            }

            result = await agent.aforward(previous_analyses, output_language="english")

            assert result.agent_name == "CrossArtifactReasoningAgent"
            assert result.analysis == [dummy_analysis]
            assert result.analyzed_artifacts == ["DatasetArtifacts"]
            assert result.retrieved_knowledge == ["Key Knowledge"]
            assert result.additional_outputs == {"summary": "Consolidated Summary"}

            # Verify that predict was called exactly num_attempts (3) times
            assert mock_predict_acall.call_count == num_attempts
            for call in mock_predict_acall.call_args_list:
                assert call.kwargs == {"previous_analyses": previous_analyses, "output_language": "english"}

            # Verify that compare was called once with the completions
            mock_compare_acall.assert_called_once_with(
                previous_analyses=previous_analyses,
                output_language="english",
                completions=[mock_completion_1, mock_completion_2, mock_completion_3]
            )
