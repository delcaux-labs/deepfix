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
    agent = CrossArtifactReasoningAgent(knowledge_bridge=None)
    assert agent.knowledge_bridge is None
    # If no knowledge_bridge is provided, it should use ChainOfThought
    assert isinstance(agent.llm, dspy.ChainOfThought)


def test_agent_initialization_with_kb():
    """Test initialization when knowledge_bridge is provided."""
    mock_kb = MagicMock(spec=KnowledgeBridge)
    agent = CrossArtifactReasoningAgent(knowledge_bridge=mock_kb)
    
    assert agent.knowledge_bridge == mock_kb
    # If knowledge_bridge is provided, it should use ReAct
    assert isinstance(agent.llm, dspy.ReAct)
    
    # Verify create_knowledge_tools itself returns 3 tools (without hybrid)
    tools = create_knowledge_tools(mock_kb, include_hybrid=False)
    assert len(tools) == 3


@pytest.mark.asyncio
async def test_agent_forward_execution():
    """Test the agent forward/aforward execution with mocked LLM."""
    agent = CrossArtifactReasoningAgent(knowledge_bridge=None)

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

    # Mock the LLM call to return dummy analysis and summary
    mock_prediction = MagicMock()
    mock_prediction.analysis = [dummy_analysis]
    mock_prediction.summary = "Dummy Summary"

    # We mock the acall of ChainOfThought / ReAct
    with patch.object(agent.llm, "acall", return_value=mock_prediction) as mock_acall:
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
        assert result.additional_outputs == {"summary": "Dummy Summary"}

        mock_acall.assert_called_once_with(
            previous_analyses=previous_analyses,
            output_language="english"
        )
