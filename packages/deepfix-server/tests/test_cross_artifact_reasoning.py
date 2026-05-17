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
from deepfix_server.config import LLMConfig, settings

def test_agent_initialization_without_kb():
    """Test initialization when knowledge_bridge is None."""
    agent = CrossArtifactReasoningAgent(knowledge_bridge=None, num_attempts=5)
    assert agent.knowledge_bridge is None
    assert agent.num_attempts == 5
    # If no knowledge_bridge is provided, it should use ChainOfThought as predictor
    assert isinstance(agent.predict, dspy.ChainOfThought)
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

    # We mock predict.acall and the synchronous compare call
    with patch.object(agent.predict, "acall", side_effect=[mock_completion_1, mock_completion_2, mock_completion_3]) as mock_predict_acall:
        with patch.object(agent, "compare", return_value=mock_final_prediction) as mock_compare:
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
            mock_compare.assert_called_once_with(
                previous_analyses=previous_analyses,
                output_language="english",
                completions=[mock_completion_1, mock_completion_2, mock_completion_3]
            )


@pytest.mark.asyncio
async def test_agent_forward_execution_no_mock():
    """Test the agent forward/aforward execution with a live/configured LLM (no mocking)."""
    # Only run this test if an LLM API key is configured
    llm_config = settings.get_llm_config()
    if not llm_config.api_key:
        pytest.skip("No DEEPFIX_LLM_API_KEY configured in the environment or .env file.")

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
                        confidence=0.85
                    ),
                    recommendations=Recommendation(
                        action="Relabel the misclassified samples or use label smoothing.",
                        rationale="Correcting label errors directly improves target alignment.",
                        confidence=0.9
                    )
                )
            ],
            analyzed_artifacts=["DatasetArtifacts"],
            retrieved_knowledge=[]
        ),
        "training_dynamics_agent": AgentResult(
            agent_name="TrainingDynamicsAgent",
            analysis=[
                Analysis(
                    findings=Finding(
                        description="Training loss continues to fall but validation loss plateaus.",
                        evidence="Validation loss variance increases after epoch 15.",
                        severity=Severity.MEDIUM,
                        confidence=0.9
                    ),
                    recommendations=Recommendation(
                        action="Apply L2 regularization and implement early stopping.",
                        rationale="L2 regularization limits weight sizes, mitigating overfitting on noisy data.",
                        confidence=0.8
                    )
                )
            ],
            analyzed_artifacts=["TrainingArtifacts"],
            retrieved_knowledge=[]
        )
    }

    result = await agent.aforward(previous_analyses, output_language="english")

    assert result.agent_name == "CrossArtifactReasoningAgent"
    assert len(result.analysis) > 0
    assert "DatasetArtifacts" in result.analyzed_artifacts
    assert "TrainingArtifacts" in result.analyzed_artifacts
    assert result.additional_outputs.get("summary") is not None
    
    # Assert specific structure on consolidated findings
    first_analysis = result.analysis[0]
    assert first_analysis.findings.description is not None
    assert first_analysis.recommendations.action is not None
