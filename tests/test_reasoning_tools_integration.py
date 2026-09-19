from __future__ import annotations

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.tavily import TavilyTools
from deepfix_server.agents.prompts import CROSS_ARTIFACT_SYSTEM_PROMPT
from deepfix_server.agents.reasoning import (
    CrossArtifactReasoningWorkflow,
    create_cross_artifact_reasoner,
    create_cross_artifact_reasoning_workflow,
)
from deepfix_server.agents.workflow import AnalysisWorkflow
from deepfix_server.engine import DiagnosticSystem
from deepfix_server.tools.search import get_search_tools


class TestCrossArtifactReasoningToolsIntegration:
    """Integration tests verifying CrossArtifactReasoningAgent tools attachment and prompt configuration."""

    def test_default_search_tools_attached(self):
        """Verify that create_cross_artifact_reasoner attaches DuckDuckGoTools by default."""
        agent = create_cross_artifact_reasoner()
        assert isinstance(agent, Agent)
        assert agent.tools is not None
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], DuckDuckGoTools)

    def test_custom_tools_override(self):
        """Verify that explicitly passed tools override default search tools."""
        custom_tool = DuckDuckGoTools()
        agent = create_cross_artifact_reasoner(tools=[custom_tool])
        assert agent.tools == [custom_tool]

    def test_empty_tools_override(self):
        """Verify that passing tools=[] explicitly disables tools."""
        agent = create_cross_artifact_reasoner(tools=[])
        assert agent.tools == []

    def test_tavily_tool_resolution_when_configured(self):
        """Verify TavilyTools is attached when configured or fallback if no key."""
        tools = get_search_tools(
            search_provider="tavily", tavily_api_key="test_dummy_key"
        )
        assert len(tools) == 1
        assert isinstance(tools[0], TavilyTools)

        agent = create_cross_artifact_reasoner(tools=tools)
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], TavilyTools)

    def test_cross_artifact_system_prompt_contains_tool_guidance(self):
        """Verify that CROSS_ARTIFACT_SYSTEM_PROMPT guides the agent on tool usage."""
        assert "External Search Tools & Domain Research" in CROSS_ARTIFACT_SYSTEM_PROMPT
        assert "DuckDuckGoTools" in CROSS_ARTIFACT_SYSTEM_PROMPT
        assert "When to Use Search" in CROSS_ARTIFACT_SYSTEM_PROMPT
        assert "How to Use Search" in CROSS_ARTIFACT_SYSTEM_PROMPT

    def test_workflow_propagates_tools_to_reasoning_chains(self):
        """Verify that CrossArtifactReasoningWorkflow propagates reasoner tools to candidate chains."""
        reasoner = create_cross_artifact_reasoner()
        assert len(reasoner.tools) == 1

        workflow = create_cross_artifact_reasoning_workflow(
            reasoner=reasoner,
            num_chains=2,
        )
        assert isinstance(workflow, CrossArtifactReasoningWorkflow)
        assert workflow.reasoner.tools == reasoner.tools

    def test_analysis_workflow_initializes_tools_for_reasoner(self):
        """Verify that AnalysisWorkflow equips reasoner with search tools."""
        workflow = AnalysisWorkflow(num_chains=2)
        assert workflow.reasoner.tools is not None
        assert len(workflow.reasoner.tools) == 1
        assert isinstance(workflow.reasoner.tools[0], DuckDuckGoTools)
        assert workflow.reasoning_workflow.reasoner.tools == workflow.reasoner.tools

    def test_diagnostic_system_initialization_without_knowledge_bridge(self):
        """Verify DiagnosticSystem initializes workflow and tools cleanly without knowledge_bridge."""
        system = DiagnosticSystem(num_chains=2)
        assert system.workflow is not None
        assert not hasattr(system, "knowledge_bridge")
        assert not hasattr(system.workflow, "knowledge_bridge")
        assert system.workflow.reasoner.tools is not None
        assert len(system.workflow.reasoner.tools) == 1
        assert isinstance(system.workflow.reasoner.tools[0], DuckDuckGoTools)

    def test_diagnostic_system_custom_tools_passed(self):
        """Verify DiagnosticSystem forwards explicit tools to workflow and reasoner."""
        custom_tool = DuckDuckGoTools()
        system = DiagnosticSystem(tools=[custom_tool], num_chains=2)
        assert system.workflow.reasoner.tools == [custom_tool]
        assert system.workflow.reasoning_workflow.reasoner.tools == [custom_tool]
