"""Cross-artifact reasoning agent for synthesizing findings from multiple analyzers.

Refactored to use Pydantic AI Agent with manual self-consistency loop,
replacing dspy.ReAct / dspy.ChainOfThought / dspy.MultiChainComparison.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Dict, Optional

from deepfix_kb import KnowledgeBridge
from deepfix_kb.tools import create_knowledge_tools

from pydantic_ai import Agent as PydanticAgent
from .schemas import CrossArtifactReasoningResult, AgentResult
from ..config import LLMConfig
from ..llm import create_model
from ..logging import get_logger
from .base import Agent

LOGGER = get_logger(__name__)


class CrossArtifactReasoningAgent(Agent):
    """Synthesizes findings from multiple artifact analyzer agents.

    Uses self-consistency: runs the analysis N times then consolidates
    the results into a final coherent output.

    Replaces DSPy's ReAct + MultiChainComparison pattern.
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        knowledge_bridge: Optional[KnowledgeBridge] = None,
        num_attempts: int = 3,
    ):
        """Initialize the cross-artifact reasoning agent.

        Args:
            llm_config: Optional LLM configuration.
            knowledge_bridge: Optional KnowledgeBridge for external knowledge.
                If provided, tools are registered with the Pydantic AI Agent.
            num_attempts: Number of self-consistency passes (default 3).
        """
        super().__init__(config=llm_config)
        self.knowledge_bridge = knowledge_bridge
        self.num_attempts = num_attempts      

        model = create_model(llm_config) if llm_config else None

        tools_list = []
        if self.knowledge_bridge and self.knowledge_bridge.has_available_sources:
            tools_list = create_knowledge_tools(self.knowledge_bridge, use_hybrid=False)

        self.agent = PydanticAgent(
            model=model,
            output_type=CrossArtifactReasoningResult,
            system_prompt=self.system_prompt,
            tools=tools_list,
        )

    async def arun(
        self,
        previous_analyses: Dict[str, AgentResult],
        output_language: str = "english",
    ) -> AgentResult:
        """Run with error handling."""
        try:
            return await self._acall(previous_analyses, output_language)
        except Exception as e:
            LOGGER.error(
                "Error with agent %s:\n%s", self.agent_name, traceback.format_exc()
            )
            return AgentResult(agent_name=self.agent_name, error_message=str(e))

    async def _acall(
        self,
        previous_analyses: Dict[str, AgentResult],
        output_language: str = "english",
    ) -> AgentResult:
        """Analyze and consolidate findings using self-consistency.

        Args:
            previous_analyses: Results from all artifact analyzers.
            output_language: Language for the output.

        Returns:
            AgentResult with consolidated cross-artifact analysis and summary.
        """
        LOGGER.debug("Running cross-artifact reasoning agent...")
        assert len(previous_analyses) > 0, "At least one analysis must be provided"

        # Build the reasoning prompt from previous analyses
        reasoning_prompt = self._build_reasoning_prompt(
            previous_analyses, output_language
        )

        # Run N self-consistency passes
        completions = []
        for i in range(self.num_attempts):
            LOGGER.debug("Self-consistency pass %d/%d", i + 1, self.num_attempts)
            result = await self.agent.run(reasoning_prompt)
            completions.append(result.output)

        # Consolidate: have the agent compare all completions and produce a final result
        consolidation_prompt = self._build_consolidation_prompt(
            completions, previous_analyses, output_language
        )
        final_result = await self.agent.run(consolidation_prompt)
        output_data = final_result.output

        # Collect analyzed artifacts and retrieved knowledge from inputs
        analyzed_artifacts: list = []
        retrieved_knowledge: list = []
        for result in previous_analyses.values():
            if result.analyzed_artifacts is not None:
                analyzed_artifacts.extend(result.analyzed_artifacts)
            if result.retrieved_knowledge is not None:
                retrieved_knowledge.extend(result.retrieved_knowledge)

        return AgentResult(
            agent_name=self.agent_name,
            analysis=final_result.output.analysis,
            analyzed_artifacts=analyzed_artifacts,
            retrieved_knowledge=retrieved_knowledge,
            additional_outputs={"summary": final_result.output.summary},
        )

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_reasoning_prompt(
        self,
        previous_analyses: Dict[str, AgentResult],
        output_language: str,
    ) -> str:
        """Build the prompt for the self-consistency reasoning passes."""
        # Serialize analyses to JSON-serializable dict
        analyses_dict = {}
        for name, ar in previous_analyses.items():
            entry: dict = {}
            if ar.analysis is not None:
                entry["analysis"] = [a.model_dump() for a in ar.analysis]
            if ar.error_message is not None:
                entry["error"] = ar.error_message
            analyses_dict[name] = entry

        analyses_json = json.dumps(analyses_dict, indent=2, default=str)

        return (
            "You are analyzing findings from multiple Machine Learning system analysis agents. "
            "Synthesize their individual findings into holistic insights.\n\n"
            f"## Previous Analyses\n\n{analyses_json}\n\n"
            f"Output language: {output_language}\n\n"
            "Produce a consolidated analysis with cross-artifact insights and a summary."
        )

    def _build_consolidation_prompt(
        self,
        completions: list,
        previous_analyses: Dict[str, AgentResult],
        output_language: str,
    ) -> str:
        """Build the prompt for the final consolidation pass."""
        comp_entries = []
        for i, comp in enumerate(completions):
            comp_entries.append(
                f"### Attempt {i + 1}\n{comp.model_dump_json(indent=2)}"
            )
        comp_text = "\n\n".join(comp_entries)

        return (
            "You ran cross-artifact reasoning multiple times with self-consistency. "
            "Below are the results from each attempt. "
            "Compare them, resolve any contradictions, and produce a single "
            "consolidated analysis that represents the best synthesis of all "
            "findings.\n\n"
            f"{comp_text}\n\n"
            f"Output language: {output_language}"
        )

    @property
    def system_prompt(self) -> str:
        return """You are an expert ML debugging and optimization consultant. You analyze and synthesize findings from multiple specialized agents to diagnose root causes and recommend actionable fixes.

                Your goal is to populate structured Analysis objects consisting of "Findings" and "Recommendations".

                ## 1. Cross-Artifact Synthesis Framework (Findings):
                When generating Findings, synthesize evidence across artifacts rather than just repeating individual agent outputs.
                - **Data-Performance Anomalies**: High performance with poor data quality suggests data leakage. Low performance with clean data points to model/hyperparameter mismatch.
                - **Training-Configuration Consistency**: Unstable curves despite conservative hyperparameters indicate dataset noise or bad loss formulation.
                - **Causal Chain Analysis**: Distinguish root causes (e.g., data leak) from symptoms (e.g., perfect validation accuracy).
                For each Finding, provide a clear description of the root cause, concrete evidence citing multiple agent results, and assign appropriate severity and confidence.

                ## 2. Optimization and Remediation Framework (Recommendations):
                For every Finding, you MUST provide a concrete Recommendation.
                - **Actionable Steps**: Provide precise action steps (e.g., specific hyperparameter adjustments, dataset filtering, augmentation techniques, or architecture changes). Avoid generic advice.
                - **Optimization Strategy**: Consider trade-offs between quick-win fixes and long-term improvements.
                - **Rationale**: Explain the rationale for why this action resolves the specific root cause and estimate the confidence in its success.

                ## Output Requirements:
                - Prioritize issues by their impact on model reliability and performance.
                - High-severity findings must have robust, cross-artifact evidence.
                - Do not hallucinate metrics; use only the data provided by the analysis agents.
                - Highlight critical deployment risks."""