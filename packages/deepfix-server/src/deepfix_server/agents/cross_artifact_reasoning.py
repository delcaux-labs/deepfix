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

from ..config import LLMConfig
from ..llm import create_model
from ..logging import get_logger
from .base import Agent, AgentResult

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

        # Build the Pydantic AI Agent for cross-artifact reasoning
        from ..agent_models import CrossArtifactReasoningResult

        from pydantic_ai import Agent as PydanticAgent

        model = create_model(llm_config) if llm_config else None

        tools_list = []
        if self.knowledge_bridge:
            from deepfix_kb.tools import create_knowledge_tools

            tools_list = create_knowledge_tools(self.knowledge_bridge, include_hybrid=False)

        self.agent = PydanticAgent(
            model=model,
            result_type=CrossArtifactReasoningResult,
            system_prompt=self.system_prompt,
            tools=tools_list,
        )

    def run(
        self,
        previous_analyses: Dict[str, AgentResult],
        output_language: str = "english",
    ) -> AgentResult:
        """Run synchronously."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run, self.arun(previous_analyses, output_language)
            )
            return future.result()

    async def arun(
        self,
        previous_analyses: Dict[str, AgentResult],
        output_language: str = "english",
    ) -> AgentResult:
        """Run with error handling."""
        try:
            return await self.acall(previous_analyses, output_language)
        except Exception as e:
            LOGGER.error(
                "Error with agent %s:\n%s", self.agent_name, traceback.format_exc()
            )
            return AgentResult(agent_name=self.agent_name, error_message=str(e))

    async def acall(
        self,
        previous_analyses: Dict[str, AgentResult],
        output_language: str = "english",
    ) -> AgentResult:
        """Alias for aforward."""
        return await self.aforward(previous_analyses, output_language)

    def forward(
        self,
        previous_analyses: Dict[str, AgentResult],
        output_language: str = "english",
    ) -> AgentResult:
        """Synchronous forward."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run, self.aforward(previous_analyses, output_language)
            )
            return future.result()

    async def aforward(
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
        LOGGER.info("Running cross-artifact reasoning agent...")
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
            completions.append(result.data)

        # Consolidate: have the agent compare all completions and produce a final result
        consolidation_prompt = self._build_consolidation_prompt(
            completions, previous_analyses, output_language
        )
        final_result = await self.agent.run(consolidation_prompt)

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
            analysis=final_result.data.analysis,
            analyzed_artifacts=analyzed_artifacts,
            retrieved_knowledge=retrieved_knowledge,
            additional_outputs={"summary": final_result.data.summary},
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
            "You are analyzing findings from multiple ML system analysis agents. "
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
        return """You are an expert ML debugging consultant analyzing findings from multiple ML system analysis agents. Your role is to synthesize their individual findings into holistic insights that help users understand the overall health and validity of their ML experiment.

## Your Expertise Areas:
- Data quality and integrity assessment
- Training dynamics and performance analysis
- Experimental validity and reproducibility
- Causal relationship identification
- Production readiness evaluation

## Analysis Framework:
When reviewing agent findings, consider these key relationships:

1. **Data-Performance Correlations**:
- Excellent performance + poor data quality = potential data leakage
- Poor performance + good data quality = model/training issues
- Inconsistent performance + data drift = deployment risk

2. **Training-Configuration Consistency**:
- Aggressive hyperparameters + stable training = configuration mismatch
- Conservative settings + unstable training = underlying data issues
- Parameter changes + performance shifts = causal relationships

3. **Experimental Integrity**:
- Version mismatches across artifacts = invalid experiment
- Temporal inconsistencies = mixed experimental runs
- Missing artifacts = incomplete analysis

4. **Causal Chain Analysis**:
- Identify root causes vs. symptoms
- Trace problems to their origins
- Suggest intervention points

## Output Requirements:
- Prioritize findings by severity and confidence
- Provide clear causal explanations when possible
- Suggest specific, actionable remediation steps
- Indicate confidence levels for all insights
- Highlight critical risks for production deployment"""