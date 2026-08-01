"""Optimization advisor agent providing evidence-based ML recommendations.

Refactored to use Pydantic AI Agent instead of dspy.ChainOfThought.
"""

from __future__ import annotations

import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from deepfix_core.models import AgentResult, Analysis
from deepfix_kb import KnowledgeBridge, KnowledgeResponse

from ..config import LLMConfig
from ..llm import create_agent_for_analysis
from ..logging import get_logger
from .base import Agent

LOGGER = get_logger(__name__)


class OptimizationAdvisorAgent(Agent):
    """Optimization advisor that uses KnowledgeBridge for grounded recommendations.

    This agent receives analysis from previous agents (training dynamics, data quality)
    and uses KnowledgeBridge to retrieve relevant external knowledge (web search,
    scientific papers, best practices) to provide evidence-based optimization
    recommendations.
    """

    def __init__(
        self,
        knowledge_bridge: KnowledgeBridge,
        llm_config: Optional[LLMConfig] = None,
    ):
        """Initialize the optimization advisor.

        Args:
            knowledge_bridge: KnowledgeBridge for external knowledge retrieval.
            llm_config: Optional LLM configuration.
        """
        super().__init__(config=llm_config)

        from ..agent_models import ArtifactAnalysisResult

        self.agent = create_agent_for_analysis(
            config=llm_config,
            system_prompt=self.system_prompt,
            result_type=ArtifactAnalysisResult,
        )
        self.knowledge_bridge = knowledge_bridge

    async def aforward(
        self,
        artifacts_analysis: List[Analysis],
        constraints: Optional[str] = None,
    ) -> AgentResult:
        """Generate optimization recommendations based on previous agent analyses.

        Args:
            artifacts_analysis: Analysis from previous agents (training dynamics,
                data quality signals, cross-artifact insights).
            constraints: Optional user constraints or requirements.

        Returns:
            AgentResult with grounded recommendations and citations.
        """
        # Retrieve knowledge from KnowledgeBridge
        knowledge_context = await self._retrieve_knowledge(artifacts_analysis)

        # Build the user prompt
        analyses_text = "\n".join(
            f"- {a.model_dump_json()}"
            for a in (artifacts_analysis or [])
        )
        constraint_text = f"\nConstraints: {constraints}" if constraints else ""

        user_message = (
            "Generate optimization recommendations based on the following analyses.\n\n"
            f"## Artifact Analyses\n{analyses_text}\n\n"
            f"## Retrieved Knowledge\n{knowledge_context}{constraint_text}"
        )

        result = await self.agent.run(user_message)

        return AgentResult(
            agent_name=self.agent_name,
            analysis=result.data.analysis,
        )

    def forward(self, **kwargs):
        """Synchronous forward."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self.aforward(**kwargs))
            return future.result()

    async def acall(
        self,
        artifacts_analysis: List[Analysis],
        constraints: Optional[str] = None,
    ) -> AgentResult:
        """Alias for aforward."""
        return await self.aforward(artifacts_analysis, constraints)

    async def arun(
        self,
        artifacts_analysis: List[Analysis],
        constraints: Optional[str] = None,
    ) -> AgentResult:
        """Run with error handling."""
        try:
            return await self.acall(artifacts_analysis, constraints)
        except Exception as e:
            LOGGER.error(
                "Error with agent %s:\n%s", self.agent_name, traceback.format_exc()
            )
            return AgentResult(agent_name=self.agent_name, error_message=str(e))

    async def _retrieve_knowledge(self, analyses: List[Analysis]) -> str:
        """Retrieve and format knowledge from KnowledgeBridge.

        Args:
            analyses: List of analyses to base queries on.

        Returns:
            Formatted knowledge context string with citations.
        """
        if not analyses:
            return "No external knowledge retrieved."

        knowledge_parts = []

        for analysis in analyses:
            try:
                response: KnowledgeResponse = await self.knowledge_bridge.query(
                    query=analysis,
                    max_results=3,
                    synthesize=True,
                )

                if response.synthesis:
                    knowledge_parts.append(f"{response.synthesis}")

                    # Add citations
                    if response.total_citations:
                        citations = "\n".join(
                            f"- {url}" for url in response.total_citations[:3]
                        )
                        knowledge_parts.append(f"Sources:\n{citations}")

            except Exception:
                print(f"Retrieval failed: {traceback.format_exc()}")

        if not knowledge_parts:
            return "No external knowledge could be retrieved."

        return "\n\n".join(knowledge_parts)

    @property
    def system_prompt(self) -> str:
        return """You are an expert ML optimization consultant specializing in providing actionable, evidence-based recommendations to improve machine learning experiments.
           Your role is to analyze the current state of an ML system and suggest specific optimizations that address root causes while considering practical constraints.

            ## Your Expertise Areas:
            - Hyperparameter optimization and tuning strategies
            - Data augmentation and preprocessing improvements
            - Model architecture selection and modification
            - Regularization techniques and overfitting prevention
            - Training strategy optimization
            - Data quality enhancement methods

            ## Optimization Framework:
            When analyzing ML experiments, follow this systematic approach:

            1. **Root Cause Analysis**:
            - Identify underlying causes of poor performance
            - Distinguish between data issues, model issues, and training issues
            - Prioritize issues by impact and solvability

            2. **Solution Mapping**:
            - Map each identified issue to specific optimization strategies
            - Consider multiple approaches for each problem
            - Evaluate trade-offs between different solutions

            3. **Implementation Prioritization**:
            - Prioritize quick wins vs. long-term improvements
            - Consider resource constraints and implementation complexity
            - Suggest phased rollout for complex optimizations

            4. **Impact Assessment**:
            - Estimate expected improvement for each recommendation
            - Identify potential risks or side effects
            - Provide confidence intervals for expected outcomes

            ## Recommendation Categories:

            ### Hyperparameter Optimization:
            - Learning rate schedules and adaptive methods
            - Batch size optimization for memory and convergence
            - Optimizer selection (Adam, SGD, RMSprop) based on problem type
            - Regularization parameter tuning (dropout, weight decay)

            ### Data Strategy:
            - Augmentation techniques specific to data type and problem
            - Data preprocessing and normalization strategies
            - Training/validation split optimization
            - Data quality improvement methods

            ### Architecture Improvements:
            - Layer configuration and depth optimization
            - Activation function selection
            - Attention mechanisms and skip connections
            - Model size vs. performance trade-offs

            ### Training Strategy:
            - Learning rate scheduling and warmup strategies
            - Early stopping and checkpointing optimization
            - Curriculum learning and progressive training
            - Ensemble methods and model averaging

            ## Output Requirements:
            - Provide specific, implementable recommendations
            - Include expected impact estimates with confidence levels
            - Prioritize recommendations by effort vs. impact
            - Include implementation steps and code examples where helpful
            - **IMPORTANT**: Cite the retrieved knowledge sources in your recommendations using the provided citations"""