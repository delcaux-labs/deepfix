import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import dspy
from deepfix_kb import KnowledgeBridge
from deepfix_kb.tools import create_knowledge_tools

from ..config import LLMConfig
from ..logging import get_logger
from .base import Agent, AgentResult
from .signatures import CrossArtifactReasoningSignature

LOGGER = get_logger(__name__)


class CrossArtifactReasoningAgent(Agent):
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        knowledge_bridge: Optional[KnowledgeBridge] = None,
    ):
        super().__init__(config=llm_config)
        self.knowledge_bridge = knowledge_bridge
        signature = type(
            f"{self.agent_name}Signature",
            (CrossArtifactReasoningSignature,),
            {"__doc__": self.system_prompt},
        )
        if self.knowledge_bridge:
            tools = create_knowledge_tools(self.knowledge_bridge, include_hybrid=False)
            self.llm = dspy.ReAct(signature, tools=tools)
        else:
            self.llm = dspy.ChainOfThought(signature)

    def run(
        self,
        previous_analyses: Dict[str, AgentResult],
        output_language: str = "english",
    ) -> AgentResult:
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
        try:
            return await self.acall(previous_analyses, output_language)
        except Exception as e:
            LOGGER.error(
                f"Error with agent {self.agent_name}:\n {traceback.format_exc()}"
            )
            return AgentResult(agent_name=self.agent_name, error_message=str(e))

    def forward(
        self,
        previous_analyses: Dict[str, AgentResult],
        output_language: str = "english",
    ) -> AgentResult:
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
        LOGGER.info("Running cross-artifact reasoning agent...")

        assert len(previous_analyses) > 0, "At least one analysis must be provided"
        with self._llm_context():
            out = await self.llm.acall(
                previous_analyses=previous_analyses, output_language=output_language
            )
        analyzed_artifacts = []
        retrieved_knowledge = []
        for result in previous_analyses.values():
            if result.analyzed_artifacts is not None:
                analyzed_artifacts.extend(result.analyzed_artifacts)
            if result.retrieved_knowledge is not None:
                retrieved_knowledge.extend(result.retrieved_knowledge)

        return AgentResult(
            agent_name=self.agent_name,
            analysis=out.analysis,
            analyzed_artifacts=analyzed_artifacts,
            retrieved_knowledge=retrieved_knowledge,
            additional_outputs={"summary": out.summary},
        )

    @property
    def system_prompt(self) -> str:
        """You are an expert ML debugging consultant analyzing findings from multiple ML system analysis agents. Your role is to synthesize their individual findings into holistic insights that help users understand the overall health and validity of their ML experiment.

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
        - Aggressive hyperparameters + stable training = config uration mismatch
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
        - Highlight critical risks for production deployment
        """
