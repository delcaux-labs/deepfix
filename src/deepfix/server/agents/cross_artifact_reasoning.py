import dspy
from typing import Dict, Optional
from ..config import LLMConfig
from .base import Agent, AgentResult
from .signatures import CrossArtifactReasoningSignature

class CrossArtifactReasoningAgent(Agent):
    def __init__(
        self,llm_config: Optional[LLMConfig] = None
    ):
        super().__init__(config=llm_config)
        self.llm = dspy.ChainOfThought(CrossArtifactReasoningSignature)
    
    def run(
        self,
        previous_analyses: Dict[str, AgentResult]
    ) -> AgentResult:
        try:
            return self(previous_analyses)
        except Exception as e:
            return AgentResult(agent_name=self.agent_name, error_message=str(e))

    def forward(
        self,
        previous_analyses: Dict[str, AgentResult]
    ) -> AgentResult:
        
        assert len(previous_analyses) > 0, "At least one analysis must be provided"
        with self._llm_context():
            out = self.llm(system_prompt=self.system_prompt,
            previous_analyses=previous_analyses)
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
            additional_outputs={'summary': out.summary}
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
        - Highlight critical risks for production deployment
        """
