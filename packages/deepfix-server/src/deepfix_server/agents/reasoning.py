"""Cross-artifact reasoning synthesis for multi-agent findings."""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any, Dict, List, Optional

from agno.agent import Agent
from agno.models.base import Model
from agno.workflow import Parallel, Step, StepInput, StepOutput, Workflow
from deepfix_core.models import Severity
from deepfix_kb import KnowledgeBridge

from ..config import LLMConfig, settings
from ..logging import get_logger
from .models import create_agno_model
from .prompts import (
    CROSS_ARTIFACT_SYNTHESIS_SYSTEM_PROMPT,
    CROSS_ARTIFACT_SYSTEM_PROMPT,
)
from .schemas import AgentResult, CrossArtifactReasoningResult, CrossArtifactReasoningInput, SynthesisJudgeInput

LOGGER = get_logger(__name__)


def serialize_previous_analyses(previous_analyses: Dict[str, AgentResult]) -> str:
    """Serialize previous agent analyses into a readable JSON string."""
    analyses_dict = {}
    for name, ar in previous_analyses.items():
        if name == "CrossArtifactReasoningAgent":
            continue
        entry: dict = {}
        if ar.analysis is not None:
            entry["analysis"] = [a.model_dump() for a in ar.analysis]
        if ar.error_message is not None:
            entry["error"] = ar.error_message
        if ar.analyzed_artifacts:
            entry["analyzed_artifacts"] = ar.analyzed_artifacts
        analyses_dict[name] = entry

    return json.dumps(analyses_dict, indent=2, default=str)


SEVERITY_WEIGHTS = {
    Severity.HIGH: 3,
    "high": 3,
    Severity.MEDIUM: 2,
    "medium": 2,
    Severity.LOW: 1,
    "low": 1,
}


def _get_prioritized_queries(
    previous_analyses: Dict[str, AgentResult],
    max_queries: int = 6,
) -> List[str]:
    """Extract and sort unique finding descriptions by severity and confidence."""
    prioritized: List[tuple[int, float, str]] = []
    for ar in previous_analyses.values():
        if not ar.analysis:
            continue
        for analysis in ar.analysis:
            finding = analysis.findings
            if finding and finding.description:
                weight = (
                    SEVERITY_WEIGHTS.get(finding.severity, 0) if finding.severity else 0
                )
                confidence = (
                    finding.confidence if finding.confidence is not None else 0.0
                )
                prioritized.append((weight, confidence, finding.description))

    # Sort descending by severity weight, then confidence score
    prioritized.sort(key=lambda x: (x[0], x[1]), reverse=True)

    seen: set[str] = set()
    queries: List[str] = []
    for _, _, desc in prioritized:
        if desc not in seen:
            seen.add(desc)
            queries.append(desc)

    return queries[:max_queries]


async def prefetch_knowledge(
    previous_analyses: Dict[str, AgentResult],
    knowledge_bridge: Optional[KnowledgeBridge],
    max_queries: int = 6,
) -> List[str]:
    """Retrieve relevant domain knowledge based on findings prioritized by severity and confidence."""
    if not knowledge_bridge or not knowledge_bridge.has_available_sources:
        return []

    queries = _get_prioritized_queries(previous_analyses, max_queries)
    retrieved: List[str] = []
    for query in queries:
        try:
            response = await knowledge_bridge.query(
                query, synthesize=False, max_results=2
            )
            if response and response.results:
                for r in response.results:
                    retrieved.append(f"[{r.source_type}] {r.content}")
        except Exception as e:
            LOGGER.warning("Knowledge prefetch error for '%s': %s", query[:50], e)

    return retrieved


def build_cross_artifact_prompt(
    previous_analyses_json: str,
    knowledge_context: List[str],
    output_language: str,
) -> str:
    """Construct the prompt for cross-artifact reasoning synthesis."""
    sections = [
        "You are analyzing findings from multiple Machine Learning system analysis agents.",
        "Synthesize their individual findings into holistic insights and actionable recommendations.\n",
        f"## Previous Analyses\n\n{previous_analyses_json}\n",
    ]

    if knowledge_context:
        kb_text = "\n".join(f"- {item}" for item in knowledge_context)
        sections.append(f"## Retrieved ML Domain Knowledge\n\n{kb_text}\n")

    sections.append(f"Output language: {output_language}\n")
    sections.append(
        "Produce a consolidated analysis with cross-artifact insights, clear root causes, and a summary."
    )

    return "\n".join(sections)


def _resolve_model(
    model: Optional[Model] = None, llm_config: Optional[LLMConfig] = None
) -> Optional[Model]:
    if model is not None:
        return model
    try:
        if llm_config is not None:
            return create_agno_model(llm_config)
        return create_agno_model(settings.get_llm_config())
    except Exception as exc:
        LOGGER.warning("Could not resolve Agno model in reasoning: %s", exc)
        return None


def create_cross_artifact_reasoner(
    model: Optional[Model] = None, llm_config: Optional[LLMConfig] = None
) -> Agent:
    """Create a CrossArtifactReasoningAgent Agno agent."""
    return Agent(
        id="cross_artifact_reasoning_agent",
        name="CrossArtifactReasoningAgent",
        model=_resolve_model(model, llm_config),
        description="Synthesizes findings across multiple ML artifacts into cohesive root-cause analyses and prioritized recommendations.",
        instructions=CROSS_ARTIFACT_SYSTEM_PROMPT,
        input_schema=CrossArtifactReasoningInput,
        output_schema=CrossArtifactReasoningResult,
        use_json_mode=True
    )


def create_cross_artifact_synthesis_judge(
    model: Optional[Model] = None, llm_config: Optional[LLMConfig] = None
) -> Agent:
    """Create an Agno agent that judges and consolidates multiple reasoning chains."""
    return Agent(
        name="CrossArtifactSynthesisJudge",
        model=_resolve_model(model, llm_config),
        description="Judges and synthesizes multiple candidate reasoning chain outputs into a single consolidated, calibrated analysis.",
        instructions=CROSS_ARTIFACT_SYNTHESIS_SYSTEM_PROMPT,
        output_schema=CrossArtifactReasoningResult,
        input_schema=SynthesisJudgeInput,
        use_json_mode=True
    )


class CrossArtifactReasoningWorkflow(Workflow):
    """Agno Workflow orchestrating multi-chain reasoning (Parallel) and synthesis judge (Sequential)."""

    def __init__(
        self,
        reasoner: Agent,
        synthesis_judge: Optional[Agent] = None,
        knowledge_bridge: Optional[KnowledgeBridge] = None,
        output_language: str = "english",
        num_chains: int = 3,
        name: str = "CrossArtifactReasoningWorkflow",
        description: Optional[str] = (
            "Runs parallel candidate reasoning chains and synthesizes consolidated findings."
        ),
        telemetry: bool = False,
        **kwargs: Any,
    ):
        self.reasoner = reasoner
        self.synthesis_judge = synthesis_judge or create_cross_artifact_synthesis_judge(
            model=reasoner.model
        )
        self.knowledge_bridge = knowledge_bridge
        self.output_language = output_language
        self.num_chains = max(1, num_chains)

        self._previous_analyses: Dict[str, AgentResult] = {}
        self._retrieved_knowledge: Optional[str] = None
        self._user_message: str = ""
        self._reasoning_chains_results: Dict[str, CrossArtifactReasoningResult] = {}

        chain_steps = [
            Step(
                name=f"ReasoningChain_{i + 1}",
                executor=self._make_chain_executor(i + 1),
            )
            for i in range(self.num_chains)
        ]

        steps = [
            Step(
                name="PrepareReasoningPrompt",
                executor=self._step_prepare_prompt,
            ),
            Parallel(
                *chain_steps,
                name="ParallelReasoningChains",
            ),
            Step(
                name="SynthesisJudge",
                executor=self._step_synthesis_judge,
            ),
            Step(
                name="FormatAgentResult",
                executor=self._step_format_result,
            ),
        ]

        super().__init__(
            name=name,
            description=description,
            steps=steps,
            telemetry=telemetry,
            **kwargs,
        )

    async def _step_prepare_prompt(self, step_input: StepInput) -> StepOutput:
        """Serialize analyses, prefetch knowledge, and build user prompt."""
        raw_input = step_input.input
        raw_analyses = raw_input.get("previous_analyses", {}) if isinstance(raw_input, dict) else {}
        output_language = raw_input.get("output_language", self.output_language) if isinstance(raw_input, dict) else self.output_language

        self._previous_analyses = {}
        for k, v in raw_analyses.items():
            if isinstance(v, dict):
                try:
                    self._previous_analyses[k] = AgentResult.model_validate(v)
                except Exception:
                    self._previous_analyses[k] = v
            else:
                self._previous_analyses[k] = v

        analyses_json = serialize_previous_analyses(self._previous_analyses)
        self._retrieved_knowledge = await prefetch_knowledge(
            self._previous_analyses, self.knowledge_bridge
        )
       
        cross_artifact_reasoning_input = CrossArtifactReasoningInput(
            artifact_analysis_results=self._previous_analyses,
            retrieved_knowledge=self._retrieved_knowledge,
            output_language=output_language
        )

        return StepOutput(
            step_name="PrepareReasoningPrompt", content=cross_artifact_reasoning_input
        )

    def _make_chain_executor(self, chain_index: int):
        chain_agent = create_cross_artifact_reasoner(model=self.reasoner.model)

        async def _run_chain(step_input: StepInput) -> StepOutput:
            LOGGER.debug("Starting reasoning chain #%d", chain_index)
            step_name=f"ReasoningChain_{chain_index}"
            try:
                run_output = await chain_agent.arun(step_input.input)
                content = run_output.content
                if isinstance(content, dict):
                    try:
                        content = CrossArtifactReasoningResult.model_validate(content)
                    except Exception:
                        pass
                if not isinstance(content, CrossArtifactReasoningResult):
                    msg = f"Unexpected content type from reasoning agent: {type(content)}"
                    LOGGER.error(msg)
                    raise ValueError(msg)

                LOGGER.debug("Reasoning chain #%d completed successfully", chain_index)                
                self._reasoning_chains_results[step_name] = content

                return StepOutput(
                    step_name=step_name, content=content
                )
            except Exception as e:
                LOGGER.warning("Reasoning chain #%d failed: %s", chain_index, e)
                return StepOutput(
                    step_name=step_name,
                    content=None,
                    error=str(e),
                    success=False,
                )

        return _run_chain

    async def _step_synthesis_judge(self, step_input: StepInput) -> StepOutput:
        """Synthesize candidate analyses using the judge agent."""
        candidates: List[CrossArtifactReasoningResult] = []
        errors: List[str] = []

        for step_name, candidate in self._reasoning_chains_results.items():
            if isinstance(candidate, CrossArtifactReasoningResult):
                candidates.append(candidate)
            else:
                errors.append(f"{step_name} failed")

        if not candidates:
            err_msg = "; ".join(errors) if errors else "no valid candidates produced"
            raise RuntimeError(
                f"All {self.num_chains} reasoning chains failed ({err_msg})."
            )

        if len(candidates) == 1:
            LOGGER.info("Only 1 reasoning chain succeeded; bypassing synthesis step.")
            return StepOutput(step_name="SynthesisJudge", content=candidates[0])

        run_output = await self.synthesis_judge.arun(SynthesisJudgeInput(
            runs=candidates,
            output_language=self.output_language
        ))
        return StepOutput(step_name="SynthesisJudge", content=run_output.content)

    def _step_format_result(self, step_input: StepInput) -> StepOutput:
        """Package final synthesized analysis and metadata into AgentResult."""
        analyzed_artifacts: list[str] = []
        for ar in self._previous_analyses.values():
            if ar.analyzed_artifacts:
                analyzed_artifacts.extend(ar.analyzed_artifacts)

        res = step_input.previous_step_content
        if not isinstance(res, CrossArtifactReasoningResult):
            res = step_input.get_step_content("SynthesisJudge")

        if not isinstance(res, CrossArtifactReasoningResult):
            LOGGER.error(
                "Expected CrossArtifactReasoningResult from SynthesisJudge, got %s: %s",
                type(res).__name__,
                res,
            )
            return StepOutput(
                step_name="FormatAgentResult",
                content=AgentResult(
                    agent_name="CrossArtifactReasoningAgent",
                    error_message=f"Synthesis failed: {res}",
                    analyzed_artifacts=list(set(analyzed_artifacts)),
                ),
            )

        agent_result = AgentResult(
            agent_name="CrossArtifactReasoningAgent",
            analysis=res.analysis,
            analyzed_artifacts=list(set(analyzed_artifacts)),
            retrieved_knowledge=self._retrieved_knowledge,
            additional_outputs={"summary": res.summary},
        )
        return StepOutput(step_name="FormatAgentResult", content=agent_result)

    async def arun_reasoning(
        self,
        previous_analyses: Dict[str, AgentResult],
        output_language: str = "english",
    ) -> AgentResult:
        """Execute reasoning directly returning AgentResult."""
        self.output_language = output_language

        self._reasoning_chains_results.clear()
        
        try:
            serialized_analyses = {}
            for k, v in previous_analyses.items():
                if hasattr(v, "model_dump"):
                    serialized_analyses[k] = v.model_dump(mode="json")
                else:
                    serialized_analyses[k] = v

            run_output = await self.arun(
                input={
                    "previous_analyses": serialized_analyses,
                    "output_language": output_language,
                }
            )
            if isinstance(run_output.content, AgentResult):
                return run_output.content
            return AgentResult(
                agent_name="CrossArtifactReasoningAgent",
                error_message=f"Reasoning workflow output: {run_output.content}",
            )
        except Exception as e:
            LOGGER.error("Error in CrossArtifactReasoningWorkflow: %s", traceback.format_exc())
            return AgentResult(
                agent_name="CrossArtifactReasoningAgent",
                error_message=str(e),
            )


def create_cross_artifact_reasoning_workflow(
    model: Optional[Model] = None,
    llm_config: Optional[LLMConfig] = None,
    reasoner: Optional[Agent] = None,
    synthesis_judge: Optional[Agent] = None,
    knowledge_bridge: Optional[KnowledgeBridge] = None,
    output_language: str = "english",
    num_chains: int = 3,
    name: str = "CrossArtifactReasoningWorkflow",
    description: Optional[str] = (
        "Runs parallel candidate reasoning chains and synthesizes consolidated findings."
    ),
    **kwargs: Any,
) -> CrossArtifactReasoningWorkflow:
    """Create and return a CrossArtifactReasoningWorkflow instance."""
    resolved_reasoner = reasoner or create_cross_artifact_reasoner(
        model=model, llm_config=llm_config
    )
    resolved_judge = synthesis_judge or create_cross_artifact_synthesis_judge(
        model=model, llm_config=llm_config
    )
    return CrossArtifactReasoningWorkflow(
        reasoner=resolved_reasoner,
        synthesis_judge=resolved_judge,
        knowledge_bridge=knowledge_bridge,
        output_language=output_language,
        num_chains=num_chains,
        name=name,
        description=description,
        **kwargs,
    )
