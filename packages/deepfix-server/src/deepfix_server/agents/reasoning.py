"""Cross-artifact reasoning synthesis for multi-agent findings."""

from __future__ import annotations

from typing import Any, List, Optional

from agno.agent import Agent
from agno.models.base import Model
from agno.tools import Toolkit
from agno.workflow import Parallel, Step, StepInput, StepOutput, Workflow

from ..config import LLMConfig, settings
from ..logging import get_logger
from ..tools import get_search_tools
from .models import create_agno_model
from .prompts import (
    CROSS_ARTIFACT_SYNTHESIS_SYSTEM_PROMPT,
    CROSS_ARTIFACT_SYSTEM_PROMPT,
)
from .schemas import (
    AgentResult,
    CrossArtifactReasoningInput,
    CrossArtifactReasoningResult,
    ReasoningWorkflowInput,
    SynthesisJudgeInput,
)

LOGGER = get_logger(__name__)


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
    model: Optional[Model] = None,
    llm_config: Optional[LLMConfig] = None,
    tools: Optional[List[Toolkit]] = None,
) -> Agent:
    """Create a CrossArtifactReasoningAgent Agno agent."""
    resolved_tools = get_search_tools() if tools is None else tools
    return Agent(
        id="cross_artifact_reasoning_agent",
        name="CrossArtifactReasoningAgent",
        model=_resolve_model(model, llm_config),
        tools=resolved_tools,
        description="Synthesizes findings across multiple ML artifacts into cohesive root-cause analyses and prioritized recommendations.",
        instructions=CROSS_ARTIFACT_SYSTEM_PROMPT,
        input_schema=CrossArtifactReasoningInput,
        output_schema=CrossArtifactReasoningResult,
        use_json_mode=True,
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
        use_json_mode=True,
    )


class CrossArtifactReasoningWorkflow(Workflow):
    """Agno Workflow orchestrating multi-chain reasoning (Parallel) and synthesis judge (Sequential)."""

    def __init__(
        self,
        reasoner: Agent,
        synthesis_judge: Optional[Agent] = None,
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
        self.output_language = output_language
        self.num_chains = max(1, num_chains)
        self.reasoning_chain_step_names = [
            f"ReasoningChain_{i}" for i in range(self.num_chains)
        ]

        chain_steps = [
            Step(
                name=name,
                executor=self._make_chain_executor(name),
            )
            for name in self.reasoning_chain_step_names
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
            input_schema=ReasoningWorkflowInput,
            **kwargs,
        )

    async def _step_prepare_prompt(self, step_input: StepInput) -> StepOutput:
        """Serialize analyses and build input for reasoning chains."""
        content = step_input.input
        assert isinstance(content, ReasoningWorkflowInput), (
            f"Input must be ReasoningWorkflowInput, got {type(content)}"
        )

        cross_artifact_reasoning_input = CrossArtifactReasoningInput(
            artifact_analysis_results=list(content.previous_analyses.values()),
            output_language=content.output_language,
        )

        return StepOutput(
            step_name="PrepareReasoningPrompt", content=cross_artifact_reasoning_input
        )

    def _make_chain_executor(self, chain_name: str):
        chain_agent = create_cross_artifact_reasoner(
            model=self.reasoner.model,
            tools=self.reasoner.tools,
        )

        async def _run_chain(step_input: StepInput) -> StepOutput:
            LOGGER.debug("Starting reasoning chain #%s", chain_name)
            try:
                run_output = await chain_agent.arun(
                    step_input.get_step_content("PrepareReasoningPrompt")
                )
                content = run_output.content
                if isinstance(content, dict):
                    try:
                        content = CrossArtifactReasoningResult.model_validate(content)
                    except Exception:
                        pass
                if not isinstance(content, CrossArtifactReasoningResult):
                    msg = (
                        f"Unexpected content type from reasoning agent: {type(content)}"
                    )
                    LOGGER.error(msg)
                    raise ValueError(msg)

                LOGGER.debug("Reasoning chain #%s completed successfully", chain_name)

                return StepOutput(step_name=chain_name, content=content)
            except Exception as e:
                LOGGER.warning("Reasoning chain #%s failed: %s", chain_name, e)
                return StepOutput(
                    step_name=chain_name,
                    content=None,
                    error=str(e),
                    success=False,
                )

        return _run_chain

    async def _step_synthesis_judge(self, step_input: StepInput) -> StepOutput:
        """Synthesize candidate analyses using the judge agent."""
        candidates: List[CrossArtifactReasoningResult] = []
        errors: List[str] = []

        for step_name in self.reasoning_chain_step_names:
            candidate = step_input.get_step_content(step_name)
            if isinstance(candidate, CrossArtifactReasoningResult):
                candidates.append(candidate)
            else:
                errors.append(
                    f"{step_name} failed. Expected CrossArtifactReasoningResult, got {type(candidate)}"
                )

        if not candidates:
            err_msg = "; ".join(errors) if errors else "no valid candidates produced"
            raise RuntimeError(
                f"All {self.num_chains} reasoning chains failed ({err_msg})."
            )

        if len(candidates) == 1:
            LOGGER.info("Only 1 reasoning chain succeeded; bypassing synthesis step.")
            return StepOutput(step_name="SynthesisJudge", content=candidates[0])

        run_output: CrossArtifactReasoningResult = await self.synthesis_judge.arun(
            SynthesisJudgeInput(
                runs=candidates, output_language=step_input.input.output_language
            )
        )
        return StepOutput(step_name="SynthesisJudge", content=run_output.content)

    def _step_format_result(self, step_input: StepInput) -> StepOutput:
        """Package final synthesized analysis and metadata into AgentResult."""
        analyzed_artifacts: list[str] = []
        agent_results: list[AgentResult] = step_input.get_step_content(
            "PrepareReasoningPrompt"
        ).artifact_analysis_results
        for ar in agent_results:
            if ar.analyzed_artifacts:
                analyzed_artifacts.extend(ar.analyzed_artifacts)

        res = step_input.get_step_content("SynthesisJudge")
        agent_result = AgentResult(
            agent_name="CrossArtifactReasoningAgent",
            analysis=res.analysis,
            analyzed_artifacts=list(set(analyzed_artifacts)),
            retrieved_knowledge=None,
            additional_outputs={"summary": res.summary},
        )
        return StepOutput(step_name="FormatAgentResult", content=agent_result)


def create_cross_artifact_reasoning_workflow(
    model: Optional[Model] = None,
    llm_config: Optional[LLMConfig] = None,
    reasoner: Optional[Agent] = None,
    synthesis_judge: Optional[Agent] = None,
    tools: Optional[List[Toolkit]] = None,
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
        model=model, llm_config=llm_config, tools=tools
    )
    resolved_judge = synthesis_judge or create_cross_artifact_synthesis_judge(
        model=model, llm_config=llm_config
    )
    return CrossArtifactReasoningWorkflow(
        reasoner=resolved_reasoner,
        synthesis_judge=resolved_judge,
        output_language=output_language,
        num_chains=num_chains,
        name=name,
        description=description,
        **kwargs,
    )
