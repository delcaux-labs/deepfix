import asyncio
import json
from typing import Any, Dict, List, Optional, Union

import mlflow
from agno.agent import Agent
from agno.db.base import AsyncBaseDb, BaseDb
from agno.models.base import Model
from agno.workflow import Parallel, Step, StepInput, StepOutput, Workflow
from agno.workflow.types import WorkflowExecutionInput
from deepfix_core.models import (
    AgentResult,
    DatasetArtifacts,
    DeepchecksArtifacts,
    ModelCheckpointArtifacts,
    TrainingArtifacts,
)
from deepfix_kb import KnowledgeBridge

from deepfix_server.config import LLMConfig
from deepfix_server.logging import get_logger
from deepfix_server.models import Result
from deepfix_server.prompt_builders import PromptBuilder

from .analyzers import (
    create_checkpoint_analyzer,
    create_dataset_analyzer,
    create_deepchecks_analyzer,
    create_training_analyzer,
    run_artifact_analyzer,
)
from .reasoning import (
    create_cross_artifact_reasoner,
    create_cross_artifact_reasoning_workflow,
    create_cross_artifact_synthesis_judge,
)
from .schemas import AgentContext

LOGGER = get_logger(__name__)


class AnalysisWorkflow(Workflow):
    """Agno Workflow orchestrating dynamic fan-out analyzer execution and cross-artifact synthesis."""

    def __init__(
        self,
        model: Optional[Model] = None,
        llm_config: Optional[LLMConfig] = None,
        knowledge_bridge: Optional[KnowledgeBridge] = None,
        num_chains: int = 3,
        prompt_builder: Optional[PromptBuilder] = None,
        name: str = "AnalysisWorkflow",
        description: Optional[str] = (
            "Orchestrates dynamic fan-out ML artifact analyzers and cross-artifact reasoning synthesis."
        ),
        db: Optional[Union[BaseDb, AsyncBaseDb]] = None,
        telemetry: bool = False,
        **kwargs: Any,
    ):
        """Initialize the analysis workflow with specialized Agno agents.

        Args:
            model: Optional Agno model instance.
            llm_config: Optional LLM configuration.
            knowledge_bridge: Optional KnowledgeBridge instance for domain knowledge retrieval.
            num_chains: Number of reasoning chains for cross-artifact synthesis.
            prompt_builder: Optional custom prompt builder instance.
            name: Name of the workflow.
            description: Description of the workflow.
            db: Optional session / run database.
            telemetry: Whether to enable Agno telemetry (defaults to False).
            **kwargs: Additional keyword arguments for Workflow.
        """
        self.llm_config = llm_config
        self.model = model
        self.knowledge_bridge = knowledge_bridge
        self.num_chains = num_chains
        self.prompt_builder = prompt_builder

        # Initialize specialized artifact analyzers
        self.dataset_analyzer: Agent = create_dataset_analyzer(
            model=model, llm_config=llm_config
        )
        self.training_analyzer: Agent = create_training_analyzer(
            model=model, llm_config=llm_config
        )
        self.checkpoint_analyzer: Agent = create_checkpoint_analyzer(
            model=model, llm_config=llm_config
        )
        self.deepchecks_analyzer: Agent = create_deepchecks_analyzer(
            model=model, llm_config=llm_config
        )
        self.reasoner: Agent = create_cross_artifact_reasoner(
            model=model, llm_config=llm_config
        )
        synthesis_judge: Agent = create_cross_artifact_synthesis_judge(
            model=model, llm_config=llm_config
        )
        self.reasoning_workflow = create_cross_artifact_reasoning_workflow(
            model=model,
            llm_config=llm_config,
            reasoner=self.reasoner,
            synthesis_judge=synthesis_judge,
            knowledge_bridge=knowledge_bridge,
            num_chains=num_chains,
        )
        self._current_context: Optional[AgentContext] = None
        self._summary: Optional[str] = None

        super().__init__(
            name=name,
            description=description,
            steps=[
                Step(name="ResolveContext", executor=self._step_resolve_context),
                Parallel(
                    Step(
                        name="DatasetArtifactsAnalyzer",
                        executor=self._step_dataset_analyzer,
                    ),
                    Step(
                        name="TrainingArtifactsAnalyzer",
                        executor=self._step_training_analyzer,
                    ),
                    Step(
                        name="ModelCheckpointArtifactsAnalyzer",
                        executor=self._step_checkpoint_analyzer,
                    ),
                    Step(
                        name="DeepchecksArtifactsAnalyzer",
                        executor=self._step_deepchecks_analyzer,
                    ),
                    name="ArtifactAnalyzers",
                ),
                Step(
                    name="CrossArtifactReasoning",
                    executor=self._step_cross_artifact_reasoning,
                ),
            ],
            db=db,
            telemetry=telemetry,
            **kwargs,
        )

    def _step_resolve_context(self, step_input: StepInput) -> StepOutput:
        """Resolve and strictly validate AgentContext from step_input."""
        raw_input = step_input.input
        if isinstance(raw_input, WorkflowExecutionInput):
            raw_input = getattr(raw_input, "input", raw_input)
        
        if isinstance(raw_input, str):
            data = json.loads(raw_input)
            raw_input = data.get("context", data)
        if isinstance(raw_input, dict):
            raw_input = AgentContext.model_validate(raw_input)

        if not isinstance(raw_input, AgentContext):
            raise TypeError(
                f"AnalysisWorkflow only accepts AgentContext, got {type(raw_input).__name__}"
            )
        self._current_context = raw_input
        return StepOutput(step_name="ResolveContext", content=self._current_context)

    async def _analyzer_step_executor(self,step_input: StepInput, step_name:str, artifacts_attr: str, agent: Agent) -> StepOutput:
            if self._current_context is None:
                return StepOutput(step_name=step_name, content=None)
            artifacts = getattr(self._current_context, artifacts_attr, None)
            if artifacts is None:
                LOGGER.info(f"{step_name} skipped: {artifacts_attr} is None")
                return StepOutput(step_name=step_name, content=None)
            name = agent.name or step_name
            try:
                res = await run_artifact_analyzer(
                    agent=agent,
                    artifacts=artifacts,
                    output_language=self._current_context.language,
                    prompt_builder=self.prompt_builder,
                    dataset_name=self._current_context.dataset_name,
                )
            except Exception as e:
                LOGGER.error(f"Error executing analyzer '{name}': {e}")
                res = AgentResult(agent_name=name, error_message=str(e))
            self._current_context.agent_results[name] = res
            return StepOutput(step_name=step_name, content=res)

    async def _step_dataset_analyzer(self, step_input: StepInput) -> StepOutput:
        """Execute DatasetArtifactsAnalyzer if dataset_artifacts are present."""
        return await self._analyzer_step_executor(step_input, step_name="DatasetArtifactsAnalyzer", artifacts_attr="dataset_artifacts", agent=self.dataset_analyzer)

    async def _step_training_analyzer(self, step_input: StepInput) -> StepOutput:
        """Execute TrainingArtifactsAnalyzer if training_artifacts are present."""
        return await self._analyzer_step_executor(step_input, step_name="TrainingArtifactsAnalyzer", artifacts_attr="training_artifacts", agent=self.training_analyzer)

    async def _step_checkpoint_analyzer(self, step_input: StepInput) -> StepOutput:
        """Execute ModelCheckpointArtifactsAnalyzer if model_checkpoint_artifacts are present."""
        return await self._analyzer_step_executor(step_input, step_name="ModelCheckpointArtifactsAnalyzer", artifacts_attr="model_checkpoint_artifacts", agent=self.checkpoint_analyzer)

    async def _step_deepchecks_analyzer(self, step_input: StepInput) -> StepOutput:
        """Execute DeepchecksArtifactsAnalyzer if deepchecks_artifacts are present."""
        return await self._analyzer_step_executor(step_input, step_name="DeepchecksArtifactsAnalyzer", artifacts_attr="deepchecks_artifacts", agent=self.deepchecks_analyzer)

    async def _step_cross_artifact_reasoning(self, step_input: StepInput) -> StepOutput:
        """Execute CrossArtifactReasoning synthesis over aggregated analyzer results."""
        reasoner_name = self.reasoner.name or "CrossArtifactReasoningAgent"
        cross_artifact_result = await self.reasoning_workflow.arun_reasoning(
            previous_analyses=self._current_context.agent_results,
            output_language=self._current_context.language,
        )
        self._summary = cross_artifact_result.additional_outputs.get("summary", None)
        self._current_context.agent_results[reasoner_name] = cross_artifact_result
        res = Result(
            context=self._current_context,
            summary=self._summary,
        )
        return StepOutput(
            step_name="CrossArtifactReasoning",
            content=res,
        )

    @mlflow.trace(name="AnalysisWorkflow.run_analysis")
    async def run_analysis(self, context: AgentContext) -> Result:
        """Convenience method to execute analysis directly returning Result.

        Args:
            context: AgentContext containing ML artifacts.

        Returns:
            Result with aggregated findings and synthesis summary.
        """
        run_output = await self.arun(input=context,
                                    #background=True
                                    )
        assert isinstance(run_output.content, Result), f"Workflow output is not a Result: type={type(run_output.content)}"
        return run_output.content
