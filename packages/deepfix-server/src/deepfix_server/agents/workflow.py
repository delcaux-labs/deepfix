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
    AgentContext
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
from .schemas import ReasoningWorkflowInput

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

        self._analyzers={"DatasetArtifactsAnalyzer":self._step_dataset_analyzer,
                    "TrainingArtifactsAnalyzer":self._step_training_analyzer,
                    "ModelCheckpointArtifactsAnalyzer":self._step_checkpoint_analyzer,
                    "DeepchecksArtifactsAnalyzer":self._step_deepchecks_analyzer,
                }

        super().__init__(
            name=name,
            description=description,
            id='analysisworkflow',
            input_schema=AgentContext,
            steps=[
                Step(name="ResolveContext", executor=self._step_resolve_context),
                Parallel(
                    *[Step(name=name, executor=executor) for name, executor in self._analyzers.items()],
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
        return StepOutput(step_name="ResolveContext", content=raw_input)

    async def _analyzer_step_executor(self,step_input: StepInput, step_name:str, artifacts_attr: str, agent: Agent) -> StepOutput:
            current_context = step_input.get_step_content("ResolveContext")
            artifacts = getattr(current_context, artifacts_attr, None)
            if artifacts is None:
                LOGGER.info(f"{step_name} skipped: {artifacts_attr} is None")
                return StepOutput(step_name=step_name, content=None)
            name = agent.name or step_name
            try:
                res: AgentResult = await run_artifact_analyzer(
                    agent=agent,
                    artifacts=artifacts,
                    output_language=current_context.language,
                    prompt_builder=self.prompt_builder,
                    dataset_name=current_context.dataset_name,
                )
            except Exception as e:
                LOGGER.error(f"Error executing analyzer '{name}': {e}")
                res = AgentResult(agent_name=name, error_message=str(e))
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
        
        artifact_analyzers_result = {}

        for step_name in self._analyzers.keys():
            content = step_input.get_step_content(step_name)
            if content is None:
                continue
            artifact_analyzers_result[step_name] = content

        current_context = step_input.get_step_content("ResolveContext")
        current_context.agent_results.update(
            **artifact_analyzers_result
        )

        result = await self.reasoning_workflow.arun(
            ReasoningWorkflowInput(
                previous_analyses=artifact_analyzers_result,
                output_language=current_context.language,
            )
        )
        cross_artifact_result = result.content
        summary = (
            cross_artifact_result.additional_outputs.get("summary", None)
            if cross_artifact_result and hasattr(cross_artifact_result, "additional_outputs") and cross_artifact_result.additional_outputs
            else None
        )
        current_context.agent_results[reasoner_name] = cross_artifact_result

        res = Result(
            context=current_context,
            summary=summary,
        ).to_api_response()
        
        return StepOutput(
            step_name="CrossArtifactReasoning",
            content=res,
        )

