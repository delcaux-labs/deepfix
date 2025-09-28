from abc import ABC, abstractmethod
from typing import List, Type, Optional, Tuple, Any
import dspy
from ..config import PromptConfig
from ..prompt_builders import PromptBuilder
from ..artifacts import Artifacts
from .models import AgentContext, AgentResult


class Agent(dspy.Module, ABC):
    @abstractmethod
    def forward(self, **kwargs) -> AgentResult:
        pass


class ArtifactAnalyzer(Agent):
    def __init__(
        self, llm: dspy.Module, config_prompt_builder: Optional[PromptConfig] = None
    ):
        self.agent_name = self.__class__.__name__.lower().replace("agent", "")
        self.prompt_builder = PromptBuilder(config=config_prompt_builder)
        self.llm = llm
    
    def _check_artifacts(self, artifacts: List[Artifacts]) -> bool:
        if not all(self.supports_artifact(a) for a in artifacts):
            raise ValueError(f"Artifacts must be supported by the analyzer. Received:{[type(a) for a in artifacts]}")

    def _run(self, context: AgentContext) -> AgentResult:
        self._check_artifacts(context.artifacts)
        prompt = self.prompt_builder.build_prompt(context.artifacts)
        response = self.llm(artifacts=prompt)
        return AgentResult(
            agent_name=self.agent_name,
            analysis=response.analysis,
            analyzed_artifacts=[type(a).__name__ for a in context.artifacts],
        )

    def forward(self, context: AgentContext) -> AgentResult:
        assert isinstance(context, AgentContext), "Context must be an instance of AgentContext"
        return self._run(context)

    @property
    def supported_artifact_types(self) -> Tuple[Type[Artifacts]]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def supports_artifact(self, artifact: Artifacts) -> bool:
        return isinstance(artifact, self.supported_artifact_types)

    @property
    def system_prompt(self) -> str:
        raise NotImplementedError("Subclasses must implement this method")
