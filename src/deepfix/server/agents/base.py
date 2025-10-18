from typing import List, Optional, Any
from contextlib import contextmanager, nullcontext
import dspy
from ..config import PromptConfig, LLMConfig
from ..prompt_builders import PromptBuilder
from ..models import AgentContext, AgentResult, Artifacts
from .signatures import ArtifactAnalysisSignature
from ...shared.models import Artifacts

from ..logging import get_logger

LOGGER = get_logger(__name__)   

class Agent(dspy.Module):
    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__()
        assert (config is None) or isinstance(config, LLMConfig), "config must be an instance of LLMConfig"
        self._llm_config = config
        self.agent_name = self.__class__.__name__.replace("agent", "")
        if config is None:
            LOGGER.warning("No LLM config provided, Make sure to use dspy-settings.configure(...) to configure the LLM.")

    @contextmanager
    def _llm_context(self):
        if self._llm_config is None:
            with nullcontext():
                yield
            return
        with dspy.context(
            lm=dspy.LM(
                model=self._llm_config.model_name,
                cache=self._llm_config.cache,
                api_base=self._llm_config.base_url,
                api_key=self._llm_config.api_key,
                temperature=self._llm_config.temperature,
                max_tokens=self._llm_config.max_tokens,
            ),
            track_usage=self._llm_config.track_usage,
        ):
            yield
    
    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement this method")
    
    @property
    def system_prompt(self) -> str:
        return ""


class ArtifactAnalyzer(Agent):
    def __init__(
        self, llm: Optional[dspy.Module] = None, config: Optional[LLMConfig] = None, config_prompt_builder: Optional[PromptConfig] = None
    ):
        super().__init__(config=config)
        self.prompt_builder = PromptBuilder(config=config_prompt_builder)
        self.llm = llm or dspy.ChainOfThought(ArtifactAnalysisSignature)
    
    def _check_artifacts(self, artifacts: List[Artifacts]) -> bool:
        if not all(self.supports_artifact(a) for a in artifacts):
            raise ValueError(f"Artifacts must be supported by the analyzer. Received:{[type(a) for a in artifacts]}")

    def run(self, context: AgentContext) -> AgentResult:
        try:
            return self(context)
        except Exception as e:
            return AgentResult(agent_name=self.agent_name, error_message=str(e))

    def forward(self, context: AgentContext) -> AgentResult:
        
        LOGGER.info(f"Running {self.agent_name} agent...")
        
        self._check_artifacts(context.artifacts)
        prompt = self.prompt_builder.build_prompt(artifacts=context.artifacts,context=None)
        with self._llm_context():
            response = self.llm(artifacts=prompt,system_prompt=self.system_prompt)
        return AgentResult(
            agent_name=self.agent_name,
            analysis=response.analysis,
            analyzed_artifacts=[type(a).__name__ for a in context.artifacts],
        )

    @property
    def supported_artifact_types(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    def supports_artifact(self, artifact:Artifacts) -> bool:
        return isinstance(artifact, self.supported_artifact_types)

