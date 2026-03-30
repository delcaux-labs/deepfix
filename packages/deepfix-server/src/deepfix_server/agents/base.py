import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from typing import Any, List, Optional

import dspy

from ..config import LLMConfig, PromptConfig
from ..logging import get_logger
from ..models import AgentContext, AgentResult, Artifacts
from ..prompt_builders import PromptBuilder
from .signatures import ArtifactAnalysisSignature

LOGGER = get_logger(__name__)


class Agent(dspy.Module):
    """Base class for all analysis agents.

    Provides common functionality for LLM configuration and context management.
    Subclasses should implement the forward method and system_prompt property.

    Attributes:
        _llm_config: Optional LLM configuration for the agent.
        agent_name: Name of the agent derived from class name.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the agent.

        Args:
            config: Optional LLM configuration. If None, a warning is logged
                and dspy-settings should be configured separately.
        """
        super().__init__()
        assert (config is None) or isinstance(config, LLMConfig), (
            "config must be an instance of LLMConfig"
        )
        self._llm_config = config
        self.agent_name = self.__class__.__name__
        if config is None:
            LOGGER.warning(
                "No LLM config provided, Make sure to use dspy-settings.configure(...) to configure the LLM."
            )

    @contextmanager
    def _llm_context(self):
        """Context manager for LLM configuration.

        Yields a dspy context with the configured LLM if config is provided,
        otherwise yields a null context.

        Yields:
            dspy context with LLM configuration or null context.
        """
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
        """Forward method to be implemented by subclasses.

        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            Result of the agent's analysis.

        Raises:
            NotImplementedError: Always raised, must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def system_prompt(self) -> str:
        """System prompt for the agent.

        Returns:
            Empty string by default. Subclasses should override to provide
            their specific system prompt.
        """
        return ""


class ArtifactAnalyzer(Agent):
    """Base class for artifact analyzers.

    Analyzers that process specific types of artifacts (dataset, training, etc.).
    Subclasses should implement supported_artifact_types property.

    Attributes:
        prompt_builder: PromptBuilder instance for creating prompts from artifacts.
        llm: DSPy module for LLM interaction (defaults to ChainOfThought).
    """

    def __init__(
        self,
        llm: Optional[dspy.Module] = None,
        config: Optional[LLMConfig] = None,
        config_prompt_builder: Optional[PromptConfig] = None,
    ):
        """Initialize the artifact analyzer.

        Args:
            llm: Optional DSPy module for LLM interaction. If None, a ChainOfThought
                module is created with the agent's signature.
            config: Optional LLM configuration.
            config_prompt_builder: Optional prompt builder configuration.
        """
        super().__init__(config=config)
        self.prompt_builder = PromptBuilder(config=config_prompt_builder)
        signature = type(
            f"{self.agent_name}Signature",
            (ArtifactAnalysisSignature,),
            {"__doc__": self.system_prompt},
        )
        self.llm = llm or dspy.ChainOfThought(signature)

    def _check_artifacts(self, artifacts: List[Artifacts]) -> bool:
        """Check if all artifacts are supported by this analyzer.

        Args:
            artifacts: List of artifacts to check.

        Returns:
            True if all artifacts are supported.

        Raises:
            ValueError: If any artifact is not supported by this analyzer.
        """
        if not all(self.supports_artifact(a) for a in artifacts):
            raise ValueError(
                f"Artifacts must be supported by the analyzer. Received:{[type(a) for a in artifacts]}"
            )

    def run(self, context: AgentContext) -> AgentResult:
        """Run the analyzer with error handling.

        Args:
            context: Agent context containing artifacts and configuration.

        Returns:
            AgentResult with analysis or error message if execution fails.
        """
        return asyncio.run(self.arun(context))

    async def arun(self, context: AgentContext) -> AgentResult:
        """Run the analyzer asynchronously with error handling.

        Args:
            context: Agent context containing artifacts and configuration.

        Returns:
            AgentResult with analysis or error message if execution fails.
        """
        try:
            return await self.acall(context)
        except Exception as e:
            LOGGER.error(f"Error running {self.agent_name} agent: {traceback.format_exc()}")
            return AgentResult(agent_name=self.agent_name, error_message=str(e))

    def forward(self, context: AgentContext) -> AgentResult:
        """Analyze artifacts and return results.

        Args:
            context: Agent context containing artifacts and language preference.

        Returns:
            AgentResult containing analysis results and analyzed artifact types.
        """
        return asyncio.run(self.aforward(context))

    async def aforward(self, context: AgentContext) -> AgentResult:
        """Analyze artifacts asynchronously and return results.

        Args:
            context: Agent context containing artifacts and language preference.

        Returns:
            AgentResult containing analysis results and analyzed artifact types.
        """
        LOGGER.info(f"Running {self.agent_name} agent...")

        self._check_artifacts(context.artifacts)
        prompt = self.prompt_builder.build_prompt(
            artifacts=context.artifacts, context=None
        )
        with self._llm_context():
            response = await self.llm.acall(
                artifacts=prompt, output_language=context.language
            )
        return AgentResult(
            agent_name=self.agent_name,
            analysis=response.analysis,
            analyzed_artifacts=[type(a).__name__ for a in context.artifacts],
        )

    @property
    def supported_artifact_types(self):
        """Get the artifact types supported by this analyzer.

        Returns:
            Tuple or single class of artifact types that this analyzer can process.

        Raises:
            NotImplementedError: Always raised, must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def supports_artifact(self, artifact: Artifacts) -> bool:
        """Check if an artifact is supported by this analyzer.

        Args:
            artifact: Artifact to check.

        Returns:
            True if the artifact type is supported, False otherwise.
        """
        return isinstance(artifact, self.supported_artifact_types)
