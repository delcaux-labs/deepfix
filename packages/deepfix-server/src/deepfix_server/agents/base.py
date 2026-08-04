"""Base agent classes for DeepFix analysis.

Provides the Agent and ArtifactAnalyzer base classes, refactored to use
Pydantic AI instead of DSPy Module / ChainOfThought.
"""

from __future__ import annotations

import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional
from abc import ABC, abstractmethod
from ..config import LLMConfig, PromptConfig
from ..llm import create_agent_for_analysis
from ..logging import get_logger
from .schemas import AgentContext, AgentResult, Artifacts, ArtifactAnalysisResult
from ..prompt_builders import PromptBuilder


LOGGER = get_logger(__name__)


class Agent(ABC):
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
                and the agent will need a configured model to function.
        """
        assert (config is None) or isinstance(config, LLMConfig), (
            "config must be an instance of LLMConfig"
        )
        self.llm_config = config
        self.agent_name = self.__class__.__name__
        if config is None:
            LOGGER.warning(
                "No LLM config provided for %s. Ensure the agent's model is "
                "configured before calling run().",
                self.agent_name,
            )

    @property
    def system_prompt(self) -> str:
        """System prompt for the agent.

        Returns:
            Empty string by default. Subclasses should override to provide
            their specific system prompt.
        """
        return ""

    def run(self, context: AgentContext) -> AgentResult:
        """Run the analyzer synchronously.

        Args:
            context: Agent context containing artifacts and configuration.

        Returns:
            AgentResult with analysis or error message if execution fails.
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self.arun(context))
            return future.result()
    
    @abstractmethod
    async def arun(self, context: AgentContext) -> AgentResult:
        """Run the analyzer asynchronously.

        Args:
            context: Agent context containing artifacts and configuration.

        Returns:
            AgentResult with analysis or error message if execution fails.
        """
        raise NotImplementedError


class ArtifactAnalyzer(Agent):
    """Base class for artifact analyzers using Pydantic AI.

    Analyzers that process specific types of artifacts (dataset, training, etc.).
    Subclasses should implement supported_artifact_types property.

    Replaces the DSPy ChainOfThought(signature) pattern with a Pydantic AI
    Agent configured with ArtifactAnalysisResult as result_type.

    Attributes:
        prompt_builder: PromptBuilder instance for creating prompts from artifacts.
        agent: Pydantic AI Agent configured for structured analysis output.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        config_prompt_builder: Optional[PromptConfig] = None,
    ):
        """Initialize the artifact analyzer.

        Args:
            config: Optional LLM configuration.
            config_prompt_builder: Optional prompt builder configuration.
        """
        super().__init__(config=config)
        self.prompt_builder = PromptBuilder(config=config_prompt_builder)       

        self.agent = create_agent_for_analysis(
            config=config,
            system_prompt=self.system_prompt,
            output_type=ArtifactAnalysisResult,
        )

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
                f"Artifacts must be supported by the analyzer. "
                f"Received: {[type(a) for a in artifacts]}"
            )
        return True

    async def arun(self, context: AgentContext) -> AgentResult:
        """Run the analyzer asynchronously with error handling.

        Args:
            context: Agent context containing artifacts and configuration.

        Returns:
            AgentResult with analysis or error message if execution fails.
        """
        try:
            LOGGER.info("Running %s agent...", self.agent_name)

            self._check_artifacts(context.artifacts)
            prompt = self.prompt_builder.build_prompt(
                artifacts=context.artifacts, context=None
            )

            # Construct the user message with language instruction
            user_message = f"Output language: {context.language}\n\n{prompt}"

            # Pydantic AI Agent returns a RunResult with .data as the structured model
            result = await self.agent.run(user_message)

            return AgentResult(
                agent_name=self.agent_name,
                analysis=result.output.analysis,
                analyzed_artifacts=[type(a).__name__ for a in context.artifacts],
            )

        except Exception as e:
            LOGGER.error(
                "Error running %s agent: %s",
                self.agent_name,
                traceback.format_exc(),
            )
            return AgentResult(agent_name=self.agent_name, error_message=str(e))

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