from typing import Dict, Any, Optional, Union
from rich.progress import Progress

from .models import IntelligenceResponse, IntelligenceProviders, IntelligenceConfig
from .providers import DspyLLMProvider, CursorAgentProvider


class IntelligenceClient:
    """Synchronous multi-provider client for LLMs and coding agents.

    Accepts a prompt string and executes it against the selected provider.
    """

    def __init__(self, config: IntelligenceConfig):
        self.config: IntelligenceConfig = config
        self.IntelligenceProviders: Dict[IntelligenceProviders, Any] = {}
        self.provider = self._initialize_provider()

    def _initialize_provider(self) -> Union[CursorAgentProvider, DspyLLMProvider]:
        # LLM
        if self.config.provider_name == IntelligenceProviders.LLM:
            return DspyLLMProvider(config=self.config.llm_config)
        # Coding Agent(s)
        elif self.config.provider_name == IntelligenceProviders.CURSOR:
            return CursorAgentProvider(config=self.config.cursor_config)
        else:
            raise NotImplementedError()

    def execute_query(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> IntelligenceResponse:
        with Progress() as progress:
            response = self.provider.execute(prompt, context or {})

        return response
