from typing import Dict, Any, Optional, List, Union
import time
import dspy

from ...models import (
    IntelligenceResponse,
    IntelligenceProviderError,
    Capabilities,
    LLMConfig,
)
from ..base import BaseProvider


class BugResolutionRecommendation(dspy.Signature):
    "Provides guidance on bug resolution in machine learning model"

    # inputs
    prompt: str = dspy.InputField(description="prompt with instructions")
    context: Optional[Dict[str, Union[str, int, float, dspy.Image]]] = dspy.InputField(
        description="additional context"
    )

    # outputs
    recommendation: str = dspy.OutputField(
        description="Recommendation to resolve bug in the model"
    )


class DspyRouter:
    """Minimal DSPy router stub to unify multiple LLM backends."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = self._initialize_module()

    def _initialize_module(
        self,
    ):
        dspy.settings.configure(
            lm=dspy.LM(
                model=self.config.model_name,
                cache=self.config.cache,
                api_base=self.config.base_url,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
            ),
            track_usage=self.config.track_usage,
        )
        return dspy.ChainOfThought(BugResolutionRecommendation)

    def generate(self, **kwargs) -> str:
        return self.llm(**kwargs)


class DspyLLMProvider(BaseProvider):
    """DSPy-backed LLM provider wrapping multiple backends."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.router = DspyRouter(config=config)

    def execute(self, **kwargs) -> IntelligenceResponse:
        start = time.time()
        try:
            result = self.router.generate(**kwargs)
            latency_ms = int((time.time() - start) * 1000)
            return IntelligenceResponse(
                content=result,
                provider=f"dspy::{self.config.model_name}",
                latency_ms=latency_ms,
            )
        except Exception as e:
            raise IntelligenceProviderError(f"DSPy provider failed: {e}")

    def get_capabilities(self) -> List[Capabilities]:
        return [
            Capabilities.TEXT_GENERATION,
            Capabilities.REASONING,
            Capabilities.IMAGE_UNDERSTANDING,
        ]
