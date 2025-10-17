from typing import List, Union, Optional, Dict, Any
from .base import Step
from ..prompt_builders import PromptBuilder
from ..config import PromptConfig


class BuildPrompt(Step):
    def __init__(self, config: Optional[PromptConfig] = None):
        self.builder = PromptBuilder(config=config)

    def run(
        self,
        context: dict,
        artifacts: Optional[List] = None,
        query_context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> dict:
        prompt = self.builder.build_prompt(
            artifacts=artifacts or context.get("artifacts"), context=query_context
        )
        context["prompt"] = prompt
        return context
