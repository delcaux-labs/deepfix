from typing import Optional
from .base import Step
import time
from ..query.intelligence.models import IntelligenceConfig
from ..query.intelligence.client import IntelligenceClient
from ...utils.logging import get_logger

LOGGER = get_logger(__name__)


class Query(Step):
    def __init__(self, config: IntelligenceConfig, run_id: Optional[str] = None):
        self.config = config
        self.intelligence_client = IntelligenceClient(self.config)
        self.run_id = run_id

    def run(self, context: dict, prompt: Optional[str] = None) -> dict:
        prompt = prompt or context.get("prompt")
        self.run_id = self.run_id or context.get("run_id")
        assert prompt is not None, "prompt must be provided"
        result = self.execute_query(prompt)
        context["advisor_result"] = result
        return result

    def execute_query(self, prompt: str):
        """
        Execute query using configured intelligence provider.

        Args:
            prompt: Generated prompt to execute
            result: Result object to update with response information
        """
        # Local import to avoid circular dependency
        from ..advisor.result import AdvisorResult
        from ..advisor.errors import IntelligenceError
        
        start_time = time.time()
        result = AdvisorResult(run_id=self.run_id)

        try:
            # Prepare execution context
            context = self.config.context or {}
            # Execute query
            response = self.intelligence_client.execute_query(
                prompt=prompt,
                context=context,
            )

            # Update result
            result.set_response(response)
            result.set_prompt(prompt)
            result.intelligence_execution_time = time.time() - start_time

            LOGGER.info(
                "Query executed successfully in %.2f seconds",
                result.intelligence_execution_time
            )
            LOGGER.debug("Response length: %d characters", result.response_length)
            return result

        except Exception as e:
            raise IntelligenceError(f"Query execution failed: {e}") from e
