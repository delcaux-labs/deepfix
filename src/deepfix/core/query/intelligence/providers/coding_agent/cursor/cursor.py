"""Main Cursor CLI integration class."""

from typing import Dict, Any, Optional, List, Union
import time
import traceback

from .process import ProcessManager
from .errors import CursorResponseError

from ....models import (
    IntelligenceResponse,
    IntelligenceProviderError,
    IntelligenceProviders,
    Capabilities,
    CursorConfig,
)
from ...base import BaseProvider


class Cursor:
    """Main integration class for Cursor CLI non-interactive mode."""

    def __init__(
        self,
        model: str = "auto",
        output_format: str = "text",
        timeout: int = 300,
        cli_path: str = "cursor-agent",
        working_directory: Optional[str] = None,
        config: Optional[CursorConfig] = None,
        **kwargs,
    ):
        """Initialize Cursor integration.

        Args:
            model: AI model to use (e.g., "auto", "gpt-5", "sonnet-4", "opus-4.1")
            output_format: Output format ("text", "json", "markdown")
            timeout: Timeout in seconds for CLI operations
            cli_path: Path to cursor-agent executable
            working_directory: Working directory for CLI operations
            **kwargs: Additional arguments to pass to Cursor CLI
        """
        if config is None:
            self.config = CursorConfig(
                model=model,
                output_format=output_format,
                timeout=timeout,
                cli_path=cli_path,
                working_directory=working_directory,
                additional_args=kwargs if kwargs else None,
            )
        else:
            self.config = config

        self.process_manager = ProcessManager(cli_path=cli_path)

    def query(self, prompt: str) -> str:
        """Send a query to Cursor CLI and return the response.

        Args:
            prompt: The prompt/query to send to Cursor

        Returns:
            The response from Cursor CLI

        Raises:
            CursorError: If there's an error with the CLI operation
            CursorResponseError: If the response indicates an error
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Get CLI arguments from config
        cli_args = self.config.to_cli_args()

        # Execute the command
        stdout, stderr, return_code = self.process_manager.execute(
            args=cli_args,
            prompt=prompt,
            timeout=self.config.timeout,
            working_directory=self.config.working_directory,
        )

        # Check for errors
        if return_code != 0:
            error_msg = stderr.strip() if stderr.strip() else "Unknown error"
            raise CursorResponseError(f"Cursor CLI returned error: {error_msg}")

        # Return the response
        response = stdout.strip()
        if not response:
            raise CursorResponseError("Empty response from Cursor CLI")

        return response

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                # Add to additional_args if not a standard config parameter
                if self.config.additional_args is None:
                    self.config.additional_args = {}
                self.config.additional_args[key] = value


class CursorAgentProvider(BaseProvider):
    def __init__(self, config: CursorConfig):
        self.config = config
        self.agent = Cursor(config=config)

    def execute(
        self, prompt: str, context: Optional[Dict[str, Any]] = None
    ) -> IntelligenceResponse:
        start = time.time()
        try:
            enhanced_prompt = self._enhance_prompt_for_coding(prompt, context or {})
            resp = self.agent.query(prompt=enhanced_prompt)
            latency_ms = int((time.time() - start) * 1000)
            return IntelligenceResponse(
                content=resp,
                provider=f"{IntelligenceProviders.CURSOR.value}::{self.config.model}",
                latency_ms=latency_ms,
            )
        except Exception:
            raise IntelligenceProviderError(
                f"Cursor agent failed: {traceback.format_exc()}"
            )

    def _enhance_prompt_for_coding(self, prompt: str, context: Dict[str, Any]) -> str:
        parts = [
            "You are an expert data scientist with 10 years experience debugging AI models.",
            "You provide actionable real-world solutions and resolution guidance.",
            prompt,
            self.instructions,
        ]
        if context.get("code_context"):
            parts.insert(-1, f"\nCode context: {context['code_context']}")
        return "\n".join(parts)

    def get_capabilities(self) -> List[Capabilities]:
        return [
            Capabilities.CODE_GENERATION,
            Capabilities.DEBUGGING,
            Capabilities.REASONING,
            Capabilities.TEXT_GENERATION,
        ]

    @property
    def instructions(self) -> str:
        return """ONLY ANSWER BASED ON THE PROVIDED INFORMATION. DO NOT MAKE UP ANYTHING or READ ANY OTHER FILES.
                 DO NOT CREATE ANY NEW FILES OR DIRECTORIES.
                 DO NOT EDIT ANY FILES.
                 DO NOT DELETE ANY FILES.
                 DO NOT RENAME ANY FILES.
                 DO NOT MOVE ANY FILES.
                 DO NOT COPY ANY FILES.
                 DO NOT PASTE ANY FILES.
                 ANSWER IN PLAIN TEXT.              
        """
