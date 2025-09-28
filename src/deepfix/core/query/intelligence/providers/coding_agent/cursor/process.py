"""Process management utilities for Cursor CLI integration."""

import subprocess
import shutil
from typing import Optional, Tuple
from .errors import CursorError, CursorTimeoutError, CursorNotFoundError


class ProcessManager:
    """Manages subprocess execution for Cursor CLI."""

    def __init__(self, cli_path: str = "cursor-agent"):
        """Initialize process manager.

        Args:
            cli_path: Path to the cursor-agent executable
        """
        self.cli_path = cli_path
        self._check_cli_available()

    def _check_cli_available(self) -> None:
        """Check if Cursor CLI is available."""
        if not shutil.which(self.cli_path):
            raise CursorNotFoundError(
                f"Cursor CLI not found at '{self.cli_path}'. "
                "Please ensure cursor-agent is installed and in your PATH."
            )

    def execute(
        self,
        args: list[str],
        prompt: str,
        timeout: int = 300,
        working_directory: Optional[str] = None,
    ) -> Tuple[str, str, int]:
        """Execute Cursor CLI command.

        Args:
            args: CLI arguments (without the prompt)
            prompt: The prompt to send to Cursor
            timeout: Timeout in seconds
            working_directory: Working directory for the process

        Returns:
            Tuple of (stdout, stderr, return_code)

        Raises:
            CursorTimeoutError: If the process times out
            CursorError: If there's an error executing the process
        """
        # Add the prompt as the last argument
        full_args = args + [prompt]

        try:
            result = subprocess.run(
                full_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_directory,
                encoding="utf-8",
            )

            return result.stdout, result.stderr, result.returncode

        except subprocess.TimeoutExpired as e:
            raise CursorTimeoutError(
                f"Cursor CLI operation timed out after {timeout} seconds"
            ) from e
        except subprocess.CalledProcessError as e:
            raise CursorError(
                f"Cursor CLI failed with return code {e.returncode}: {e.stderr}"
            ) from e
        except Exception as e:
            raise CursorError(f"Unexpected error executing Cursor CLI: {e}") from e
