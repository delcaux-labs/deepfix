"""
Result management for DeepSight Advisor.

This module provides the AdvisorResult class and related functionality
for managing and serializing analysis results.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import json
import yaml
from pydantic import BaseModel, Field

from ...core.query.intelligence.models import IntelligenceResponse


class AdvisorResult(BaseModel):
    """Result object containing all information from an advisor analysis run."""

    # Basic information
    run_id: Optional[str] = Field(None, description="MLflow run ID that was analyzed")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the analysis was completed",
    )

    # Query information
    prompt_generated: Optional[str] = Field(
        default=None, description="Generated prompt from artifacts"
    )
    prompt_length: Optional[int] = Field(
        default=None, description="Length of the generated prompt in characters"
    )

    # Intelligence response
    response: Optional[IntelligenceResponse] = Field(
        default=None, description="Response from intelligence provider"
    )
    response_content: Optional[str] = Field(
        default=None, description="Text content of the intelligence response"
    )
    response_length: Optional[int] = Field(
        default=None, description="Length of the response content in characters"
    )

    # Execution information
    execution_time: float = Field(
        default=0.0, description="Total execution time in seconds"
    )
    query_generation_time: float = Field(
        default=0.0, description="Time spent generating query in seconds"
    )
    intelligence_execution_time: float = Field(
        default=0.0, description="Time spent executing intelligence query in seconds"
    )

    def set_prompt(self, prompt: str) -> None:
        """Set the generated prompt and calculate its length."""
        self.prompt_generated = prompt
        self.prompt_length = len(prompt) if prompt else 0

    def set_response(self, response: IntelligenceResponse) -> None:
        """Set the intelligence response and extract content."""
        self.response = response
        self.response_content = response.content if response else None
        self.response_length = (
            len(self.response_content) if self.response_content else 0
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis results."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "prompt_generated": self.prompt_generated is not None,
            "response_received": self.response is not None,
            "execution_time": self.execution_time,
        }

    def to_dict(self, include_content: bool = True) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result_dict = self.model_dump()

        # Convert datetime to ISO string
        result_dict["timestamp"] = self.timestamp.isoformat()

        # Optionally exclude large content fields
        if not include_content:
            result_dict.pop("prompt_generated", None)
            result_dict.pop("response_content", None)
            if "response" in result_dict and result_dict["response"]:
                result_dict["response"] = {
                    "provider": result_dict["response"].get("provider"),
                    "timestamp": result_dict["response"].get("timestamp"),
                    "content_length": len(result_dict["response"].get("content", "")),
                }

        return result_dict

    def to_json(
        self, file_path: Optional[Union[str, Path]] = None, include_content: bool = True
    ) -> str:
        """Convert result to JSON string or save to file."""
        result_dict = self.to_dict(include_content=include_content)
        json_str = json.dumps(result_dict, indent=2, default=str)

        if file_path:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(json_str)

        return json_str

    def to_yaml(
        self, file_path: Optional[Union[str, Path]] = None, include_content: bool = True
    ) -> str:
        """Convert result to YAML string or save to file."""
        result_dict = self.to_dict(include_content=include_content)
        yaml_str = yaml.dump(result_dict, default_flow_style=False, indent=2)

        if file_path:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(yaml_str)

        return yaml_str

    def to_text(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """Convert result to human-readable text format."""
        lines = []
        lines.append("=" * 60)
        lines.append("DEEPSIGHT ADVISOR ANALYSIS RESULT")
        lines.append("=" * 60)
        lines.append(f"Run ID: {self.run_id}")
        lines.append(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Execution Time: {self.execution_time:.2f} seconds")
        lines.append("")

        # Query section
        if self.prompt_generated:
            lines.append("PROMPT:")
            lines.append("-" * 20)
            lines.append(f"Length: {self.prompt_length} characters")
            lines.append("Content:")
            lines.append(self.prompt_generated)
            lines.append("")

        # Response section
        if self.response_content:
            lines.append("INTELLIGENCE RESPONSE:")
            lines.append("-" * 20)
            lines.append(f"Length: {self.response_length} characters")
            lines.append("Content:")
            lines.append(self.response_content)
            lines.append("")

        text_content = "\n".join(lines)

        if file_path:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(text_content)

        return text_content

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvisorResult":
        """Create AdvisorResult from dictionary."""
        # Convert string paths back to Path objects
        if "output_paths" in data:
            data["output_paths"] = {k: Path(v) for k, v in data["output_paths"].items()}

        # Convert ISO string back to datetime
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        return cls(**data)
