import asyncio
import logging
import os
import sys
from typing import Any, Optional, Union

import structlog
from pydantic import SecretStr

from openhands.sdk import LLM, Agent, AgentContext, Conversation, Workspace
from openhands.sdk.conversation.goal import run_goal
from openhands.sdk.tools import FileEditorTool, TerminalTool, TaskTrackerTool
from openhands.sdk.tools.core import Tool


from deepfix_core.models.api import APIResponse
from .config import AutonomousFixConfig
from .skill_builder import DeepFixSkillBuilder
from .logging import get_logger


LOGGER = get_logger(__name__)


class OpenHandsExecutor:
    """Manages the lifecycle of OpenHands autonomous fix sessions using the OpenHands SDK."""

    def __init__(self, config: AutonomousFixConfig):
        self.config = config
        self.skill_builder = DeepFixSkillBuilder()

    async def launch_autonomous_fix(self, job_id: str, diagnosis_response: APIResponse) -> None:
        """Launches the OpenHands agent to fix the identified issues.
        
        Args:
            job_id: The unique identifier for this fix job.
            diagnosis_response: The prior diagnostic findings from DeepFix Server.
        """
        LOGGER.info("Preparing autonomous fix session", job_id=job_id)
        
        # We don't have experiment_id natively in diagnosis_response, so default to 0
        exp_id = 0
        self.config.setup_otel_environment(exp_id)

        llm_kwargs: dict[str, Any] = {"model": self.config.openhands_llm_model}
        if self.config.openhands_llm_api_key:
            llm_kwargs["api_key"] = SecretStr(self.config.openhands_llm_api_key)
        if self.config.openhands_llm_base_url:
            llm_kwargs["base_url"] = self.config.openhands_llm_base_url
        
        agent_llm_kwargs = llm_kwargs.copy()
        agent_llm_kwargs["usage_id"] = "agent"
        llm = LLM(**agent_llm_kwargs)
        
        judge_llm_kwargs = llm_kwargs.copy()
        judge_llm_kwargs["usage_id"] = "goal-judge"
        judge_llm = LLM(**judge_llm_kwargs)

        system_prompt = self._build_system_prompt(job_id, diagnosis_response)
        
        try:
            # We configure the OpenHands agent with essential standard tools
            agent = Agent(
                llm=llm,
                tools=[
                    Tool(name=TerminalTool.name), 
                    Tool(name=TaskTrackerTool.name),
                    Tool(name=FileEditorTool.name)
                ],
            )
            # Equip our domain-specific skills
            agent.agent_context = self.skill_builder.build_agent_context()

            # Initialize a connection to the sandbox
            workspace = Workspace(host=self.config.openhands_server_url)
            conversation = Conversation(agent=agent, workspace=workspace)

            # Iterate until goal is reached
            outcome = run_goal(
                conversation=conversation,
                objective=system_prompt,
                judge_llm=judge_llm,
                max_iterations=10,
            )
            
            LOGGER.info(
                "OpenHands Goal finished",
                job_id=job_id,
                status=outcome.status,
                iterations=outcome.iterations,
            )
        except Exception as e:
            LOGGER.exception("Failed to run OpenHands agent", job_id=job_id, error=str(e))

    def _build_system_prompt(self, job_id: str, diagnosis_response: APIResponse) -> str:
        """Constructs the system prompt instructing OpenHands on what to do."""
        diagnosis_text = diagnosis_response.get_results_as_text()
        
        prompt = f"""
                    You are an autonomous Machine Learning engineer. Your goal is to apply fixes to a model based on the following diagnostic findings:

                    === DIAGNOSIS FINDINGS ===
                    {diagnosis_text}
                    ==========================

                    You are executing in a sandboxed environment with access to specific tools.

                    CRITICAL INSTRUCTIONS:
                    1. Use your `mlflow-data-access` skill to download datasets and models from MLflow.
                    2. Iterate on the training script (`train.py`) to resolve the issues found above. Run the script to evaluate metrics.
                    3. When you have achieved a satisfactory fix, or exhausted all possibilities, you MUST communicate the result back to the server.
                    4. Use your `deepfix-communication` skill to run `report_completion.py`. Your Job ID is: {job_id}.
                    """
        return prompt.strip()
