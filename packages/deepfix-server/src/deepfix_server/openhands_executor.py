import asyncio
import logging
import os
import sys
from typing import Any, Optional, Union

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, AgentContext, Conversation, Workspace
from openhands.workspace import DockerWorkspace
from openhands.sdk.conversation.goal import run_goal
from openhands.tools import FileEditorTool, TerminalTool, TaskTrackerTool
from openhands.sdk.tool import Tool
from openhands.sdk.utils.async_utils import AsyncCallbackWrapper

import pathlib
import deepfix_kb
from openhands.sdk.context import AgentContext
from openhands.sdk.skills import Skill, load_skills_from_dir

from deepfix_core.models.api import APIResponse
from .config import AutonomousFixConfig
from .logging import get_logger


LOGGER = get_logger(__name__)

def detect_platform():
    import platform
    """Detects the correct Docker platform string."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"

DEFAULT_AUTONOMOUS_FIX_SYSTEM_SUFFIX = (
    "You are an autonomous ML model fix agent for DeepFix.\n"
    "Your objective is to diagnose performance bottlenecks, generate code fix patches, "
    "run sandboxed experiments with MLflow tracking, evaluate metric improvements across iterations, "
    "and iterate until you are satisfied.\n\n"
    "Workflow Guidelines:\n"
    "1. Data Access: Use the `mlflow-data-access` skill to load models and datasets if needed.\n"
    "2. Sandboxed Execution: Execute candidate fix scripts in your terminal. Ensure your script evaluates the model and captures the metrics.\n"
    "3. Completion: When you are done fixing the model, you MUST use the `deepfix-communication` skill to report your status using the webhook script.\n"
)


class OpenHandsExecutor:
    """Manages the lifecycle of OpenHands autonomous fix sessions using the OpenHands SDK."""

    def __init__(self, config: AutonomousFixConfig):
        self.config = config
        self.skills_dir = pathlib.Path(__file__).parent / "skills"

    def load_skills(self) -> list[Skill]:
        """Load skills from the skills directory using OpenHands SDK load_skills_from_dir.

        Returns:
            List of loaded Skill objects.

        Raises:
            FileNotFoundError: If skills_dir does not exist.
        """
        if not self.skills_dir.exists():
            raise FileNotFoundError(f"Skills directory does not exist: {self.skills_dir}")

        repo_skills, knowledge_skills, agent_skills = load_skills_from_dir(self.skills_dir)

        loaded_skills: list[Skill] = []
        for skill_dict in (repo_skills, knowledge_skills, agent_skills):
            loaded_skills.extend(skill_dict.values())
        return loaded_skills

    def build_agent_context(
        self,
        system_message_suffix: str | None = None,
        load_public_skills: bool = False,
    ) -> AgentContext:
        """Build OpenHands AgentContext populated with loaded DeepFix skills.

        Args:
            system_message_suffix: Optional system message instructions for agent workflow.
                                  Defaults to DEFAULT_AUTONOMOUS_FIX_SYSTEM_SUFFIX.
            load_public_skills: Whether to load public skills into context (default False).

        Returns:
            Configured AgentContext instance.
        """
        skills = self.load_skills()
        suffix = system_message_suffix or DEFAULT_AUTONOMOUS_FIX_SYSTEM_SUFFIX

        return AgentContext(
            skills=skills,
            load_public_skills=load_public_skills,
            system_message_suffix=suffix,
            load_memory=self.config.load_memory
        )

    async def launch_autonomous_fix(self, job_id: str, diagnosis_response: APIResponse, mlflow_experiment_id:int=0) -> None:
        """Launches the OpenHands agent to fix the identified issues.
        
        Args:
            job_id: The unique identifier for this fix job.
            diagnosis_response: The prior diagnostic findings from DeepFix Server.
        """
        LOGGER.info("Preparing autonomous fix session", job_id=job_id)
        self.config.setup_otel_environment(mlflow_experiment_id)

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
            agent.agent_context = self.build_agent_context()

            # Build conversation kwargs for persistence
            conversation_kwargs = {"persistence_dir":self.config.persistence_dir, "conversation_id":job_id}

            def run_sync():
                if self.config.openhands_use_local_server:
                    workspace = Workspace(host=self.config.openhands_server_url)
                    conversation = Conversation(agent=agent, workspace=workspace, **conversation_kwargs)
                    
                    outcome = run_goal(
                        conversation=conversation,
                        objective=system_prompt,
                        judge_llm=judge_llm,
                        max_iterations=10,
                    )
                else:
                    with DockerWorkspace(
                        server_image=self.config.openhands_docker_image,
                        host_port=self.config.openhands_sandbox_port,
                        platform=detect_platform(),
                    ) as workspace:
                        conversation = Conversation(agent=agent, workspace=workspace, **conversation_kwargs)
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

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, run_sync)

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
