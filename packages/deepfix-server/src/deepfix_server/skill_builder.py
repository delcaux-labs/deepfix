"""Skill builder module for DeepFix autonomous fix system."""

import pathlib
import deepfix_kb
from openhands.sdk.context import AgentContext
from openhands.sdk.skills import Skill, load_skills_from_dir


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


class DeepFixSkillBuilder:
    """Builder for loading DeepFix Skills from deepfix-kb and constructing OpenHands AgentContext."""

    def __init__(self, skills_dir: pathlib.Path | str | None = None) -> None:
        """Initialize DeepFixSkillBuilder.

        Args:
            skills_dir: Optional custom path to skills directory. Defaults to
                       packages/deepfix-kb/src/deepfix_kb/skills/.
        """
        if skills_dir is None:
            self.skills_dir = pathlib.Path(deepfix_kb.__file__).parent / "skills"
        else:
            self.skills_dir = pathlib.Path(skills_dir)

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
        )
