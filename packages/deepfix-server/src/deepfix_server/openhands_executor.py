import asyncio
import pathlib
from typing import Any

from openhands.sdk import LLM, Agent, AgentContext, Conversation, Workspace
from openhands.sdk.conversation.goal import run_goal
from openhands.sdk.skills import Skill, load_skills_from_dir
from openhands.sdk.tool import Tool
from openhands.tools import FileEditorTool, TaskTrackerTool, TerminalTool
from openhands.workspace import DockerWorkspace
from pydantic import SecretStr

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
    "1. DeepFix SDK & Diagnostics: Use the `deepfix-sdk` skill (e.g. `deepfix-sdk diagnose --dataset <name>`) "
    "to inspect data integrity, multicollinearity, and model evaluation metrics.\n"
    "2. Data Access: Use the `mlflow-data-access` skill to load models and datasets if needed.\n"
    "3. Sandboxed Execution: Execute candidate fix scripts in your terminal. Ensure your script evaluates the model and captures the metrics.\n"
    "4. S3 Persistence: Use the `s3-weights-storage` skill (`push_weights_to_s3.py`) to persist trained weights.\n"
    "5. Completion: When you are done fixing the model, you MUST use the `deepfix-communication` skill to report your status using the webhook script.\n"
)


class OpenHandsExecutor:
    """Manages the lifecycle of OpenHands autonomous fix sessions using the OpenHands SDK."""

    _active_tasks: dict[str, asyncio.Task] = {}

    def __init__(self, config: AutonomousFixConfig):
        self.config = config
        self.skills_dir = pathlib.Path(__file__).parent / "skills"

    @classmethod
    def register_task(cls, job_id: str, task: asyncio.Task) -> None:
        """Register an active background fix task."""
        cls._active_tasks[job_id] = task

    @classmethod
    def unregister_task(cls, job_id: str) -> None:
        """Unregister a completed or cancelled fix task."""
        cls._active_tasks.pop(job_id, None)

    @classmethod
    def cancel_task(cls, job_id: str) -> bool:
        """Cancel a running fix task by job ID.

        Returns:
            bool: True if a running task was found and cancelled, False otherwise.
        """
        task = cls._active_tasks.get(job_id)
        if task and not task.done():
            LOGGER.info(f"Cancelling active OpenHands fix task for {job_id}")
            task.cancel()
            return True
        return False

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
            load_memory=self.config.load_memory,
        )

    def _build_system_prompt(
        self,
        job_id: str,
        diagnosis: str = "",
        dataset_name: str | None = None,
        model_name: str | None = None,
        target_metric: str | None = "accuracy",
        target_value: float | None = 0.90,
        max_iterations: int | None = 5,
        s3_bucket: str | None = None,
        dataset_uri: str | None = None,
        model_uri: str | None = None,
        is_dataset_only: bool = False,
    ) -> str:
        """Constructs the structured system prompt instructing OpenHands on what to do.

        Compiles diagnostic findings and modeling remediation guidelines explicitly
        addressing key failure modes (multicollinearity, class imbalance, validation,
        dataset-only mode, and S3 persistence).
        """
        effective_bucket = s3_bucket or self.config.s3_bucket
        bucket_info = (
            f"Target S3 Bucket: {effective_bucket}"
            if effective_bucket
            else "S3 Bucket: (check AWS_S3_BUCKET / DEEPFIX_S3_BUCKET env)"
        )
        mode_label = "DATASET-ONLY REPAIR MODE" if (is_dataset_only or not model_name) else "MODEL REPAIR & TRAINING MODE"

        diagnosis_section = (
            diagnosis.strip()
            if diagnosis.strip()
            else (
                "No pre-computed diagnostic report was attached. Use the `deepfix-sdk` skill "
                f"(e.g. `deepfix-sdk diagnose --dataset {dataset_name or 'dataset'}`) to analyze data integrity, "
                "multicollinearity, drift, and validation issues before implementing fixes."
            )
        )

        dataset_info = f"- Dataset Name: `{dataset_name}`" if dataset_name else "- Dataset: registered in MLflow / S3"
        if dataset_uri:
            dataset_info += f" (URI: `{dataset_uri}`)"

        model_info = (
            f"- Model Name: `{model_name}`" if model_name else "- Model: None (Focus on dataset partitioning & preprocessing repair)"
        )
        if model_uri:
            model_info += f" (URI: `{model_uri}`)"

        prompt = f"""You are an autonomous Machine Learning Engineer and Data Quality Specialist for DeepFix.
Your objective is to repair and improve the ML pipeline for job `{job_id}` based on the following task specifications and diagnostic findings.

================================================================================
JOB SPECIFICATION & CONFIGURATION
================================================================================
- Mode: {mode_label}
- Job ID: `{job_id}`
{dataset_info}
{model_info}
- Target Metric: `{target_metric or 'accuracy'}` (Target Threshold: `{target_value or 0.90}`)
- Max Iterations: {max_iterations or 5}
- {bucket_info}

================================================================================
DIAGNOSIS FINDINGS
================================================================================
{diagnosis_section}

================================================================================
FAILURE-MODE SPECIFIC REMEDIATION GUIDELINES
================================================================================
You MUST explicitly inspect and remediate the following potential failure modes:

1. MULTICOLLINEARITY & HIGH-PPS FEATURE REDUNDANCY:
   - When severe multicollinearity, geometric feature redundancy, or high Predictive Power Score (PPS) between features is detected:
     a. Apply L1/L2 regularization (e.g. Lasso, Ridge, ElasticNet, or regularized linear models) to shrink redundant feature weights.
     b. Perform feature reduction / feature selection: drop redundant high-PPS or collinear features without losing critical signal.
     c. Utilize tree-based ensemble models (e.g. LightGBM, XGBoost, CatBoost, RandomForestClassifier) that are inherently robust to collinear feature sets.

2. CLASS IMBALANCE REMEDIATION:
   - When the target distribution is imbalanced or skewed:
     a. Implement class weighting (`class_weight='balanced'`, `scale_pos_weight`) or cost-sensitive loss formulations.
     b. Do NOT rely purely on overall accuracy; prioritize balanced evaluation metrics: Recall, Precision, Macro/Weighted F1-Score, and ROC-AUC.
     c. Ensure minority class representations are preserved across all folds and evaluation splits.

3. LEAK-FREE VALIDATION STRATEGY & SMALL SAMPLE SIZE:
   - To ensure reliable evaluation and prevent data leakage:
     a. Use Stratified K-Fold Cross-Validation (e.g. `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) instead of a single unstable train/test split.
     b. Fit all preprocessing pipelines, scalers (e.g. `StandardScaler`, `RobustScaler`), and encoders strictly on the training fold, and transform the validation fold to avoid data snooping.

4. DATASET-ONLY MODE (when model is omitted / not provided):
   - When running in Dataset-Only Repair Mode:
     a. Focus strictly on dataset partitioning and preprocessing repair.
     b. Eliminate train-test sample leakage and duplicate rows.
     c. Construct leak-free stratified splits that strictly preserve class balance.
     d. Filter out multicollinear, corrupted, or constant columns.
     e. Save the repaired partitioned datasets (e.g. `train.parquet`, `test.parquet` or dataset directory) and push to S3 using `s3-weights-storage`.

5. S3 ARTIFACT PERSISTENCE:
   - Push your best trained model checkpoint weights (e.g. `model.pt`, `model.joblib`, `model.pkl`) or repaired partitioned datasets to S3:
     `python s3-weights-storage/push_weights_to_s3.py --weights-path <path_to_weights_or_data> --job-id {job_id}`
   - Capture the output S3 URI (`s3://...`).

6. WEBHOOK COMPLETION REPORTING:
   - When finished iterating (or upon hitting the target metric or maximum iterations), you MUST send the final report payload to the DeepFix Server:
     `python deepfix-communication/report_completion.py --job-id {job_id} --status COMPLETED --s3-weights-uri <s3_uri> --final-metrics '<json_metrics>' --applied-fixes <fix_1> <fix_2> ... --summary '<summary_text>'`

================================================================================
EXECUTION STEPS
================================================================================
1. Data Retrieval: Use the `mlflow-data-access` skill (`download_dataset.py` or Python API) or local S3 path to load the dataset and baseline artifacts.
2. Formulate Strategy: Review the diagnostic findings above, identify the primary failure modes, and choose the appropriate remediation techniques.
3. Implement & Execute: Write candidate script `train.py` (or `repair_dataset.py`) in your terminal, run training experiments, and track metrics.
4. Iterate & Evaluate: Evaluate intermediate metrics across iterations against the target `{target_metric}` >= `{target_value}`. Refine features or hyperparameters as needed.
5. Push to S3 & Report: Upload winning model weights or dataset to S3, then trigger `report_completion.py` to conclude the autonomous session.
"""
        return prompt.strip()

    async def launch_autonomous_fix(
        self,
        job_id: str,
        diagnosis: str = "",
        mlflow_experiment_id: int | str = 0,
        s3_bucket: str | None = None,
        mlflow_tracking_uri: str | None = None,
        dataset_name: str | None = None,
        model_name: str | None = None,
        target_metric: str | None = "accuracy",
        target_value: float | None = 0.90,
        max_iterations: int | None = 5,
        dataset_uri: str | None = None,
        model_uri: str | None = None,
        is_dataset_only: bool = False,
    ) -> None:
        """Launches the OpenHands agent to fix the identified issues.

        Args:
            job_id: The unique identifier for this fix job.
            diagnosis: The prior diagnostic findings from DeepFix Server.
            mlflow_experiment_id: MLflow experiment ID.
            s3_bucket: Target S3 bucket for model weights persistence.
            mlflow_tracking_uri: MLflow tracking server URI.
            dataset_name: Name of the dataset being repaired.
            model_name: Name of the baseline model, if any.
            target_metric: Key of target evaluation metric.
            target_value: Target metric threshold.
            max_iterations: Maximum allowed refinement loops.
            dataset_uri: URI to dataset in S3 or local path.
            model_uri: URI to baseline model artifact.
            is_dataset_only: Whether the job is in dataset-only mode.
        """
        LOGGER.info(f"Preparing autonomous fix session for {job_id}")
        sandbox_env = self.config.get_sandbox_environment(
            job_id=job_id,
            mlflow_experiment_id=mlflow_experiment_id,
            s3_bucket=s3_bucket,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
        for k, v in sandbox_env.items():
            import os
            os.environ[k] = str(v)

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

        effective_max_iterations = max_iterations or self.config.max_fix_iterations
        is_ds_only = is_dataset_only or (model_name is None)

        system_prompt = self._build_system_prompt(
            job_id=job_id,
            diagnosis=diagnosis,
            dataset_name=dataset_name,
            model_name=model_name,
            target_metric=target_metric,
            target_value=target_value,
            max_iterations=effective_max_iterations,
            s3_bucket=s3_bucket or self.config.s3_bucket,
            dataset_uri=dataset_uri,
            model_uri=model_uri,
            is_dataset_only=is_ds_only,
        )

        try:
            # Configure OpenHands agent with essential tools and domain skills
            agent = Agent(
                llm=llm,
                tools=[
                    Tool(name=TerminalTool.name),
                    Tool(name=TaskTrackerTool.name),
                    Tool(name=FileEditorTool.name),
                ],
                agent_context=self.build_agent_context(),
            )

            # Build conversation kwargs for persistence
            conversation_kwargs = {
                "persistence_dir": self.config.persistence_dir,
                "conversation_id": job_id,
            }

            forward_env_keys = list(
                set(
                    [
                        "DEBUG",
                        "SESSION_API_KEY",
                        "OH_SESSION_API_KEYS_0",
                    ]
                    + list(sandbox_env.keys())
                )
            )

            def run_sync():
                if self.config.openhands_use_local_server:
                    workspace = Workspace(host=self.config.openhands_server_url)
                    conversation = Conversation(
                        agent=agent, workspace=workspace, **conversation_kwargs
                    )

                    outcome = run_goal(
                        conversation=conversation,
                        objective=system_prompt,
                        judge_llm=judge_llm,
                        max_iterations=effective_max_iterations,
                    )
                else:
                    with DockerWorkspace(
                        server_image=self.config.openhands_docker_image,
                        host_port=self.config.openhands_sandbox_port,
                        platform=detect_platform(),
                        forward_env=forward_env_keys,
                        network=self.config.openhands_container_network,
                    ) as workspace:
                        conversation = Conversation(
                            agent=agent, workspace=workspace, **conversation_kwargs
                        )
                        outcome = run_goal(
                            conversation=conversation,
                            objective=system_prompt,
                            judge_llm=judge_llm,
                            max_iterations=effective_max_iterations,
                        )

                LOGGER.info(
                    f"OpenHands Goal finished for {job_id}: status={outcome.status}, iterations={outcome.iterations}"
                )

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, run_sync)

        except asyncio.CancelledError:
            LOGGER.info(f"OpenHands execution task for {job_id} was cancelled.")
            raise
        except Exception as e:
            LOGGER.exception(f"Failed to run OpenHands agent for {job_id}: {e}")
            raise
        finally:
            self.unregister_task(job_id)
