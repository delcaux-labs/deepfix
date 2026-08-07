import asyncio
import sys
import os

from deepfix_server.config import settings
from deepfix_server.openhands_executor import OpenHandsExecutor
from deepfix_server.logging import setup_mlflow_tracing

async def main():
    print("==================================================")
    print("Starting OpenHands Agent Debug Script")
    print("==================================================")

    # Resolve paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    diagnosis_path = os.path.join(project_root, "saved_diagnosis.txt")
    
    if not os.path.exists(diagnosis_path):
        print(f"Error: Could not find diagnosis file at {diagnosis_path}")
        print("Please run the e2e tests (e.g. pytest tests/test_tabular_workflow_e2e.py) first to generate the diagnosis file.")
        sys.exit(1)
        
    with open(diagnosis_path, "r", encoding="utf-8") as f:
        diagnosis = f.read()

    print(f"Loaded diagnosis ({len(diagnosis)} bytes)")
    
    # Initialize the executor
    print("Initializing OpenHandsExecutor...")
    fix_config = settings.get_autonomous_fix_config()
    executor = OpenHandsExecutor(config=fix_config)
    
    job_id = "debug_job_local"
    
    # Set up LLM tracing if MLFlow is configured
    if settings.mlflow_exp_name and settings.mlflow_tracking_uri:
        setup_mlflow_tracing(
            experiment_name=settings.mlflow_exp_name,
            tracking_uri=settings.mlflow_tracking_uri,
        )

    print(f"Launching agent with job_id: {job_id}")
    print("To stop the agent early, you can press Ctrl+C.")
    
    try:
        await executor.launch_autonomous_fix(job_id=job_id, diagnosis=diagnosis)
        print("Agent run completed successfully.")
    except KeyboardInterrupt:
        print("\nAgent run stopped by user.")
    except Exception as e:
        print(f"Error running agent: {e}")

if __name__ == "__main__":
    asyncio.run(main())
