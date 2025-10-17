from fastapi import FastAPI
import traceback

from .coordinators import ArtifactAnalysisCoordinator
from ..shared.models import APIRequest, APIResponse
from .models import AgentContext
from .config import LLMConfig

app = FastAPI()

@app.get("/info")
def read_root():
    return {"message": "DeepFix API is running"}

@app.post("/v1/analyze")
def analyze_artifacts(request: APIRequest):

    try:
        llm_config = LLMConfig.load_from_env()
        coordinator = ArtifactAnalysisCoordinator(llm_config=llm_config)
        context = AgentContext(dataset_artifacts=request.dataset_artifacts,
        training_artifacts=request.training_artifacts, 
        deepchecks_artifacts=request.deepchecks_artifacts, 
        model_checkpoint_artifacts=request.model_checkpoint_artifacts, 
        dataset_name=request.dataset_name)
        results = coordinator.run(context)
        response = APIResponse(agent_results=context.agent_results, summary=results.summary, additional_outputs=results.additional_outputs, error_message=results.error_message)
    except Exception as e:
        error_message = f"Error analyzing artifacts: {e}"
        response =  APIResponse(error_message=error_message)

    return response
