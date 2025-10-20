from fastapi import FastAPI
import litserve as ls
from fastapi import HTTPException
import traceback
from .coordinators import ArtifactAnalysisCoordinator
from ..shared.models import APIRequest, APIResponse
from .models import AgentContext
from .config import LLMConfig
from .logging import get_logger

LOGGER = get_logger(__name__)


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
                                dataset_name=request.dataset_name
                            )
        results = coordinator.run(context)

        response = APIResponse(agent_results=context.agent_results, 
                                summary=results.summary, 
                                additional_outputs=results.additional_outputs, 
                                error_messages=results.get_error_messages(),
                                dataset_name=request.dataset_name
                            )
    except Exception as e:
        LOGGER.error(traceback.format_exc())
        response =  APIResponse(error_messages={'error':traceback.format_exc()})
    
    return response

# Litserve API
class InferenceAPI(ls.LitAPI):
    """Inference API."""

    def setup(self, device):
        llm_config = LLMConfig.load_from_env()
        self.coordinator = ArtifactAnalysisCoordinator(llm_config=llm_config)

    def decode_request(self, request: APIRequest):
        try:
            return AgentContext(dataset_artifacts=request.dataset_artifacts,
                                training_artifacts=request.training_artifacts, 
                                deepchecks_artifacts=request.deepchecks_artifacts, 
                                model_checkpoint_artifacts=request.model_checkpoint_artifacts, 
                                dataset_name=request.dataset_name
                            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error decoding request: {e}",
            )

    def predict(self, context: AgentContext) -> APIResponse:
        try:
            results = self.coordinator.run(context)
            response = APIResponse(agent_results=results.get_agent_results(), 
                                summary=results.summary, 
                                additional_outputs=results.additional_outputs, 
                                error_messages=results.get_error_messages(),
                                dataset_name=context.dataset_name
                            )
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=traceback.format_exc())


def run_inference_api(port=4141, host="0.0.0.0"):
    server = ls.LitServer(InferenceAPI())
    server.run(
        host=host,
        port=port,
        generate_client_file=False,
        pretty_logs=True,
    )