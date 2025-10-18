from deepchecks.vision import VisionData
from typing import Optional

from .base import Step
from ..integrations import DeepchecksRunner
from ...shared.models import DeepchecksConfig


class Checks(Step):
    def __init__(
        self, deepchecks_config: DeepchecksConfig, dataset_name: Optional[str] = None
    ):
        self.deepchecks_config = deepchecks_config
        self.dataset_name = dataset_name

    def run(
        self,
        context: dict,
        train_data: Optional[VisionData] = None,
        test_data: Optional[VisionData] = None,
        **kwargs,
    ) -> dict:
        deepchecks_config = self.deepchecks_config or context.get("deepchecks_config")
        deepchecks_runner = DeepchecksRunner(config=deepchecks_config)
        dataset_name = self.dataset_name or context.get("dataset_name")

        assert dataset_name is not None, (
            "dataset_name must be provided in context or as an argument"
        )

        artifacts = deepchecks_runner.run_suites(
            train_data=train_data or context.get("train_data"),
            test_data=test_data or context.get("test_data"),
            dataset_name=dataset_name,
        )
        context["checks_artifacts"] = artifacts
        return context
    
    def get_name(self) -> str:
        return "ml_tests"