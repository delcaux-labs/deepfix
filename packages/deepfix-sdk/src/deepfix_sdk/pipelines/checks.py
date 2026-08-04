from typing import Any, Optional

from ..config import DeepchecksConfig
from ..integrations.deepchecks import get_deepchecks_runner
from .base import Step


class Checks(Step):
    def __init__(
        self,
        deepchecks_config: DeepchecksConfig,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.deepchecks_config = deepchecks_config
        self.dataset_name = dataset_name
        self.model_name = model_name

    def run(
        self,
        context: dict,
        train_data: Any = None,
        test_data: Any = None,
        model: Optional[Any] = None,
        **kwargs,
    ) -> dict:
        deepchecks_config = self.deepchecks_config or context.get("deepchecks_config")
        deepchecks_runner = get_deepchecks_runner(
            data_type=deepchecks_config.data_type, config=deepchecks_config
        )
        dataset_name = self.dataset_name or context.get("dataset_name")
        model_name = self.model_name or context.get("model_name")
        assert dataset_name is not None, (
            "dataset_name must be provided in context or as an argument"
        )

        artifacts = deepchecks_runner.run_suites(
            train_data=train_data or context.get("train_data"),
            test_data=test_data or context.get("test_data"),
            model=model or context.get("model"),
            model_name=model_name,
            dataset_name=dataset_name,
            **kwargs,
        )
        context["checks_artifacts"] = artifacts
        return context

    def get_name(self) -> str:
        return "ml_tests"
