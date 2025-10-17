"""
Deepchecks integration for automated model validation.

This module provides comprehensive Deepchecks integration including:
- Pre-configured computer vision test suites
- Custom overfitting detection checks
- Batch processing workflows
- Result parsing and analysis
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from deepchecks.vision.suites import (
    train_test_validation,
    data_integrity,
    model_evaluation,
)
from deepchecks.vision import VisionData
from deepchecks.core import SuiteResult, CheckResult, CheckFailure
import json
from pathlib import Path
import traceback
import base64

from ..utils.logging import get_logger
from ..core.artifacts.datamodel import (
    DeepchecksParsedResult,
    DeepchecksArtifacts,
    DeepchecksResultHeaders,
)
from ..core.config import DeepchecksConfig

LOGGER = get_logger(__name__)


class CheckResultsParser:
    def run(self, results: SuiteResult) -> List[DeepchecksParsedResult]:
        parsed_txts = self.parse_txt(results)
        parsed_displays = self.parse_display(results)
        parsed_results = []
        keys = list(parsed_txts.keys())
        keys.extend(parsed_displays.keys())
        keys = list(set(keys))
        for header in keys:
            if header in parsed_displays.keys():
                display_images = [
                    base64.b64encode(i).decode("utf-8")
                    for i in parsed_displays[header]["images"]
                ]
                display_txt = parsed_displays[header]["txt"]
            else:
                display_images = None
                display_txt = None
            r = DeepchecksParsedResult(
                header=header,
                display_images=display_images,
                display_txt=display_txt,
                json_result=parsed_txts[header],
            )
            parsed_results.append(r)
        return parsed_results

    def parse_txt(self, results: SuiteResult) -> Dict[str, Dict[str, Any]]:
        parsed_results = {}
        for result in results.results:
            header = result.get_metadata().get("header")
            if header == DeepchecksResultHeaders.HeatmapComparison.value:
                json_result = json.loads(result.to_json(with_display=False))
                json_result["value"].pop("diff")
                parsed_results[header] = json_result
                continue
            parsed_results[header] = json.loads(result.to_json(with_display=False))
        return parsed_results

    def parse_display(
        self, results: SuiteResult
    ) -> Dict[str, Dict[str, Union[List[bytes], str]]]:
        display_result = {}
        for result in results.results:
            if isinstance(result, CheckFailure):
                continue
            if not result.have_display():
                continue

            header = result.get_metadata().get("header")
            images, txt = self._parse_display(result)

            if header in [
                DeepchecksResultHeaders.ImagePropertyOutliers.value,
                DeepchecksResultHeaders.NewLabels.value,
            ]:
                txt = None

            display_result[header] = {"images": images, "txt": txt}
        return display_result

    def _parse_display(self, result: CheckResult) -> Tuple[List[bytes], str]:
        images = self._load_display_as_image(result)
        txt = self._parse_display_txt(result)
        return images, txt

    def _load_display_as_image(self, result: CheckResult) -> List[bytes]:
        images = []
        for d in result.display:
            if hasattr(d, "to_image"):
                images.append(d.to_image())
        return images

    def _parse_display_txt(self, result: CheckResult) -> List[str]:
        txts = [
            d.replace("<span>", "").replace("</span>", "")
            for d in result.display
            if isinstance(d, str)
        ]
        txts = " ".join(txts)
        return txts


class DeepchecksRunner:
    """
    Deepchecks integration for automated model validation and testing.
    Provides high-level interface for running Deepchecks suites.
    """

    def __init__(self, config: Optional[DeepchecksConfig] = None):
        """
        Initialize Deepchecks runner with configuration.
        """
        self.config = config or DeepchecksConfig()
        self.parser = CheckResultsParser()

        self.suite_train_test_validation = train_test_validation()
        self.suite_data_integrity = data_integrity()
        self.suite_model_evaluation = model_evaluation()
        self.output_dir = Path(self.config.output_dir or "results")

    def _save_artifact(self, artifact: DeepchecksArtifacts, dataset_name: str) -> None:
        try:
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = self.output_dir / f"{dataset_name}.json"
            with open(artifact_path, "w") as f:
                json.dump(artifact.to_dict(), f, indent=3)
            LOGGER.info(f"Artifact saved to {artifact_path}")
        except Exception:
            LOGGER.error(
                f"Failed to save results to {self.output_dir}. {traceback.format_exc()}"
            )

    def run_suites(
        self,
        train_data: VisionData,
        dataset_name: str,
        test_data: Optional[VisionData] = None,
    ) -> DeepchecksArtifacts:
        output = {}
        if self.config.train_test_validation:
            out_train_test_validation = self.run_suite_train_test_validation(
                train_data, test_data=test_data
            )
            output["train_test_validation"] = self.parser.run(out_train_test_validation)

        if self.config.data_integrity:
            out_data_integrity = self.run_suite_data_integrity(
                train_data, test_data=test_data
            )
            output["data_integrity"] = self.parser.run(out_data_integrity)

        if self.config.model_evaluation:
            out_model_evaluation = self.run_suite_model_evaluation(
                train_data, test_data=test_data
            )
            output["model_evaluation"] = self.parser.run(out_model_evaluation)

        artifact = DeepchecksArtifacts(
            dataset_name=dataset_name, results=output, config=self.config
        )

        if self.config.save_results:
            self._save_artifact(artifact=artifact, dataset_name=dataset_name)

        return artifact

    def run_suite_train_test_validation(
        self, train_data: VisionData, test_data: Optional[VisionData] = None
    ) -> SuiteResult:
        LOGGER.info("Running train-test validation suite")
        return self.suite_train_test_validation.run(
            train_dataset=train_data,
            test_dataset=test_data,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
        )

    def run_suite_data_integrity(
        self, train_data: VisionData, test_data: Optional[VisionData] = None
    ) -> SuiteResult:
        LOGGER.info("Running data integrity suite")
        return self.suite_data_integrity.run(
            train_dataset=train_data,
            test_dataset=test_data,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
        )

    def run_suite_model_evaluation(
        self, train_data: VisionData, test_data: Optional[VisionData] = None
    ) -> SuiteResult:
        LOGGER.info("Running model evaluation suite")
        return self.suite_model_evaluation.run(
            train_dataset=train_data,
            test_dataset=test_data,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
        )
