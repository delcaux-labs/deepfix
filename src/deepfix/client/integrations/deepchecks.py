"""
Deepchecks integration for automated model validation.

This module provides comprehensive Deepchecks integration including:
- Pre-configured computer vision test suites
- Custom overfitting detection checks
- Batch processing workflows
- Result parsing and analysis
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import json
from pathlib import Path
import traceback
import base64
from deepchecks.vision.suites import (
    train_test_validation,
    data_integrity,
    model_evaluation,
)
from deepchecks.vision import VisionData
from deepchecks.tabular.suites import (
    train_test_validation as tabular_train_test_validation,
    data_integrity as tabular_data_integrity,
    model_evaluation as tabular_model_evaluation,
)
from deepchecks.tabular import Dataset as TabularDataset
from deepchecks.core import SuiteResult, CheckResult, CheckFailure
from deepchecks.nlp.suites import (
        train_test_validation as nlp_train_test_validation,
        model_evaluation as nlp_model_evaluation,
    )
from deepchecks.nlp import TextData

from ..utils.logging import get_logger
from ...shared.models import (
    DeepchecksParsedResult,
    DeepchecksArtifacts,
    DeepchecksResultHeaders,
    DeepchecksConfig
)

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


class BaseDeepchecksRunner(ABC):

    def __init__(self, config: Optional[DeepchecksConfig] = None):
        self.config = config or DeepchecksConfig()
        self.parser = CheckResultsParser()

    def _save_artifact(self, artifact: DeepchecksArtifacts, dataset_name: str) -> None:
        try:
            output_dir = Path(self.config.output_dir or "results")
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = output_dir / f"{dataset_name}.json"
            with open(artifact_path, "w", encoding="utf-8") as f:
                json.dump(artifact.to_dict(), f, indent=3)
            LOGGER.info("Artifact saved to %s", artifact_path)
        except (IOError, OSError):
            LOGGER.error(
                "Failed to save results to %s. %s", output_dir, traceback.format_exc()
            )

    @abstractmethod
    def run_suites(self, train_data: Union[VisionData, TabularDataset, 'TextData'], dataset_name: str, test_data: Optional[Union[VisionData, TabularDataset, 'TextData']] = None) -> DeepchecksArtifacts:
        pass

    @abstractmethod
    def run_suite_train_test_validation(self, train_data: Union[VisionData, TabularDataset, 'TextData'], test_data: Optional[Union[VisionData, TabularDataset, 'TextData']] = None) -> SuiteResult:
        pass

    @abstractmethod
    def run_suite_data_integrity(self, train_data: Union[VisionData, TabularDataset, 'TextData'], test_data: Optional[Union[VisionData, TabularDataset, 'TextData']] = None) -> SuiteResult:
        pass

    @abstractmethod
    def run_suite_model_evaluation(self, train_data: Union[VisionData, TabularDataset, 'TextData'], test_data: Optional[Union[VisionData, TabularDataset, 'TextData']] = None) -> SuiteResult:
        pass


class DeepchecksRunnerForVision(BaseDeepchecksRunner):
    """
    Deepchecks integration for automated model validation and testing.
    Provides high-level interface for running Deepchecks suites.
    """

    def __init__(self, config: Optional[DeepchecksConfig] = None):
        """
        Initialize Deepchecks runner with configuration.
        """
        super().__init__(config=config)

        self.suite_train_test_validation = train_test_validation()
        self.suite_data_integrity = data_integrity()
        self.suite_model_evaluation = model_evaluation()
        self.output_dir = Path(self.config.output_dir or "results")

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


class DeepchecksRunnerForTabular(BaseDeepchecksRunner):
    """
    Deepchecks integration for automated model validation and testing on tabular data.
    Provides high-level interface for running Deepchecks tabular test suites.
    """

    def __init__(self, config: Optional[DeepchecksConfig] = None):
        """
        Initialize Deepchecks runner with configuration for tabular data.
        """
        super().__init__(config=config)

        self.suite_train_test_validation = tabular_train_test_validation()
        self.suite_data_integrity = tabular_data_integrity()
        self.suite_model_evaluation = tabular_model_evaluation()
        self.output_dir = Path(self.config.output_dir or "results")

    def run_suites(
        self,
        train_data: TabularDataset,
        dataset_name: str,
        test_data: Optional[TabularDataset] = None,
    ) -> DeepchecksArtifacts:
        """
        Run all configured Deepchecks suites on tabular data.
        
        Args:
            train_data: Training dataset as Deepchecks TabularDataset
            dataset_name: Name of the dataset for artifact storage
            test_data: Optional test dataset as Deepchecks TabularDataset
            
        Returns:
            DeepchecksArtifacts containing results from all executed suites
        """
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
        self, train_data: TabularDataset, test_data: Optional[TabularDataset] = None
    ) -> SuiteResult:
        """
        Run train-test validation suite on tabular data.
        
        Validates consistency between training and testing datasets, checking for
        feature drift, label distribution differences, and potential data leakage.
        
        Args:
            train_data: Training dataset as Deepchecks TabularDataset
            test_data: Optional test dataset as Deepchecks TabularDataset
            
        Returns:
            SuiteResult containing validation results
        """
        LOGGER.info("Running train-test validation suite for tabular data")
        return self.suite_train_test_validation.run(
            train_dataset=train_data,
            test_dataset=test_data,
        )

    def run_suite_data_integrity(
        self, train_data: TabularDataset, test_data: Optional[TabularDataset] = None
    ) -> SuiteResult:
        """
        Run data integrity suite on tabular data.
        
        Validates the correctness of datasets by checking for issues like duplicates,
        missing values, and inconsistent labels.
        
        Args:
            train_data: Training dataset as Deepchecks TabularDataset
            test_data: Optional test dataset as Deepchecks TabularDataset
            
        Returns:
            SuiteResult containing data integrity check results
        """
        LOGGER.info("Running data integrity suite for tabular data")
        return self.suite_data_integrity.run(
            train_dataset=train_data,
            test_dataset=test_data,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
        )

    def run_suite_model_evaluation(
        self, train_data: TabularDataset, test_data: Optional[TabularDataset] = None
    ) -> SuiteResult:
        """
        Run model evaluation suite on tabular data.
        
        Assesses a trained model's performance, examining metrics, comparing to benchmarks,
        and identifying underperforming segments.
        
        Args:
            train_data: Training dataset as Deepchecks TabularDataset
            test_data: Optional test dataset as Deepchecks TabularDataset
            
        Returns:
            SuiteResult containing model evaluation results
        """
        LOGGER.info("Running model evaluation suite for tabular data")
        return self.suite_model_evaluation.run(
            train_dataset=train_data,
            test_dataset=test_data,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
        )


class DeepchecksRunnerForText(BaseDeepchecksRunner):
    """
    Deepchecks integration for automated model validation and testing on text/NLP data.
    Provides high-level interface for running Deepchecks NLP test suites.
    """

    def __init__(self, config: Optional[DeepchecksConfig] = None):
        """
        Initialize Deepchecks runner with configuration for NLP text data.
        """
        super().__init__(config=config)

        self.suite_train_test_validation = nlp_train_test_validation()
        self.suite_model_evaluation = nlp_model_evaluation()
        self.output_dir = Path(self.config.output_dir or "results")

    def run_suites(
        self,
        train_data: 'TextData',
        dataset_name: str,
        test_data: Optional['TextData'] = None,
    ) -> DeepchecksArtifacts:
        """
        Run all configured Deepchecks NLP suites on text data.
        
        Args:
            train_data: Training dataset as Deepchecks TextData
            dataset_name: Name of the dataset for artifact storage
            test_data: Optional test dataset as Deepchecks TextData
            
        Returns:
            DeepchecksArtifacts containing results from all executed suites
        """
        output = {}
        if self.config.train_test_validation:
            out_train_test_validation = self.run_suite_train_test_validation(
                train_data, test_data=test_data
            )
            output["train_test_validation"] = self.parser.run(out_train_test_validation)

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
        self, train_data: 'TextData', test_data: Optional['TextData'] = None
    ) -> SuiteResult:
        """
        Run train-test validation suite on NLP text data.
        
        Validates consistency between training and testing datasets, checking for
        distribution differences, label inconsistencies, and potential data leakage.
        
        Args:
            train_data: Training dataset as Deepchecks TextData
            test_data: Optional test dataset as Deepchecks TextData
            
        Returns:
            SuiteResult containing validation results
        """
        LOGGER.info("Running train-test validation suite for NLP text data")
        return self.suite_train_test_validation.run(
            train_dataset=train_data,
            test_dataset=test_data,
            random_state=self.config.random_state,
        )

    def run_suite_model_evaluation(
        self, train_data: 'TextData', test_data: Optional['TextData'] = None
    ) -> SuiteResult:
        """
        Run model evaluation suite on NLP text data.
        
        Assesses a trained model's performance on text tasks, examining metrics,
        identifying underperforming segments, and comparing to baseline models.
        
        Args:
            train_data: Training dataset as Deepchecks TextData
            test_data: Optional test dataset as Deepchecks TextData
            
        Returns:
            SuiteResult containing model evaluation results
        """
        LOGGER.info("Running model evaluation suite for NLP text data")
        return self.suite_model_evaluation.run(
            train_dataset=train_data,
            test_dataset=test_data,
            random_state=self.config.random_state,
        )

    def run_suite_data_integrity(
        self, train_data: 'TextData', test_data: Optional['TextData'] = None
    ) -> SuiteResult:
        """
        Not implemented for NLP text data.
        
        The NLP module does not provide a data_integrity suite.
        Use train_test_validation or model_evaluation instead.
        
        Raises:
            NotImplementedError: This suite is not available for NLP data
        """
        raise NotImplementedError("Data integrity suite is not available for NLP text data")