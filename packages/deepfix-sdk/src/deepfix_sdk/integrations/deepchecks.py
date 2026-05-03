"""
Deepchecks integration for automated model validation.

This module provides comprehensive Deepchecks integration including:
- Pre-configured computer vision test suites
- Custom overfitting detection checks
- Batch processing workflows
- Result parsing and analysis
"""

import base64
import json
import re
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

from deepchecks.core import CheckFailure, CheckResult, SuiteResult
from deepchecks.nlp import TextData
from deepchecks.nlp.suites import (
    data_integrity as nlp_data_integrity,
)
from deepchecks.nlp.suites import (
    model_evaluation as nlp_model_evaluation,
)
from deepchecks.nlp.suites import (
    train_test_validation as nlp_train_test_validation,
)
from deepchecks.tabular.suites import (
    data_integrity as tabular_data_integrity,
)
from deepchecks.tabular.suites import (
    model_evaluation as tabular_model_evaluation,
)
from deepchecks.tabular.suites import (
    train_test_validation as tabular_train_test_validation,
)
from deepchecks.vision import VisionData
from deepchecks.vision.suites import (
    data_integrity as vision_data_integrity,
)
from deepchecks.vision.suites import (
    model_evaluation as vision_model_evaluation,
)
from deepchecks.vision.suites import (
    train_test_validation as vision_train_test_validation,
)
from deepfix_core.models import (
    DataType,
    DeepchecksArtifacts,
    DeepchecksCheckResult,
    DeepchecksConditionResult,
    DeepchecksParsedResult,
    DeepchecksResultHeaders,
)
from sklearn.base import BaseEstimator

from ..config import DeepchecksConfig, DefaultPaths
from ..data.datasets import TabularDataset, InformationRetrievalDataset, NLPDataset
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class Extractor:
    def extract_urls_regex(self, text: str) -> list[str]:
        """
        Extract URLs using regex pattern matching.
        """
        # Comprehensive URL regex pattern
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        # Find all matches
        urls = re.findall(url_pattern, text)
        return urls

    # Remove HTML anchor tags with empty href attributes
    def remove_anchor_tags(self, text: str) -> str:
        """
        Remove any HTML anchor tags (all <a> tags).
        Handles both regular and self-closing anchor tags.
        """
        # Pattern to match any anchor tag with content
        pattern = r"<a\b[^>]*>.*?</a>"
        # Remove all anchor tags
        cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        return cleaned_text

    def extract_values(
        self, json_result: Dict[str, Any], images: Optional[List[str]] = None
    ) -> DeepchecksCheckResult:
        check = json_result.get("check", {})
        check_name = check.get("name", None)
        params = check.get("params", None)
        summary = check.get("summary", None)
        value = json_result.get("value", None)
        conditions_results = []

        for cr in json_result.get("conditions_results", []):
            status = cr.get("Status", {})
            condition = cr.get("Condition", {})
            more_info = cr.get("More Info", {})
            conditions_results.append(
                DeepchecksConditionResult(
                    status=status, condition=condition, more_info=more_info
                )
            )

        links = self.extract_urls_regex(summary)
        if len(links) > 0:
            link_in_summary = " ".join(links)
            summary = self.remove_anchor_tags(summary).strip()
        else:
            link_in_summary = None

        return DeepchecksCheckResult(
            check=check_name,
            params=params,
            summary=summary,
            value=value,
            conditions_results=conditions_results,
            link_in_summary=link_in_summary,
            display_text=json_result.get("txt", None),
            display_images=images,
        )


class CheckResultsParser:
    def __init__(self):
        self.extractor = Extractor()

    def run(self, results: SuiteResult) -> List[DeepchecksParsedResult]:
        parsed_txts = self.parse_txt(results)
        parsed_displays = self.parse_display(results)
        headers = list(parsed_txts.keys()) + list(parsed_displays.keys())
        parsed_results = []
        for header in set(headers):
            display_images = parsed_displays.get(header, {}).get("images", None)
            if display_images is not None:
                display_images = [
                    base64.b64encode(i).decode("utf-8") for i in display_images
                ]

            r = self.extractor.extract_values(
                json_result=parsed_txts[header], images=display_images
            )
            parsed_results.append(DeepchecksParsedResult(header=header, result=r))

        return parsed_results

    def parse_txt(self, results: SuiteResult) -> Dict[str, Dict[str, Any]]:
        parsed_results = {}
        for result in results.results:
            header = result.get_metadata().get("header")
            if header == DeepchecksResultHeaders.HeatmapComparison.value:
                json_result = json.loads(result.to_json(with_display=False))
                json_result.get("value", {"diff": None}).pop("diff")
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

    def _parse_display(self, result: CheckResult) -> Tuple[Optional[List[bytes]], str]:
        images = None  # self._load_display_as_image(result)
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


SupportedDatasetType = Union[
    VisionData, TabularDataset, NLPDataset, InformationRetrievalDataset
]
TClassPred = Union[Sequence[int], Sequence[str], Sequence[Sequence[int]]]
TTextProba = Sequence[Sequence[float]]


class BaseDeepchecksRunner(ABC):
    def __init__(
        self,
        suite_train_test_validation: SuiteResult,
        suite_data_integrity: SuiteResult,
        suite_model_evaluation: SuiteResult,
        config: Optional[DeepchecksConfig] = None,
    ):
        self.config = config or DeepchecksConfig()
        self.parser = CheckResultsParser()
        self.suite_train_test_validation = suite_train_test_validation
        self.suite_data_integrity = suite_data_integrity
        self.suite_model_evaluation = suite_model_evaluation
        self.output_dir = Path(self.config.output_dir or "results")

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

    def run_suites(
        self,
        train_data: SupportedDatasetType,
        dataset_name: str,
        test_data: Optional[SupportedDatasetType] = None,
        model: Optional[BaseEstimator] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> DeepchecksArtifacts:
        output = {}
        if self.config.train_test_validation:
            out_train_test_validation = self.run_suite_train_test_validation(
                train_data, test_data=test_data, **kwargs
            )
            output["train_test_validation"] = self.parser.run(out_train_test_validation)

        if self.config.data_integrity:
            out_data_integrity = self.run_suite_data_integrity(
                train_data, test_data=test_data, **kwargs
            )
            output["data_integrity"] = self.parser.run(out_data_integrity)

        if self.config.model_evaluation:
            out_model_evaluation = self.run_suite_model_evaluation(
                model=model, train_data=train_data, test_data=test_data, **kwargs
            )
            output["model_evaluation"] = self.parser.run(out_model_evaluation)

        artifact = DeepchecksArtifacts(
            dataset_name=dataset_name,
            model_name=model_name,
            results=output,
            config=self.config,
        )

        if self.config.save_results:
            self._save_artifact(artifact=artifact, dataset_name=dataset_name)

        return artifact

    @abstractmethod
    def run_suite_train_test_validation(
        self,
        train_data: SupportedDatasetType,
        test_data: Optional[SupportedDatasetType] = None,
        **kwargs,
    ) -> SuiteResult:
        pass

    @abstractmethod
    def run_suite_data_integrity(
        self,
        train_data: SupportedDatasetType,
        test_data: Optional[SupportedDatasetType] = None,
        **kwargs,
    ) -> SuiteResult:
        pass

    @abstractmethod
    def run_suite_model_evaluation(
        self,
        model: Any,
        train_data: SupportedDatasetType,
        test_data: Optional[SupportedDatasetType] = None,
        **kwargs,
    ) -> SuiteResult:
        pass


def get_deepchecks_runner(
    data_type: Union[str, DataType],
    train_test_validation: bool = True,
    data_integrity: bool = True,
    model_evaluation: bool = False,
    config: Optional[DeepchecksConfig] = None,
) -> BaseDeepchecksRunner:
    if config is None:
        config = DeepchecksConfig(
            train_test_validation=train_test_validation,
            data_integrity=data_integrity,
            model_evaluation=model_evaluation,
        )
    _type = DataType(data_type) if isinstance(data_type, str) else data_type
    if _type == DataType.VISION:
        return DeepchecksRunnerForVision(config=config)
    elif _type == DataType.TABULAR:
        return DeepchecksRunnerForTabular(config=config)
    elif _type == DataType.NLP:
        return DeepchecksRunnerForNLP(config=config)
    elif _type == DataType.IR:
        return DeepchecksRunnerForIR(config=config)
    else:
        raise ValueError(f"Unsupported data type: {_type}")


class DeepchecksRunnerForIR(BaseDeepchecksRunner):
    """
    Unified Deepchecks runner for Information Retrieval tasks.
    Orchestrates both NLP and Tabular validation suites.
    """

    def __init__(self, config: Optional[DeepchecksConfig] = None):
        """
        Initialize the IR runner with both NLP and Tabular sub-runners.
        """
        super().__init__(
            config=config,
            suite_train_test_validation=None,
            suite_data_integrity=None,
            suite_model_evaluation=None,
        )
        self.nlp_runner = DeepchecksRunnerForNLP(config=config)
        self.tabular_runner = DeepchecksRunnerForTabular(config=config)

    def run_suites(
        self,
        train_data: InformationRetrievalDataset,
        dataset_name: str,
        test_data: Optional[InformationRetrievalDataset] = None,
        model: Optional[BaseEstimator] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> DeepchecksArtifacts:
        """
        Run both NLP and Tabular Deepchecks suites for IR data.
        """
        assert isinstance(train_data, InformationRetrievalDataset), (
            f"Expected InformationRetrievalDataset but got {type(train_data).__name__}"
        )
        assert isinstance(test_data, InformationRetrievalDataset), (
            f"Expected InformationRetrievalDataset but got {type(test_data).__name__}"
        )
        LOGGER.info("Running unified IR validation for %s", dataset_name)

        # 1. Run NLP suites (wrap as NLPDataset since IR no longer inherits from it)
        nlp_artifact = self.nlp_runner.run_suites(
            train_data=train_data.to_nlp_dataset(),
            dataset_name=f"{dataset_name}_nlp",
            test_data=test_data.to_nlp_dataset() if test_data else None,
            model=model,
            model_name=model_name,
            **kwargs,
        )

        # 2. Run Tabular suites
        tabular_artifact = self.tabular_runner.run_suites(
            train_data=train_data.to_tabular(),
            dataset_name=f"{dataset_name}_tabular",
            test_data=test_data.to_tabular() if test_data else None,
            model=model,
            model_name=model_name,
            **kwargs,
        )

        # 3. Combine results into a single artifact
        combined_results = {}
        for category, results in nlp_artifact.results.items():
            combined_results[f"nlp_{category}"] = results
        for category, results in tabular_artifact.results.items():
            combined_results[f"tabular_{category}"] = results

        artifact = DeepchecksArtifacts(
            dataset_name=dataset_name,
            model_name=model_name,
            results=combined_results,
            config=self.config,
        )

        if self.config.save_results:
            self._save_artifact(artifact=artifact, dataset_name=dataset_name)

        return artifact

    def run_suite_train_test_validation(
        self,
        train_data: SupportedDatasetType,
        test_data: Optional[SupportedDatasetType] = None,
        **kwargs,
    ) -> SuiteResult:
        raise NotImplementedError(
            "Direct suite execution is not supported for IR runner"
        )

    def run_suite_data_integrity(
        self,
        train_data: SupportedDatasetType,
        test_data: Optional[SupportedDatasetType] = None,
        **kwargs,
    ) -> SuiteResult:
        raise NotImplementedError(
            "Direct suite execution is not supported for IR runner"
        )

    def run_suite_model_evaluation(
        self,
        model: Any,
        train_data: SupportedDatasetType,
        test_data: Optional[SupportedDatasetType] = None,
        **kwargs,
    ) -> SuiteResult:
        raise NotImplementedError(
            "Direct suite execution is not supported for IR runner"
        )


class DeepchecksRunnerForVision(BaseDeepchecksRunner):
    """
    Deepchecks integration for automated model validation and testing.
    Provides high-level interface for running Deepchecks suites.
    """

    def __init__(self, config: Optional[DeepchecksConfig] = None):
        """
        Initialize Deepchecks runner with configuration.
        """
        super().__init__(
            config=config,
            suite_train_test_validation=vision_train_test_validation(),
            suite_data_integrity=vision_data_integrity(),
            suite_model_evaluation=vision_model_evaluation(),
        )

    def run_suite_train_test_validation(
        self,
        train_data: SupportedDatasetType,
        test_data: Optional[SupportedDatasetType] = None,
        **kwargs,
    ) -> SuiteResult:
        LOGGER.info("Running train-test validation suite")
        self._check_inputs(train_data, test_data)
        return self.suite_train_test_validation.run(
            train_dataset=train_data,
            test_dataset=test_data,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
        )

    def run_suite_data_integrity(
        self,
        train_data: SupportedDatasetType,
        test_data: Optional[SupportedDatasetType] = None,
        **kwargs,
    ) -> SuiteResult:
        LOGGER.info("Running data integrity suite")
        self._check_inputs(train_data, test_data)
        return self.suite_data_integrity.run(
            train_dataset=train_data,
            test_dataset=test_data,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
        )

    def run_suite_model_evaluation(
        self,
        model: Any,
        train_data: SupportedDatasetType,
        test_data: Optional[SupportedDatasetType] = None,
        **kwargs,
    ) -> SuiteResult:
        raise NotImplementedError("TODO: debug this method")
        self._check_inputs(train_data, test_data)
        LOGGER.info("Running model evaluation suite")
        return self.suite_model_evaluation.run(
            train_dataset=train_data,
            model=model,
            test_dataset=test_data,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
        )

    def _check_inputs(
        self, train_data: VisionData, test_data: Optional[VisionData] = None
    ) -> None:
        assert isinstance(train_data, VisionData), (
            f"train_data must be an instance of VisionData, got {type(train_data)}"
        )
        if test_data is not None:
            assert isinstance(test_data, VisionData), (
                f"test_data must be an instance of VisionData, got {type(test_data)}"
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
        super().__init__(
            config=config,
            suite_train_test_validation=tabular_train_test_validation(),
            suite_data_integrity=tabular_data_integrity(),
            suite_model_evaluation=tabular_model_evaluation(),
        )

    def run_suite_train_test_validation(
        self,
        train_data: SupportedDatasetType,
        test_data: Optional[SupportedDatasetType] = None,
        **kwargs,
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
        self._check_inputs(train_data, test_data)

        return self.suite_train_test_validation.run(
            train_dataset=train_data.dataset,
            test_dataset=test_data.dataset if test_data is not None else None,
        )

    def run_suite_data_integrity(
        self,
        train_data: SupportedDatasetType,
        test_data: Optional[SupportedDatasetType] = None,
        **kwargs,
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
        self._check_inputs(train_data, test_data)
        return self.suite_data_integrity.run(
            train_dataset=train_data.dataset,
            test_dataset=test_data.dataset if test_data is not None else None,
        )

    def run_suite_model_evaluation(
        self,
        model: BaseEstimator,
        train_data: SupportedDatasetType,
        test_data: Optional[SupportedDatasetType] = None,
        **kwargs,
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
        self._check_inputs(train_data, test_data)
        return self.suite_model_evaluation.run(
            train_dataset=train_data.dataset,
            test_dataset=test_data.dataset if test_data is not None else None,
            model=model,
        )

    def _check_inputs(
        self, train_data: TabularDataset, test_data: Optional[TabularDataset] = None
    ) -> None:
        assert isinstance(train_data, TabularDataset), (
            f"train_data must be an instance of TabularDataset, got {type(train_data)}"
        )
        if test_data is not None:
            assert isinstance(test_data, TabularDataset), (
                f"test_data must be an instance of TabularDataset, got {type(test_data)}"
            )


class DeepchecksRunnerForNLP(BaseDeepchecksRunner):
    """
    Deepchecks integration for automated model validation and testing on text/NLP data.
    Provides high-level interface for running Deepchecks NLP test suites.
    """

    def __init__(self, config: Optional[DeepchecksConfig] = None):
        """
        Initialize Deepchecks runner with configuration for NLP text data.
        """
        super().__init__(
            config=config,
            suite_train_test_validation=nlp_train_test_validation(),
            suite_data_integrity=nlp_data_integrity(),
            suite_model_evaluation=nlp_model_evaluation(),
        )

    def run_suite_train_test_validation(
        self,
        train_data: NLPDataset,
        test_data: Optional[NLPDataset] = None,
        **kwargs,
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
            train_dataset=train_data.dataset,
            test_dataset=test_data.dataset if test_data is not None else None,
            random_state=self.config.random_state,
            **kwargs,
        )

    def run_suite_model_evaluation(
        self,
        train_data: NLPDataset,
        test_data: Optional[NLPDataset] = None,
        model: Optional[Any] = None,
        model_classes: Optional[List[str]] = None,
        train_predictions: Optional[TClassPred] = None,
        train_probabilities: Optional[TTextProba] = None,
        test_predictions: Optional[TClassPred] = None,
        test_probabilities: Optional[TTextProba] = None,
        **kwargs,
    ) -> SuiteResult:
        """
        Run model evaluation suite on NLP text data.
        """
        LOGGER.info("Running model evaluation suite for NLP text data")

        # Deepchecks NLP suite.run accepts these arguments
        return self.suite_model_evaluation.run(
            train_dataset=train_data.dataset,
            test_dataset=test_data.dataset if test_data is not None else None,
            train_predictions=train_predictions,
            test_predictions=test_predictions,
            train_probabilities=train_probabilities,
            test_probabilities=test_probabilities,
            model_classes=model_classes,
            random_state=self.config.random_state,
            **kwargs,
        )

    def run_suite_data_integrity(
        self,
        train_data: NLPDataset,
        test_data: Optional[NLPDataset] = None,
        **kwargs,
    ) -> SuiteResult:
        """
        Not implemented for NLP text data.

        The NLP module does not provide a data_integrity suite.
        Use train_test_validation or model_evaluation instead.

        Raises:
            NotImplementedError: This suite is not available for NLP data
        """
        return self.suite_data_integrity.run(
            train_dataset=train_data.dataset,
            test_dataset=test_data.dataset if test_data is not None else None,
            random_state=self.config.random_state,
            **kwargs,
        )
