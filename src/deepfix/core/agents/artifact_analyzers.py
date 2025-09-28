from typing import List, Dict, Any, Type
import numpy as np
import os
import dspy

from .base import Agent, ArtifactAnalyzer
from ..artifacts import TrainingArtifacts, Artifacts, DeepchecksArtifacts, DatasetArtifacts, ModelCheckpointArtifacts
from .models import AgentContext, AgentResult, ArtifactAnalysisConfig


from ..query.intelligence import IntelligenceClient
from .signatures import (TrainingArtifactsAnalysisSignature, 
                        DeepchecksArtifactsAnalysisSignature, 
                        DatasetArtifactsAnalysisSignature, 
                        ModelCheckpointArtifactsAnalysisSignature, 
                        ArtifactsAnalysisSignature)


class TrainingArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in ML training diagnostics, hyperparameter analysis, convergence patterns"""

    def __init__(self,):
        super().__init__()
        self.llm = dspy.ChainOfThought(TrainingArtifactsAnalysisSignature)

    def _run(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert ML training diagnostics specialist with deep expertise in:
                    - Training metrics analysis and anomaly detection
                    - Hyperparameter optimization and configuration validation
                    - Learning dynamics patterns and convergence analysis
                    - Training stability assessment and debugging

                    Your role is to analyze training artifacts (metrics, parameters) and provide actionable insights about:
                    1. Training quality and convergence patterns
                    2. Potential issues like overfitting, underfitting, or instability
                    3. Hyperparameter optimization opportunities
                    4. Configuration best practices and recommendations

                    Analysis Focus Areas:
                    - **Metrics Validation**: Completeness, consistency, anomaly detection
                    - **Learning Dynamics**: Convergence patterns, stability, plateaus
                    - **Parameter Assessment**: Hyperparameter quality, best practices
                    - **Performance Indicators**: Training efficiency, optimization potential

                    When analyzing training metadata, consider:
                    - Loss convergence trends and stability
                    - Training vs validation metric divergence
                    - Learning rate schedules and optimizer effectiveness
                    - Batch size impact on training dynamics
                    - Model architecture appropriateness
                    - Early stopping and regularization effectiveness

                    Provide specific, actionable recommendations with clear rationale and expected impact.
            """
    
    @property
    def supported_artifact_types(self) -> List[Type[Artifacts]]:
        return [TrainingArtifacts]
    
    def _serialize_artifact(self, artifact: TrainingArtifacts) -> Dict[str, Any]:
        """Convert training artifacts to LLM-friendly format."""        
        
        serialized = {
            "artifact_type": type(artifact).__name__,
            "metrics_path": artifact.metrics_path,
            "has_metrics_data": artifact.metrics_values is not None,
            "has_parameters": artifact.params is not None
        }
        
        # Include metrics summary statistics
        if artifact.metrics_values is not None:
            df = artifact.metrics_values
            serialized["metrics_summary"] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
                "missing_values": df.isnull().sum().to_dict(),
                "basic_stats": df.describe().to_dict() if len(df) > 0 else {}
            }
            
            # Include sample of recent metrics for trend analysis
            if len(df) > 0:
                serialized["recent_metrics_sample"] = df.tail(min(10, len(df))).to_dict('records')
        
        # Include parameters
        if artifact.params is not None:
            serialized["parameters"] = artifact.params
            
        return serialized
    
    def _format_artifacts_for_prompt(self, artifacts: List[Dict[str, Any]]) -> str:
        """Format training artifacts for LLM prompt."""
        formatted_sections = []
        
        for i, artifact in enumerate(artifacts):
            section = f"### Training Artifact {i+1}\n"
            
            # Basic info
            section += f"- Type: {artifact['artifact_type']}\n"
            section += f"- Metrics File: {artifact.get('metrics_path', 'None')}\n"
            section += f"- Has Metrics Data: {artifact['has_metrics_data']}\n"
            section += f"- Has Parameters: {artifact['has_parameters']}\n\n"
        
        # Metrics analysis
            if artifact['has_metrics_data'] and 'metrics_summary' in artifact:
                summary = artifact['metrics_summary']
                section += "**Metrics Summary:**\n"
                section += f"- Shape: {summary['shape']}\n"
                section += f"- Columns: {summary['columns']}\n"
                section += f"- Missing Values: {summary['missing_values']}\n"
                
                if summary['basic_stats']:
                    section += "- Basic Statistics Available: Yes\n"
                    # Include key statistics for loss columns
                    for col, stats in summary['basic_stats'].items():
                        if 'loss' in col.lower():
                            section += f"  - {col}: mean={stats.get('mean', 'N/A'):.4f}, std={stats.get('std', 'N/A'):.4f}\n"
                
                # Recent metrics trends
                if 'recent_metrics_sample' in artifact:
                    section += f"\n**Recent Metrics Sample (last {len(artifact['recent_metrics_sample'])} entries):**\n"
                    for record in artifact['recent_metrics_sample'][-3:]:  # Show last 3 for brevity
                        section += f"- {record}\n"
            
            # Parameters
            if artifact['has_parameters'] and 'parameters' in artifact:
                section += "\n**Training Parameters:**\n"
                params = artifact['parameters']
                
                # Highlight key parameters
                key_params = ['learning_rate', 'batch_size', 'epochs', 'optimizer', 'model_name', 'num_classes']
                for param in key_params:
                    if param in params:
                        section += f"- {param}: {params[param]}\n"
                
                # Show all other parameters
                other_params = {k: v for k, v in params.items() if k not in key_params}
                if other_params:
                    section += f"- Other parameters: {other_params}\n"
            
            formatted_sections.append(section)
        
        return "\n".join(formatted_sections)


class DeepchecksArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in data quality and validation, data drift detection, data integrity assessment, and outlier identification"""

    def __init__(self,):
        super().__init__()
        self.llm = dspy.ChainOfThought(DeepchecksArtifactsAnalysisSignature)

    def _run(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError

    @property
    def system_prompt(self) -> str:
        return """You are an expert data quality and validation specialist with deep expertise in:
                - Data drift detection and distribution analysis
                - Data integrity assessment and outlier identification  
                - Train-test validation and data leakage detection
                - Computer vision data quality patterns and issues

                Your role is to analyze Deepchecks validation results and provide actionable insights about:
                1. Data quality degradation and drift patterns
                2. Training data integrity and consistency issues
                3. Potential data leakage or bias problems
                4. Feature-target relationship changes and anomalies

                Analysis Focus Areas:
                - **Data Drift Analysis**: Distribution shifts, feature drift, label drift
                - **Integrity Assessment**: Outliers, inconsistent labeling, correlation changes
                - **Validation Quality**: Train-test splits, data leakage indicators
                - **Performance Impact**: How data issues affect model performance

                Deepchecks Result Categories to Consider:
                **Train-Test Validation:**
                - Label Drift: Changes in label distribution between train/test
                - Image Dataset Drift: Overall dataset distribution changes
                - Image Property Drift: Feature distribution shifts
                - Property Label Correlation Change: Feature-target relationship changes
                - Heatmap Comparison: Visual similarity analysis results
                - New Labels: Unseen classes in test data

                **Data Integrity:**
                - Image Property Outliers: Anomalous data points
                - Property Label Correlation: Feature-target relationships
                - Label Property Outliers: Inconsistent label assignments  
                - Class Performance: Per-class performance variations

                When analyzing Deepchecks results, focus on:
                - Severity and impact of detected issues
                - Patterns across multiple validation checks
                - Root cause analysis of quality problems
                - Prioritized recommendations for data improvement
                - Risk assessment for model deployment

                Provide specific, actionable recommendations with clear impact assessment.
        """
    
    @property
    def supported_artifact_types(self) -> List[Type[Artifacts]]:
        return [DeepchecksArtifacts]
    
    def _serialize_artifact(self, artifact: DeepchecksArtifacts) -> Dict[str, Any]:
        """Convert Deepchecks artifacts to LLM-friendly format."""
        serialized = {
            "artifact_type": type(artifact).__name__,
            "dataset_name": artifact.dataset_name,
            "has_config": artifact.config is not None,
            "result_categories": list(artifact.results.keys()),
            "total_results": sum(len(results) for results in artifact.results.values())
        }
        
        # Organize results by category with summaries
        serialized["results_by_category"] = {}
        for category, results in artifact.results.items():
            category_summary = {
                "count": len(results),
                "checks": []
            }
        
        for result in results:
            check_summary = {
                "header": result.header,
                "has_json_result": bool(result.json_result),
                "has_display_text": result.display_txt is not None,
                "has_images": result.display_images is not None and len(result.display_images) > 0
            }
            
            # Extract key insights from JSON result if available
            if result.json_result:
                # Common patterns in Deepchecks results
                if "value" in result.json_result:
                    check_summary["result_value"] = result.json_result["value"]
                if "conditions_results" in result.json_result:
                    check_summary["condition_results"] = result.json_result["conditions_results"]
                if "display" in result.json_result:
                    # Truncate display data for prompt efficiency
                    display_data = str(result.json_result["display"])
                    check_summary["display_summary"] = display_data[:500] + "..." if len(display_data) > 500 else display_data
            
            # Include concise display text
            if result.display_txt:
                text = result.display_txt
                check_summary["display_text"] = text[:300] + "..." if len(text) > 300 else text
            
            category_summary["checks"].append(check_summary)
            
            serialized["results_by_category"][category] = category_summary
        
        # Include config summary if available
        if artifact.config:
            serialized["config_summary"] = {
                "has_config": True,
                # Add specific config fields that are relevant for analysis
                "config_type": str(type(artifact.config).__name__)
            }
        
        return serialized
    
    def _format_artifacts_for_prompt(self, artifacts: List[Dict[str, Any]]) -> str:
        """Format Deepchecks artifacts for LLM prompt."""
        formatted_sections = []
        
        for i, artifact in enumerate(artifacts):
            section = f"### Deepchecks Artifact {i+1}\n"
            
            # Basic info
            section += f"- Dataset: {artifact['dataset_name']}\n"
            section += f"- Total Results: {artifact['total_results']}\n"
            section += f"- Categories: {artifact['result_categories']}\n"
            section += f"- Has Config: {artifact['has_config']}\n\n"
            
            # Results by category
            section += "**Validation Results by Category:**\n\n"
            
            for category, category_data in artifact["results_by_category"].items():
                section += f"#### {category} ({category_data['count']} checks)\n"
                
                for check in category_data["checks"]:
                    section += f"**{check['header']}:**\n"
                    
                    # Result value
                    if "result_value" in check:
                        section += f"- Result Value: {check['result_value']}\n"
                    
                    # Condition results
                    if "condition_results" in check:
                        section += f"- Condition Results: {check['condition_results']}\n"
                    
                    # Display text (truncated)
                    if "display_text" in check:
                        section += f"- Summary: {check['display_text']}\n"
                    
                    # Display summary (truncated)
                    if "display_summary" in check:
                        section += f"- Display Data: {check['display_summary']}\n"
                    
                    section += f"- Has Images: {check['has_images']}\n\n"
            
            formatted_sections.append(section)
        
        return "\n".join(formatted_sections)


class DatasetArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in dataset analysis and quality assessment, data distribution analysis, feature quality assessment, and class balance evaluation"""

    def __init__(self,):
        super().__init__()
        self.llm = dspy.ChainOfThought(DatasetArtifactsAnalysisSignature)

    def _run(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError

    @property
    def system_prompt(self) -> str:
        return """You are an expert data scientist specializing in dataset analysis and quality assessment with deep expertise in:
                - Dataset statistics interpretation and quality evaluation
                - Data distribution analysis and anomaly detection
                - Feature quality assessment and correlation analysis  
                - Class balance evaluation and sampling strategy recommendations

                Your role is to analyze dataset artifacts and provide actionable insights about:
                1. Dataset completeness and statistical quality
                2. Data distribution patterns and potential biases
                3. Feature quality and representativeness
                4. Adequacy for machine learning model training

                Analysis Focus Areas:
                - **Completeness Assessment**: Missing statistics, incomplete features, data coverage
                - **Distribution Analysis**: Feature distributions, class balance, outlier detection  
                - **Quality Evaluation**: Data integrity, feature diversity, statistical validity
                - **ML Readiness**: Dataset suitability, potential training challenges

                Key Dataset Quality Indicators:
                - **Sample Sufficiency**: Adequate data volume per class/feature
                - **Class Balance**: Distribution across target classes
                - **Feature Quality**: Distribution normality, missing values, outliers
                - **Statistical Validity**: Meaningful statistics, proper data types
                - **Representativeness**: Coverage of problem domain

                When analyzing dataset statistics, consider:
                - Sample size adequacy for reliable model training
                - Class imbalance severity and impact on training
                - Feature distribution characteristics and potential preprocessing needs
                - Missing value patterns and imputation strategies
                - Outlier presence and impact on model performance
                - Feature correlation patterns and redundancy
                - Data type consistency and validation

                Provide specific recommendations for:
                - Data quality improvements
                - Preprocessing strategies
                - Sampling approaches for imbalanced data
                - Feature engineering opportunities
                - Data collection priorities

                Focus on actionable insights that directly impact model training success."""
    
    @property
    def supported_artifact_types(self) -> List[Type[Artifacts]]:
        return [DatasetArtifacts]
    
    def _serialize_artifact(self, artifact: DatasetArtifacts) -> Dict[str, Any]:
        """Convert dataset artifacts to LLM-friendly format."""
        serialized = {
            "artifact_type": "DatasetArtifacts",
            "dataset_name": artifact.dataset_name,
            "has_statistics": artifact.statistics is not None
        }
        
        if artifact.statistics is not None:
            stats = artifact.statistics
            
            # Organize statistics by type for better analysis
            serialized["statistics_summary"] = {
                "available_metrics": list(stats.keys()),
                "data_overview": {},
                "feature_statistics": {},
                "class_information": {},
                "quality_indicators": {}
            }
            
            # Extract common dataset statistics patterns
            summary = serialized["statistics_summary"]
            
            # Data overview
            if "total_samples" in stats:
                summary["data_overview"]["total_samples"] = stats["total_samples"]
            if "num_features" in stats:
                summary["data_overview"]["num_features"] = stats["num_features"]
            if "num_classes" in stats:
                summary["data_overview"]["num_classes"] = stats["num_classes"]
            
            # Feature statistics
            feature_keys = [k for k in stats.keys() if "feature" in k.lower() or "column" in k.lower()]
            for key in feature_keys:
                summary["feature_statistics"][key] = stats[key]
            
            # Class information
            class_keys = [k for k in stats.keys() if "class" in k.lower() or "label" in k.lower() or "target" in k.lower()]
            for key in class_keys:
                summary["class_information"][key] = stats[key]
            
            # Quality indicators
            quality_keys = [k for k in stats.keys() if any(term in k.lower() for term in ["missing", "null", "outlier", "duplicate", "unique"])]
            for key in quality_keys:
                summary["quality_indicators"][key] = stats[key]
            
            # Include raw statistics for comprehensive analysis
            # (truncated to prevent prompt overflow)
            serialized["raw_statistics"] = {
                k: v for k, v in stats.items() 
                if not isinstance(v, (dict, list)) or len(str(v)) < 1000
            }
        
        return serialized
    
    def _format_artifacts_for_prompt(self, artifacts: List[Dict[str, Any]]) -> str:
        """Format dataset artifacts for LLM prompt."""
        formatted_sections = []
        
        for i, artifact in enumerate(artifacts):
            section = f"### Dataset Artifact {i+1}\n"
            
            # Basic info
            section += f"- Dataset Name: {artifact['dataset_name']}\n"
            section += f"- Has Statistics: {artifact['has_statistics']}\n\n"
            
            if artifact['has_statistics'] and 'statistics_summary' in artifact:
                summary = artifact['statistics_summary']
                
                # Data overview
                if summary['data_overview']:
                    section += "**Data Overview:**\n"
                    for key, value in summary['data_overview'].items():
                        section += f"- {key}: {value}\n"
                    section += "\n"
                
                # Feature statistics
                if summary['feature_statistics']:
                    section += "**Feature Statistics:**\n"
                    for key, value in summary['feature_statistics'].items():
                        # Format complex values for readability
                        if isinstance(value, dict):
                            section += f"- {key}: {len(value)} features\n"
                            # Show sample of feature stats
                            sample_keys = list(value.keys())[:3]
                            for sample_key in sample_keys:
                                section += f"  - {sample_key}: {value[sample_key]}\n"
                            if len(value) > 3:
                                section += f"  - ... and {len(value) - 3} more\n"
                        else:
                            section += f"- {key}: {value}\n"
                    section += "\n"
                
                # Class information
                if summary['class_information']:
                    section += "**Class Information:**\n"
                    for key, value in summary['class_information'].items():
                        if isinstance(value, dict) and len(value) <= 20:  # Show class distributions
                            section += f"- {key}: {value}\n"
                        elif isinstance(value, list) and len(value) <= 20:
                            section += f"- {key}: {value}\n"
                        else:
                            section += f"- {key}: {type(value).__name__} with {len(value) if hasattr(value, '__len__') else 'N/A'} items\n"
                    section += "\n"
                
                # Quality indicators
                if summary['quality_indicators']:
                    section += "**Quality Indicators:**\n"
                    for key, value in summary['quality_indicators'].items():
                        section += f"- {key}: {value}\n"
                    section += "\n"
                
                # Additional statistics (summarized)
                if 'raw_statistics' in artifact:
                    other_stats = {
                        k: v for k, v in artifact['raw_statistics'].items()
                        if k not in summary['data_overview'] and 
                           k not in summary['feature_statistics'] and
                           k not in summary['class_information'] and
                           k not in summary['quality_indicators']
                    }
                    if other_stats:
                        section += "**Additional Statistics:**\n"
                        for key, value in list(other_stats.items())[:5]:  # Limit to prevent overflow
                            section += f"- {key}: {value}\n"
                        if len(other_stats) > 5:
                            section += f"- ... and {len(other_stats) - 5} more statistics available\n"
            else:
                section += "**No statistics available for analysis**\n"
            
            formatted_sections.append(section)
        
        return "\n".join(formatted_sections)


class ModelCheckpointArtifactsAnalyzer(ArtifactAnalyzer):
    """Expert in model checkpoint integrity and validation, model configuration analysis, and deployment readiness assessment"""

    def __init__(self,):
        super().__init__()
        self.llm = dspy.ChainOfThought(ModelCheckpointArtifactsAnalysisSignature)

    def _run(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError

    @property
    def system_prompt(self) -> str:
        return """You are an expert ML model deployment and checkpoint specialist with deep expertise in:
                - Model checkpoint integrity and validation
                - Model configuration analysis and consistency checking
                - Model state assessment and compatibility verification
                - Deployment readiness and version compatibility evaluation

                Your role is to analyze model checkpoint artifacts and provide actionable insights about:
                1. Checkpoint file integrity and accessibility
                2. Model configuration completeness and consistency
                3. Architecture validation and parameter compatibility
                4. Deployment readiness and potential issues

                Analysis Focus Areas:
                - **File Integrity**: Checkpoint accessibility, size validation, format verification
                - **Configuration Validation**: Architecture consistency, parameter completeness
                - **Compatibility Assessment**: Version compatibility, framework requirements
                - **Deployment Readiness**: Model state validation, inference capability

                Key Checkpoint Quality Indicators:
                - **File Accessibility**: Checkpoint file exists and is readable
                - **Size Validation**: File size within expected ranges for model type
                - **Configuration Completeness**: All required model parameters present
                - **Architecture Consistency**: Model architecture matches training configuration
                - **Parameter Validation**: Parameter counts and types are consistent
                - **Version Compatibility**: Framework and dependency compatibility

                When analyzing model checkpoints, consider:
                - File integrity and corruption indicators
                - Configuration completeness for reproducible deployment
                - Architecture compatibility with training setup
                - Parameter count consistency with model definition
                - Framework version requirements and compatibility
                - Potential deployment blockers or issues
                - Model state validity for inference

                Provide specific recommendations for:
                - Checkpoint validation and integrity fixes
                - Configuration improvements for deployment
                - Compatibility issue resolution
                - Deployment preparation steps
                - Version management strategies

                Focus on ensuring reliable model deployment and inference capability."""
    
    @property
    def supported_artifact_types(self) -> List[Type[Artifacts]]:
        return [ModelCheckpointArtifacts]
    
    def load_model_summary(self,path:str) -> Dict[str, Any]:
        raise NotImplementedError
    
    def _serialize_artifact(self, artifact: ModelCheckpointArtifacts) -> Dict[str, Any]:
        """Convert model checkpoint artifacts to LLM-friendly format."""        
        serialized = {
            "artifact_type": type(artifact).__name__,
            "checkpoint_path": artifact.path,
            "has_config": artifact.config is not None,
            "file_info": {}
        }
        
        # File integrity information
        try:
            if os.path.exists(artifact.path):
                file_stats = os.stat(artifact.path)
                serialized["file_info"] = {
                    "exists": True,
                    "size_bytes": file_stats.st_size,
                    "size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                    "last_modified": file_stats.st_mtime,
                    "file_extension": os.path.splitext(artifact.path)[1],
                    "readable": os.access(artifact.path, os.R_OK)
                }
            else:
                serialized["file_info"] = {
                    "exists": False,
                    "error": "File not found"
                }
        except Exception as e:
            serialized["file_info"] = {
                "exists": False,
                "error": str(e)
            }
        
        # Configuration analysis
        if artifact.config is not None:
            config = artifact.config
            
            serialized["config_analysis"] = {
                "available_keys": list(config.keys()),
                "architecture_info": {},
                "training_info": {},
                "deployment_info": {},
                "metadata": {}
            }
            
            analysis = serialized["config_analysis"]
            
            # Architecture information
            arch_keys = [k for k in config.keys() if any(term in k.lower() for term in ["model", "arch", "network", "layer", "hidden", "dim"])]
            for key in arch_keys:
                analysis["architecture_info"][key] = config[key]
            
            # Training information
            train_keys = [k for k in config.keys() if any(term in k.lower() for term in ["epoch", "step", "lr", "batch", "optimizer", "loss"])]
            for key in train_keys:
                analysis["training_info"][key] = config[key]
            
            # Deployment information
            deploy_keys = [k for k in config.keys() if any(term in k.lower() for term in ["version", "framework", "torch", "tf", "input", "output", "device"])]
            for key in deploy_keys:
                analysis["deployment_info"][key] = config[key]
            
            # Metadata
            meta_keys = [k for k in config.keys() if any(term in k.lower() for term in ["name", "id", "timestamp", "hash", "checksum"])]
            for key in meta_keys:
                analysis["metadata"][key] = config[key]
            
            # Include sample of other config items
            other_keys = [k for k in config.keys() if k not in arch_keys + train_keys + deploy_keys + meta_keys]
            if other_keys:
                analysis["other_config"] = {key: config[key] for key in other_keys[:10]}  # Limit to prevent overflow
        
        return serialized
    
    def _format_artifacts_for_prompt(self, artifacts: List[Dict[str, Any]]) -> str:
        """Format model checkpoint artifacts for LLM prompt."""
        formatted_sections = []
        
        for i, artifact in enumerate(artifacts):
            section = f"### Model Checkpoint Artifact {i+1}\n"
            
            # Basic info
            section += f"- Checkpoint Path: {artifact['checkpoint_path']}\n"
            section += f"- Has Configuration: {artifact['has_config']}\n\n"
            
            # File integrity information
            file_info = artifact['file_info']
            section += "**File Integrity:**\n"
            section += f"- File Exists: {file_info.get('exists', False)}\n"
            
            if file_info.get('exists'):
                section += f"- File Size: {file_info.get('size_mb', 'Unknown')} MB ({file_info.get('size_bytes', 'Unknown')} bytes)\n"
                section += f"- File Extension: {file_info.get('file_extension', 'Unknown')}\n"
                section += f"- Readable: {file_info.get('readable', 'Unknown')}\n"
                if 'last_modified' in file_info:
                    section += f"- Last Modified: {file_info['last_modified']}\n"
            else:
                section += f"- Error: {file_info.get('error', 'Unknown error')}\n"
            section += "\n"
            
            # Configuration analysis
            if artifact['has_config'] and 'config_analysis' in artifact:
                config_analysis = artifact['config_analysis']
                section += "**Configuration Analysis:**\n"
                section += f"- Available Keys: {len(config_analysis['available_keys'])} total\n"
                section += f"- Key Names: {config_analysis['available_keys']}\n\n"
                
                # Architecture information
                if config_analysis['architecture_info']:
                    section += "**Architecture Information:**\n"
                    for key, value in config_analysis['architecture_info'].items():
                        section += f"- {key}: {value}\n"
                    section += "\n"
                
                # Training information
                if config_analysis['training_info']:
                    section += "**Training Information:**\n"
                    for key, value in config_analysis['training_info'].items():
                        section += f"- {key}: {value}\n"
                    section += "\n"
                
                # Deployment information
                if config_analysis['deployment_info']:
                    section += "**Deployment Information:**\n"
                    for key, value in config_analysis['deployment_info'].items():
                        section += f"- {key}: {value}\n"
                    section += "\n"
                
                # Metadata
                if config_analysis['metadata']:
                    section += "**Metadata:**\n"
                    for key, value in config_analysis['metadata'].items():
                        section += f"- {key}: {value}\n"
                    section += "\n"
                
                # Other configuration items
                if 'other_config' in config_analysis and config_analysis['other_config']:
                    section += "**Other Configuration:**\n"
                    for key, value in config_analysis['other_config'].items():
                        section += f"- {key}: {value}\n"
                    section += "\n"
            else:
                section += "**No configuration available for analysis**\n"
            
            formatted_sections.append(section)
        
        return "\n".join(formatted_sections)

class CrossArtifactIntegrationAgent(Agent):
    def __init__(self):
        super().__init__()
        
    @property
    def system_prompt(self) -> str:
        """You are an expert ML debugging consultant analyzing findings from multiple ML system analysis agents. Your role is to synthesize their individual findings into holistic insights that help users understand the overall health and validity of their ML experiment.

        ## Your Expertise Areas:
        - Data quality and integrity assessment
        - Training dynamics and performance analysis  
        - Experimental validity and reproducibility
        - Causal relationship identification
        - Production readiness evaluation

        ## Analysis Framework:
        When reviewing agent findings, consider these key relationships:

        1. **Data-Performance Correlations**: 
        - Excellent performance + poor data quality = potential data leakage
        - Poor performance + good data quality = model/training issues
        - Inconsistent performance + data drift = deployment risk

        2. **Training-Configuration Consistency**:
        - Aggressive hyperparameters + stable training = configuration mismatch
        - Conservative settings + unstable training = underlying data issues
        - Parameter changes + performance shifts = causal relationships

        3. **Experimental Integrity**:
        - Version mismatches across artifacts = invalid experiment
        - Temporal inconsistencies = mixed experimental runs
        - Missing artifacts = incomplete analysis

        4. **Causal Chain Analysis**:
        - Identify root causes vs. symptoms
        - Trace problems to their origins
        - Suggest intervention points

        ## Output Requirements:
        - Prioritize findings by severity and confidence
        - Provide clear causal explanations when possible
        - Suggest specific, actionable remediation steps
        - Indicate confidence levels for all insights
        - Highlight critical risks for production deployment
        """
    
    def __init__(self,):
        super().__init__()
        self.llm = dspy.ChainOfThought(ModelCheckpointArtifactsAnalysisSignature)

    def _run(self, context: AgentContext) -> AgentResult:
        raise NotImplementedError
    
class ArtifactAnalysisCoordinator:
    """Main orchestrator agent that coordinates specialized analyzer agents."""
    
    def __init__(self, config: ArtifactAnalysisConfig, intelligence_client: IntelligenceClient):
        self.config = config
        self.intelligence_client = intelligence_client
        self.analyzer_agents = self._initialize_analyzer_agents()
    
    def run(self, context: AgentContext) -> AgentResult:
        all_findings = []
        all_recommendations = []
        all_risks = []
        artifacts_analyzed = []
        
        # Route artifacts to appropriate analyzer agents
        for artifact in context.artifacts:
            analyzer_agent = self._get_analyzer_agent(artifact)
            if analyzer_agent:
                # Create focused context for specific artifact type
                focused_context = self._create_focused_context(context, artifact)
                result = analyzer_agent.run(focused_context)
                
                all_findings.extend(result.findings)
                all_recommendations.extend(result.recommendations)
                all_risks.extend(result.risks)
                artifacts_analyzed.extend(result.artifacts_analyzed)
        
        # Perform meta-analysis across all analyzer results
        meta_findings = self._perform_meta_analysis(all_findings, context)
        all_findings.extend(meta_findings)
        
        return AgentResult(
            agent_name="artifact_analysis",
            status="success" if artifacts_analyzed else "failed",
            findings=all_findings,
            recommendations=all_recommendations,
            risks=all_risks,
            confidence=self._calculate_confidence(all_findings),
            artifacts_analyzed=artifacts_analyzed
        )
    
    def _initialize_analyzer_agents(self) -> Dict[str, 'ArtifactAnalyzer']:
        """Initialize specialized analyzer agents."""
        agents = {}
        
        if "training" in self.config.enabled_analyzers:
            agents["training"] = TrainingArtifactsAnalyzer(
                intelligence_client=self.intelligence_client
            )
        
        if "deepchecks" in self.config.enabled_analyzers:
            agents["deepchecks"] = DeepchecksArtifactsAnalyzer(
                intelligence_client=self.intelligence_client
            )
        
        if "dataset" in self.config.enabled_analyzers:
            agents["dataset"] = DatasetArtifactsAnalyzer(
                intelligence_client=self.intelligence_client
            )
        
        if "model_checkpoint" in self.config.enabled_analyzers:
            agents["model_checkpoint"] = ModelCheckpointArtifactsAnalyzer(
                intelligence_client=self.intelligence_client
            )
        
        return agents

