"""DSPy signatures for agent reasoning"""
import dspy
from typing import List, Optional
from .models import Analysis,AgentResult

class ArtifactAnalysisSignature(dspy.Signature):
    """Analyze dataset, model checkpoint, training artifacts for issues and recommendations"""
    artifacts: str = dspy.InputField(desc="Structured artifacts (dataset, model checkpoint, training artifacts)")
    system_prompt: Optional[str] = dspy.InputField(desc="System instructions for the analyzer")
    
    analysis: List[Analysis] = dspy.OutputField(desc="Findings and recommendations based on the artifacts")


class CrossArtifactReasoningSignature(dspy.Signature):
    """Integrate findings from multiple artifact analyzers"""
    previous_analyses: List[AgentResult] = dspy.InputField(desc="Results from multiple artifact analyzers")
    system_prompt: Optional[str] = dspy.InputField(desc="System instructions for the reasoning agent")
    
    analysis: List[Analysis] = dspy.OutputField(desc="Consolidated analysis with cross-artifact insights. Findings and recommendations based on the agents results")
    summary: str = dspy.OutputField(desc="Summary of the cross-artifact reasoning & analysis")


class QueryGenerationSignature(dspy.Signature):
    """Transform agent knowledge request into optimized retrieval queries"""
    
    agent_context: str = dspy.InputField(
        desc="Context from requesting agent (findings, artifacts, constraints)"
    )
    domain: str = dspy.InputField(
        desc="Knowledge domain (training, data_quality, optimization, architecture)"
    )
    query_type: str = dspy.InputField(
        desc="Type of knowledge needed (best_practice, diagnostic, solution, validation)"
    )
    
    retrieval_queries: List[str] = dspy.OutputField(
        desc="List of 3-5 optimized queries for multi-aspect retrieval"
    )
    search_strategy: str = dspy.OutputField(
        desc="Retrieval strategy (semantic, hybrid, keyword-based)"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief reasoning for query formulation"
    )


class EvidenceValidationSignature(dspy.Signature):
    """Validate and score retrieved evidence"""
    
    context: str = dspy.InputField(
        desc="Original request context and requirements"
    )
    evidence: str = dspy.InputField(
        desc="Retrieved evidence to validate"
    )
    question: str = dspy.InputField(
        desc="Original question or query"
    )
    
    confidence_score: str = dspy.OutputField(
        desc="Confidence score between 0.0 and 1.0"
    )
    relevance_explanation: str = dspy.OutputField(
        desc="Explanation of why this evidence is or isn't relevant"
    )
    is_actionable: str = dspy.OutputField(
        desc="Whether the evidence provides actionable insights (yes/no)"
    )


class ResponseSynthesisSignature(dspy.Signature):
    """Synthesize multiple evidence pieces into coherent response"""
    
    original_query: str = dspy.InputField(
        desc="Original knowledge request query"
    )
    evidence_items: str = dspy.InputField(
        desc="Retrieved evidence items with scores"
    )
    domain: str = dspy.InputField(
        desc="Knowledge domain context"
    )
    
    synthesis: str = dspy.OutputField(
        desc="Coherent summary synthesizing all evidence"
    )
    key_insights: List[str] = dspy.OutputField(
        desc="3-5 key insights extracted from evidence"
    )
    supporting_points: List[str] = dspy.OutputField(
        desc="Main supporting points from evidence"
    )
    contradictions: str = dspy.OutputField(
        desc="Any contradictions or caveats found in evidence"
    )