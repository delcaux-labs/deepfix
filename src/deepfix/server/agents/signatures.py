"""DSPy signatures for agent reasoning"""
import dspy
from typing import List, Optional
from .models import (Analysis,
                    AgentKnowledgeRequest,
                    AgentResult,
                    QueryGenerationResult,
                    EvidenceValidationResult,
                    KnowledgeItem
                )

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
    
    request: AgentKnowledgeRequest = dspy.InputField(
        desc="Context from requesting agent (findings, artifacts, constraints)"
    )    
    
    result: QueryGenerationResult = dspy.OutputField(
        desc="Query generation result with optimized queries, search strategy, rationale etc."
    )


class EvidenceValidationSignature(dspy.Signature):
    """Validate and score retrieved evidences for a given question"""
    
    question: str = dspy.InputField(description="Original question or query")
    context: AgentKnowledgeRequest = dspy.InputField(description="Requesting agent's context")    
    retrieved_evidences: List[KnowledgeItem] = dspy.InputField(description="Retrieved evidences to validate")
    
    result: List[EvidenceValidationResult] = dspy.OutputField(
        desc="Evidence validation result for each evidence"
    )


class KnowledgeBridgeReActSignature(dspy.Signature):
    """Retrieve evidence from knowledge base for a given question"""
    
    request: AgentKnowledgeRequest = dspy.InputField(description="Requesting agent's context")
    
    queries: List[str] = dspy.OutputField(description="Generated queries")
    retrieved_evidences: List[List[KnowledgeItem]] = dspy.OutputField(description="List of retrieved evidences for each query")
    evidence_validations: List[List[EvidenceValidationResult]] = dspy.OutputField(
        desc="List of evidence validation results for each retrieved evidence"
    )


class ResponseSynthesisSignature(dspy.Signature):
    """Synthesize multiple evidence pieces into coherent response"""
    
    original_query: str = dspy.InputField(
        desc="Original knowledge request query"
    )
    evidence_items: List[KnowledgeItem] = dspy.InputField(
        desc="Retrieved evidence items with scores"
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