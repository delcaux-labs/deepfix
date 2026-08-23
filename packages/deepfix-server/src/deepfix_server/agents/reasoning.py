"""Cross-artifact reasoning synthesis for multi-agent findings."""

from __future__ import annotations

import json
import traceback
from typing import Dict, List, Optional

from deepfix_core.models import Severity

from deepfix_kb import KnowledgeBridge
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..logging import get_logger
from .schemas import AgentResult, CrossArtifactReasoningResult

LOGGER = get_logger(__name__)

from .prompts import CROSS_ARTIFACT_SYSTEM_PROMPT


def serialize_previous_analyses(previous_analyses: Dict[str, AgentResult]) -> str:
    """Serialize previous agent analyses into a readable JSON string."""
    analyses_dict = {}
    for name, ar in previous_analyses.items():
        if name == "CrossArtifactReasoningAgent":
            continue
        entry: dict = {}
        if ar.analysis is not None:
            entry["analysis"] = [a.model_dump() for a in ar.analysis]
        if ar.error_message is not None:
            entry["error"] = ar.error_message
        if ar.analyzed_artifacts:
            entry["analyzed_artifacts"] = ar.analyzed_artifacts
        analyses_dict[name] = entry

    return json.dumps(analyses_dict, indent=2, default=str)



SEVERITY_WEIGHTS = {
    Severity.HIGH: 3,
    "high": 3,
    Severity.MEDIUM: 2,
    "medium": 2,
    Severity.LOW: 1,
    "low": 1,
}


async def prefetch_knowledge(
    previous_analyses: Dict[str, AgentResult],
    knowledge_bridge: Optional[KnowledgeBridge],
    max_queries: int = 3,
) -> List[str]:
    """Retrieve relevant domain knowledge based on findings prioritized by severity and confidence."""
    if not knowledge_bridge or not knowledge_bridge.has_available_sources:
        return []

    prioritized_findings: List[tuple[int, float, str]] = []
    for ar in previous_analyses.values():
        if ar.analysis:
            for analysis in ar.analysis:
                finding = analysis.findings
                if finding and finding.description:
                    weight = SEVERITY_WEIGHTS.get(finding.severity, 0) if finding.severity else 0
                    confidence = finding.confidence if finding.confidence is not None else 0.0
                    prioritized_findings.append((weight, confidence, finding.description))

    # Sort descending by severity weight, then by confidence score
    prioritized_findings.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # Deduplicate queries while preserving priority order
    seen: set[str] = set()
    queries: List[str] = []
    for _, _, desc in prioritized_findings:
        if desc not in seen:
            seen.add(desc)
            queries.append(desc)

    retrieved: List[str] = []
    for query in queries[:max_queries]:  # Limit to top findings to avoid excessive lookups
        try:
            response = await knowledge_bridge.query(query, synthesize=False, max_results=2)
            if response and response.results:
                for r in response.results:
                    retrieved.append(f"[{r.source_type}] {r.content}")
        except Exception as e:
            LOGGER.warning("Knowledge prefetch error for '%s': %s", query[:50], e)

    return retrieved


def build_cross_artifact_prompt(
    previous_analyses_json: str,
    knowledge_context: List[str],
    output_language: str,
) -> str:
    """Construct the prompt for cross-artifact reasoning synthesis."""
    sections = [
        "You are analyzing findings from multiple Machine Learning system analysis agents.",
        "Synthesize their individual findings into holistic insights and actionable recommendations.\n",
        f"## Previous Analyses\n\n{previous_analyses_json}\n",
    ]

    if knowledge_context:
        kb_text = "\n".join(f"- {item}" for item in knowledge_context)
        sections.append(f"## Retrieved ML Domain Knowledge\n\n{kb_text}\n")

    sections.append(f"Output language: {output_language}\n")
    sections.append("Produce a consolidated analysis with cross-artifact insights, clear root causes, and a summary.")

    return "\n".join(sections)


async def run_cross_artifact_reasoning(
    previous_analyses: Dict[str, AgentResult],
    llm: BaseChatModel,
    knowledge_bridge: Optional[KnowledgeBridge] = None,
    output_language: str = "english",
) -> AgentResult:
    """Execute cross-artifact reasoning synthesis given prior agent analyses.

    Args:
        previous_analyses: Results collected from analyzer agents.
        llm: Configured LangChain BaseChatModel.
        knowledge_bridge: Optional KnowledgeBridge for domain context.
        output_language: Target language for output.

    Returns:
        AgentResult with synthesized findings, recommendations, and summary.
    """
    agent_name = "CrossArtifactReasoningAgent"
    try:
        LOGGER.info("Running %s synthesis...", agent_name)

        # 1. Serialize analyses
        analyses_json = serialize_previous_analyses(previous_analyses)

        # 2. Pre-fetch relevant knowledge
        retrieved_knowledge = await prefetch_knowledge(previous_analyses, knowledge_bridge)

        # 3. Construct user prompt
        user_message = build_cross_artifact_prompt(
            previous_analyses_json=analyses_json,
            knowledge_context=retrieved_knowledge,
            output_language=output_language,
        )

        # 4. Invoke LLM with structured output
        structured_llm = llm.with_structured_output(CrossArtifactReasoningResult)
        messages = [
            SystemMessage(content=CROSS_ARTIFACT_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        result: CrossArtifactReasoningResult = await structured_llm.ainvoke(messages)

        # 5. Collect metadata
        analyzed_artifacts: list = []
        for ar in previous_analyses.values():
            if ar.analyzed_artifacts:
                analyzed_artifacts.extend(ar.analyzed_artifacts)

        return AgentResult(
            agent_name=agent_name,
            analysis=result.analysis,
            analyzed_artifacts=list(set(analyzed_artifacts)),
            retrieved_knowledge=retrieved_knowledge,
            additional_outputs={"summary": result.summary},
        )

    except Exception as e:
        LOGGER.error("Error in %s: %s", agent_name, traceback.format_exc())
        return AgentResult(
            agent_name=agent_name,
            error_message=str(e),
        )
