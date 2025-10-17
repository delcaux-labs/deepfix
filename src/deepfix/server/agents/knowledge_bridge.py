"""KnowledgeBridge: Intelligent knowledge retrieval agent"""
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import time
from pathlib import Path
import dspy
from functools import partial
from concurrent.futures import ThreadPoolExecutor


from .base import Agent
from ..models import (
    AgentKnowledgeRequest, KnowledgeResponse, 
    KnowledgeItem, KnowledgeDomain, EvidenceValidationResult, QueryGenerationResult
)
from .signatures import (
    QueryGenerationSignature, 
    EvidenceValidationSignature,
    KnowledgeBridgeReActSignature,
)
from ..config import LLMConfig
from ..logging import get_logger
from ..knowledge_base.manager import KnowledgeBaseManager

LOGGER = get_logger(__name__)


class QueryGenerator(Agent):
    """Query generator for knowledge retrieval"""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        super().__init__(config=llm_config)
        self.query_generator = dspy.ChainOfThought(QueryGenerationSignature)
    
    def forward(self, request: AgentKnowledgeRequest) -> QueryGenerationResult:
        """Generate optimized queries using DSPy"""
        with self._llm_context():
            out = self.query_generator(request=request)
        return out.result


class KnowledgeBridge(Agent):
    """Knowledge retrieval agent for providing evidence-based recommendations"""
    
    def __init__(
        self, 
        documents_dir: str,
        embed_model: Callable,
        llm_config: Optional[LLMConfig] = None,        
        num_workers: int = 3
    ):
        super().__init__(config=llm_config)        
        self.num_workers = num_workers
        self.kb_manager = KnowledgeBaseManager(documents_dir=Path(documents_dir), embed_model=embed_model)
        # Load documents on initialization
        self.kb_manager.load_documents()
        
        self._indices_loaded = False
        
        # Initialize DSPy modules
        self.query_generator = QueryGenerator(llm_config=llm_config)
        self.evidence_validator = dspy.ChainOfThought(EvidenceValidationSignature)
    
    def _ensure_indices(self):
        """Lazy load indices on first use"""
        if not self._indices_loaded and self.kb_manager:
            LOGGER.info("Building knowledge base indices...")
            # Build indices for each domain
            for domain in KnowledgeDomain:
                self.kb_manager.build_index(domain)
            self._indices_loaded = True
            LOGGER.info("Knowledge base indices ready")
        
    def _generate_queries(self, request: AgentKnowledgeRequest) -> List[str]:
        """Generate optimized retrieval queries using DSPy"""                
        # Generate queries with DSPy
        with self._llm_context():
            out = self.query_generator(
                request=request,
            )        
        queries = out.result.retrieval_queries
        LOGGER.debug(f"Generated {len(queries)} queries for {request.requesting_agent}.")        
        return queries
    
    def _validate_evidence(
        self,query: str, request: AgentKnowledgeRequest, retrieved_evidences: List[KnowledgeItem]
    ) -> List[EvidenceValidationResult]:
        """Validate evidence using DSPy"""
        
        with self._llm_context():
            out = self.evidence_validator(question=query, context=request, retrieved_evidences=retrieved_evidences)
                
        return out.result
    
    def _retrieve_evidence(self, query: str, domain: KnowledgeDomain, top_k: int = 5) -> List[KnowledgeItem]:
        """Retrieve evidence using DSPy"""
        results = self.kb_manager.search_documents(
                query=query,
                domain=domain,
                top_k=top_k
            )
        return results
    
    def _eval_one_query(self, query: str, request: AgentKnowledgeRequest) -> Tuple[str, List[KnowledgeItem], List[EvidenceValidationResult]]:
        """Evaluate one query and return the results"""
        results = self._retrieve_evidence(query=query, domain=request.domain, top_k=request.max_results)
        validation_result = self._validate_evidence(query=query, request=request, retrieved_evidences=results)
        return query, results, validation_result
    
    def _format_response(self, query: str, evidence: List[KnowledgeItem], validation: List[EvidenceValidationResult]) -> KnowledgeResponse:
        return KnowledgeResponse(
            query=query,
            retrieved_knowledge=evidence,
            validation_results=validation,
        )        
    
    def forward(
        self, 
        request: AgentKnowledgeRequest
    ) -> List[KnowledgeResponse]:
        """
        Main retrieval method with DSPy-enhanced query generation and synthesis
        
        Args:
            request: Knowledge request from agent
            
        Returns:
            KnowledgeResponse with retrieved knowledge and metadata
        """
        
        # 0. Ensure indices are loaded
        self._ensure_indices()
        
        # 1. Generate optimized queries
        queries = self._generate_queries(request)
        
        # 2. Retrieve and validate from knowledge base for each query
        start_time = time.time()
        responses = []
        func = partial(self._eval_one_query, request=request)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for query,evidence, validation in executor.map(func, queries):        
                r = self._format_response(query=query, evidence=evidence, validation=validation)
                responses.append(r)
        end_time = time.time()
        LOGGER.info(f"{len(responses)} responses returned in {end_time - start_time:.2f}s")
        
        return responses
    
    @property
    def system_prompt(self) -> str:
        return (
            "You are KnowledgeBridge, an intelligent knowledge retrieval agent. "
            "Your role is to provide evidence-based insights and best practices "
            "from the machine learning knowledge base to support other agents' "
            "recommendations and analyses."
        )


class KnowledgeBridgeReActAgent(KnowledgeBridge):
    """ReAct-based agentic RAG for knowledge retrieval"""
    
    def __init__(self, llm_config: LLMConfig, documents_dir: str, embed_model: Callable, max_iters: int = 3):
        super().__init__(
            llm_config=llm_config,
            documents_dir=documents_dir,
            embed_model=embed_model,
        )
        self.orchestrator = dspy.ReAct(KnowledgeBridgeReActSignature, tools=self.get_tools(),max_iters=max_iters)    

    def _get_knowledge_domain(self, request: AgentKnowledgeRequest) -> KnowledgeDomain:
        return request.domain
    
    def _get_top_k(self, request: AgentKnowledgeRequest) -> int:
        return request.max_results

    def get_tools(self) -> List[dspy.Tool]:
        return [
            dspy.Tool(
                name="query_generator",
                desc="Generate optimized queries for knowledge retrieval",
                func=self._generate_queries,
            ),
            dspy.Tool(
                name="evidence_validator",
                desc="Validate evidence for a given query",
                func=self._validate_evidence,
            ),            
            dspy.Tool(
                name="evidence_retriever",
                desc="Retrieve evidence for a given query",
                func=self._retrieve_evidence
            ),
            dspy.Tool(
                name="knowledge_domain",
                desc="Get knowledge domain for a given request",
                func=self._get_knowledge_domain
            ),
            dspy.Tool(
                name="top_k",
                desc="Get top k for a given request",
                func=self._get_top_k
            ),
        ]
    
    def forward(self, request: AgentKnowledgeRequest) -> List[KnowledgeResponse]:
        
        # 0. Ensure indices are loaded
        self._ensure_indices()

        start_time = time.time()
        
        # 1. Run the orchestrator
        with self._llm_context():
            out = self.orchestrator(request=request)

        # 2. Format the responses
        responses = []
        for query, evidence, validation in zip(out.queries, out.retrieved_evidences, out.evidence_validations):
            r = self._format_response(query=query, evidence=evidence, validation=validation)
            responses.append(r)
        
        end_time = time.time()        
        LOGGER.info(f"{len(responses)} responses returned in {end_time - start_time:.2f}s")      
        return responses