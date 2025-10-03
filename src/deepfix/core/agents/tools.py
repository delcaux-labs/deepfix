from abc import ABC, abstractmethod
from llama_index.core import VectorStoreIndex

class Tool(ABC):
    """Tools for the Knowledge Bridge"""

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def __call__(self, query: str) -> str:
        pass

    def _format_with_citations(self, response) -> str:
        """Format response with source citations for traceability"""
        citations = [
            f"[{i+1}] {node.metadata['source']}" 
            for i, node in enumerate(response.source_nodes)
        ]
        return f"{response.response}\n\nSources:\n" + "\n".join(citations)


class DomainRetrieverTool(Tool):
    """Domain-specific knowledge retrieval"""
    
    def __init__(self, index: VectorStoreIndex, domain: str,similarity_top_k=5):
        super().__init__()
        self.query_engine = index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode="tree_summarize"
        )
        self.domain = domain
    
    def __call__(self, query: str) -> str:
        """Execute retrieval and return formatted results"""
        response = self.query_engine.query(query)
        return self._format_with_citations(response)
    
class SemanticRetriever(DomainRetrieverTool):
    """Semantic knowledge retrieval"""
    
    def __init__(self, index: VectorStoreIndex, domain: str,similarity_top_k=5):
        super().__init__(index=index, domain=domain,similarity_top_k=similarity_top_k)

class ExampleRetriever(DomainRetrieverTool):
    """Example retrieval"""
    
    def __init__(self, index: VectorStoreIndex, domain: str,similarity_top_k=5):
        super().__init__(index=index, domain=domain,similarity_top_k=similarity_top_k)