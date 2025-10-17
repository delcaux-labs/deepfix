"""Knowledge base manager for document loading and indexing"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Callable
from datetime import datetime

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
)

from ..agents.models import KnowledgeDocument, KnowledgeDomain, KnowledgeItem
from ...utils.logging import get_logger
from ..config import DefaultPaths

logger = get_logger(__name__)


class KnowledgeBaseManager:
    """Manages knowledge base documents and indexing"""
    
    def __init__(
        self, 
        embed_model: Callable,
        documents_dir: Optional[Path] = None,
        index_dir: Optional[Path] = None,
    ):                
        self.documents_dir = documents_dir or Path(DefaultPaths.KNOWLEDGE_BASE_DOCUMENTS_DIR.value)
        self.index_dir = index_dir or Path(DefaultPaths.KNOWLEDGE_BASE_INDICES_DIR.value)
        self.embed_model = embed_model        
        self.documents: List[KnowledgeDocument] = []
        self.indices: Dict[str, VectorStoreIndex] = {}
        
    def load_documents(self) -> List[KnowledgeDocument]:
        """Load all knowledge documents from JSON files"""
        logger.info(f"Loading documents from {self.documents_dir}")
        documents = []
        
        for json_file in self.documents_dir.glob("*.json"):
            logger.info(f"Loading {json_file.name}")
            with open(json_file, 'r') as f:
                data = json.load(f)
                for doc_data in data:
                    # Convert last_updated string to datetime if present
                    if 'last_updated' in doc_data and doc_data['last_updated']:
                        doc_data['last_updated'] = datetime.fromisoformat(doc_data['last_updated'])
                    
                    doc = KnowledgeDocument(**doc_data)
                    documents.append(doc)
        
        self.documents = documents
        logger.info(f"Loaded {len(documents)} knowledge documents")
        return documents
    
    def get_index_name(self, domain: KnowledgeDomain) -> str:
        return f"{domain.value}_index"
    
    def get_index_path(self, domain: KnowledgeDomain) -> Path:
        return self.index_dir / self.get_index_name(domain)

    def _create_llama_documents(self, kb_docs: List[KnowledgeDocument]) -> List[Document]:
        """Convert KnowledgeDocument to LlamaIndex Document format"""
        llama_docs = []
        
        for kb_doc in kb_docs:
            # Create rich text with metadata
            text = f"Title: {kb_doc.title}\n\n{kb_doc.content}"
            
            if kb_doc.examples:
                text += "\n\nExamples:\n" + "\n".join(f"- {ex}" for ex in kb_doc.examples)
            
            # Create metadata dict
            metadata = {
                "title": kb_doc.title,
                "domain": kb_doc.domain.value,
                "knowledge_type": kb_doc.knowledge_type.value,
                "source": kb_doc.source,
                "confidence_level": kb_doc.confidence_level,
                "tags": ", ".join(kb_doc.tags),
                "ml_frameworks": ", ".join(kb_doc.ml_frameworks),
                "model_types": ", ".join(kb_doc.model_types),
            }
            
            llama_doc = Document(text=text, metadata=metadata)
            llama_docs.append(llama_doc)
        
        return llama_docs
    
    def build_index(self, domain: KnowledgeDomain) -> VectorStoreIndex:
        """Build vector index for documents, optionally filtered by domain"""
        if len(self.documents) == 0:
            self.load_documents()
        
        # Filter by domain if specified
        filtered_docs = [d for d in self.documents if d.domain == domain]
        index_name = self.get_index_name(domain)
        
        logger.info(f"Building index '{index_name}' with {len(filtered_docs)} documents")
        
        # Convert to LlamaIndex documents
        llama_docs = self._create_llama_documents(filtered_docs)
                
        index = VectorStoreIndex.from_documents(
            llama_docs,
            embed_model=self.embed_model,
            show_progress=True
        )
        
        self.indices[index_name] = index
        logger.info(f"Index '{index_name}' built successfully")
        
        return index
    
    def get_index(self, domain: KnowledgeDomain) -> VectorStoreIndex:
        """Get index for domain, building if necessary"""
        index_name = self.get_index_name(domain)
        
        if index_name not in self.indices:
            self.build_index(domain)
        
        return self.indices[index_name]
    
    def save_index(self, domain: KnowledgeDomain)->None:
        """Save index to disk"""
        index_name = self.get_index_name(domain)
        
        if index_name not in self.indices:
            logger.warning(f"Index '{index_name}' not found, building it first")
            self.build_index(domain)
        
        index = self.indices[index_name]
        persist_path = self.get_index_path(domain)
        index.storage_context.persist(persist_dir=str(persist_path))
        logger.info(f"Index '{index_name}' saved to {persist_path}")
    
    def load_index(self, domain: KnowledgeDomain) -> VectorStoreIndex:
        """Load index from disk"""
                
        index_name = self.get_index_name(domain)
        persist_path = self.get_index_path(domain)
        
        if not persist_path.exists():
            logger.warning(f"Index '{index_name}' not found on disk, building new one")
            return self.build_index(domain)
        
        logger.info(f"Loading index '{index_name}' from {persist_path}")
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
        index = load_index_from_storage(storage_context)
        
        self.indices[index_name] = index
        return index
    
    def get_documents_by_domain(self, domain: KnowledgeDomain) -> List[KnowledgeDocument]:
        """Get all documents for a specific domain"""
        if not self.documents:
            self.load_documents()
        return [d for d in self.documents if d.domain == domain]
    
    def search_documents(
        self, 
        query: str, 
        domain: KnowledgeDomain = KnowledgeDomain.GLOBAL,
        top_k: int = 5
    ) -> List[KnowledgeItem]:
        """Search knowledge base and return results"""
        index = self.get_index(domain)
        retriever = index.as_retriever(similarity_top_k=top_k)
        
        response = retriever.retrieve(query)
        
        results = [KnowledgeItem(
            content=node.text,
            source=node.metadata.get("source", "Unknown"),
            confidence=node.metadata.get("confidence_level", None),
            relevance_score=node.score,
            metadata=node.metadata
        ) for node in response.source_nodes]
               
        return results
