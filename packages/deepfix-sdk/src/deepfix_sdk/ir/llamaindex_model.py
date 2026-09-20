from pathlib import Path
from typing import Any, List, Optional

import lancedb
import numpy as np
import pandas as pd
import Stemmer
from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers.fusion_retriever import (
    QueryFusionRetriever,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator, ClassifierMixin

from ..async_utils import run_async
from ..logging import get_logger
from ..settings import settings
from .dataset import InformationRetrievalDataset

LOGGER = get_logger(__name__)


class RetrieverEvent(Event):
    """Result of running retrieval."""

    nodes: list[NodeWithScore]


class RerankEvent(Event):
    """Result of running reranking on retrieved nodes."""

    nodes: list[NodeWithScore]


class RetrievalResult(BaseModel):
    """Retrieved document candidate."""

    doc_id: str = Field(..., description="Unique identifier for the document")
    score: float = Field(..., description="Retrieval score")
    text: str = Field(..., description="The text of the document")
    embedding: list[float] | None = Field(
        default=None, description="The vector embedding of the document node"
    )


class RetrievalWorkflow(Workflow):
    """Event-driven RAG Workflow supporting Ingestion, Retrieval, Reranking, and Synthesis."""

    def __init__(
        self,
        dataset: InformationRetrievalDataset,
        load_if_exists: bool = True,
        lancedb_index_dir: str = "lancedb",
        top_k: int = 5,
        language: str = "english",
        retrieval_mode: str = "hybrid",
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        enable_reranking: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.load_if_exists = load_if_exists
        self.lancedb_index_dir = lancedb_index_dir
        self.top_k = top_k
        self.language = language
        self.retrieval_mode = retrieval_mode
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.enable_reranking = enable_reranking
        self._index: BaseIndex = None
        self._embed_model: BaseEmbedding = None
        self._reranker: CohereRerank | None = None

        assert retrieval_mode in ["dense", "hybrid"], (
            f"Retriever mode must be 'dense' or 'hybrid', but got '{retrieval_mode}'"
        )
        assert dense_weight + bm25_weight == 1, (
            f"Dense and BM25 weights must sum to 1, but got {dense_weight} + {bm25_weight}"
        )

    def _get_embed_model(self) -> BaseEmbedding:
        if self._embed_model is not None:
            return self._embed_model
        self._embed_model = OpenAIEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            api_base=settings.EMBEDDING_BASE_URL,
            api_key=settings.EMBEDDING_API_KEY,
        )
        return self._embed_model

    def _get_reranker(self) -> CohereRerank | None:
        if self._reranker is not None:
            return self._reranker
        if not self.enable_reranking:
            return None
        self._reranker = CohereRerank(
            model=settings.COHERE_MODEL,
            top_n=self.top_k,
            api_key=settings.COHERE_API_KEY,
            base_url=settings.COHERE_BASE_URL,
        )
        return self._reranker

    def _get_vector_store(self) -> BasePydanticVectorStore:

        uri = self.lancedb_index_dir
        storage_options: dict[str, str] = {}

        if uri.startswith("s3://"):
            if settings.S3_ACCESS_KEY_ID:
                storage_options["aws_access_key_id"] = str(settings.S3_ACCESS_KEY_ID)
            if settings.S3_SECRET_ACCESS_KEY:
                storage_options["aws_secret_access_key"] = str(
                    settings.S3_SECRET_ACCESS_KEY
                )
            if settings.S3_REGION:
                storage_options["region"] = str(settings.S3_REGION)
            if settings.S3_ENDPOINT_URL:
                storage_options["endpoint"] = str(settings.S3_ENDPOINT_URL)

        conn = lancedb.connect(
            uri=uri,
            storage_options=storage_options if storage_options else None,
        )
        return LanceDBVectorStore(uri=uri, connection=conn, table_name="documents")

    def _get_storage_context(self, persist_dir=None) -> StorageContext:
        return StorageContext.from_defaults(
            persist_dir=persist_dir,
            vector_store=self._get_vector_store(),
            image_store=None,  # images of scanned PDFs
        )

    def _index_exists(self) -> bool:
        """Check if an index already exists for the configured storage backend."""
        try:
            uri = self.lancedb_index_dir
            if not str(uri).startswith("s3://") and not Path(uri).exists():
                return False
            vector_store = self._get_vector_store()
            conn = getattr(vector_store, "_connection", None)
            if conn is not None:
                table_name = getattr(vector_store, "_table_name", None)
                if table_name is None:
                    return False
                tables_res = (
                    conn.list_tables()
                    if hasattr(conn, "list_tables")
                    else conn.table_names()
                )
                tables = getattr(tables_res, "tables", tables_res)
                tables = list(tables) if not isinstance(tables, list) else tables
                if table_name in tables:
                    tbl = conn.open_table(table_name)
                    count = tbl.count_rows() if hasattr(tbl, "count_rows") else len(tbl)
                    return count > 0
            return False
        except Exception as e:
            LOGGER.debug(f"Index existence check failed: {e}")
            return False

    def _load_existing_index(self) -> BaseIndex:
        """Load an existing index from the storage backend."""
        vector_store = self._get_vector_store()
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self._get_embed_model(),
        )

    def _build_index(
        self, documents: list[Document], storage_context=None, **kwargs
    ) -> BaseIndex:
        if storage_context is None:
            storage_context = self._get_storage_context()

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self._get_embed_model(),
        )
        return index

    def _get_bm25_retriever(
        self,
        index: BaseIndex,
    ) -> BaseRetriever:
        """Construct a BM25 sparse lexical retriever."""
        bm25_kwargs = {}
        if hasattr(index, "docstore") and len(getattr(index.docstore, "docs", {})) > 0:
            bm25_kwargs["index"] = index
        elif hasattr(index, "nodes") and len(getattr(index, "nodes", [])) > 0:
            bm25_kwargs["nodes"] = index.nodes
        else:
            nodes = [
                Document(
                    text=doc.get("text") or doc.get("body") or "",
                    doc_id=str(doc.get("docno") or doc.get("doc_id") or ""),
                    metadata={
                        "doc_id": str(doc.get("docno") or doc.get("doc_id") or "")
                    },
                )
                for doc in self.dataset.get_corpus_iter()
            ]
            bm25_kwargs["nodes"] = nodes

        retrieval_k = max(self.top_k * 4, 20) if self.enable_reranking else self.top_k
        bm25 = BM25Retriever.from_defaults(
            similarity_top_k=retrieval_k,
            language=self.language,
            stemmer=Stemmer.Stemmer(self.language),
            **bm25_kwargs,
        )
        return bm25

    def _build_retriever(
        self,
        index: BaseIndex,
        use_async: bool = True,
    ) -> BaseRetriever:
        backend = self.retrieval_mode

        retrieval_k = max(self.top_k * 4, 20) if self.enable_reranking else self.top_k

        if backend == "dense":
            return index.as_retriever(similarity_top_k=retrieval_k)

        elif backend == "hybrid":
            dense_retriever = index.as_retriever(similarity_top_k=retrieval_k)
            bm25_retriever = self._get_bm25_retriever(index)
            from llama_index.core.llms.mock import MockLLM

            return QueryFusionRetriever(
                [dense_retriever, bm25_retriever],
                retriever_weights=[
                    self.dense_weight,
                    self.bm25_weight,
                ],
                llm=getattr(self, "llm", None) or MockLLM(),
                num_queries=1,
                use_async=use_async,
                similarity_top_k=retrieval_k,
            )

        else:
            raise ValueError(f"Invalid retrieval backend: {backend}")

    @step
    async def ingestion(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point for Ingestion."""
        ingest: bool = ev.get("ingest", False)
        if ingest:
            if self.load_if_exists and self._index_exists():
                LOGGER.info("Existing index detected. Loading pre-built index...")
                self._index = self._load_existing_index()
            else:
                LOGGER.info("Ingesting corpus documents...")
                documents = [
                    Document(
                        text=doc.get("text") or doc.get("body") or "",
                        doc_id=str(doc.get("docno") or doc.get("doc_id") or ""),
                        metadata={
                            "doc_id": str(doc.get("docno") or doc.get("doc_id") or "")
                        },
                    )
                    for doc in self.dataset.get_corpus_iter()
                ]
                self._index = self._build_index(documents)

            return StopEvent(result=self._index)

        return None

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        """Entry point for RAG, triggered by a StartEvent with `query`."""
        query = ev.get("query", None)
        index: BaseIndex | None = ev.get("index", self._index)

        if not query:
            return None

        if index is None:
            LOGGER.warning("Query provided but no index available. Skipping ...")
            return None

        LOGGER.debug(f"Query: {query}")
        await ctx.store.set("query", query)

        retriever = self._build_retriever(
            index=index,
            use_async=True,
        )
        nodes = await retriever.aretrieve(query)

        LOGGER.debug(f"Retrieved {len(nodes)} nodes")

        return RetrieverEvent(nodes=nodes)

    @step
    async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
        """Rerank candidate nodes using CohereRerank with custom base_url."""
        query = await ctx.get("query")
        nodes = ev.nodes
        reranker = self._get_reranker()

        if reranker is None:
            LOGGER.debug("No reranker configured")
            return RerankEvent(nodes=nodes)

        nodes = reranker.postprocess_nodes(nodes, query_str=query)
        LOGGER.debug(f"Reranked {len(nodes)} nodes")

        return RerankEvent(nodes=nodes)

    def convert_nodes_to_results(
        self, nodes: list[NodeWithScore]
    ) -> list[RetrievalResult]:
        """Convert a list of NodeWithScore objects to standardized RetrievalResult models."""
        results = []
        for node in nodes:
            doc_id_val = node.node.metadata.get("doc_id", node.node.node_id)
            embedding = getattr(
                node.node, "embedding", getattr(node, "embedding", None)
            )
            if embedding is not None and not isinstance(embedding, list):
                embedding = list(embedding)
            results.append(
                RetrievalResult(
                    doc_id=str(doc_id_val),
                    score=node.score or 0.0,
                    text=node.node.get_content(),
                    embedding=embedding,
                )
            )
        return results

    @step
    async def end(self, ctx: Context, ev: RerankEvent) -> StopEvent:
        """Synthesize response using CompactAndRefine."""
        response_model = self.convert_nodes_to_results(ev.nodes)
        return StopEvent(result=response_model)


class LlamaindexModel(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible estimator wrapping LlamaIndex RetrievalWorkflow.

    Provides a standard scikit-learn interface (fit, predict, predict_proba)
    for Information Retrieval tasks to interface with scikit-learn pipelines,
    Deepchecks suites, and the DeepFix diagnostic platform.
    """

    def __init__(
        self,
        workflow: Optional[RetrievalWorkflow] = None,
        dataset: Optional[InformationRetrievalDataset] = None,
        top_k: int = 5,
        retrieval_mode: str = "hybrid",
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        enable_reranking: bool = False,
        lancedb_index_dir: str = "lancedb",
        load_if_exists: bool = True,
        language: str = "english",
        classes: Optional[List[Any]] = None,
        score_threshold: Optional[float] = None,
    ):
        self.workflow = workflow
        self.dataset = dataset or (
            getattr(workflow, "dataset", None) if workflow is not None else None
        )
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.enable_reranking = enable_reranking
        self.lancedb_index_dir = lancedb_index_dir
        self.load_if_exists = load_if_exists
        self.language = language
        self.classes = classes
        self.score_threshold = score_threshold
        self.embed_model = OpenAIEmbedding(
            model_name=settings.EMBEDDING_MODEL,
            api_base=settings.EMBEDDING_BASE_URL,
            api_key=settings.EMBEDDING_API_KEY,
        )
        self.index_ = None

    def get_params(self, deep=False) -> dict:
        """Override get_params to avoid serializing the datasets."""
        return {
            "classes": self.classes,
            "dense_weight": self.dense_weight,
            "bm25_weight": self.bm25_weight,
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "enable_reranking": self.enable_reranking,
        }

    def fit(self, X: Any = None, y: Any = None) -> "LlamaindexModel":
        """Fit the estimator by initializing the workflow and building or loading the index.

        Args:
            X: Optional InformationRetrievalDataset or DataFrame.
            y: Ignored.
        """
        if isinstance(X, InformationRetrievalDataset):
            self.dataset_ = X
        elif self.dataset is not None:
            self.dataset_ = self.dataset
        elif (
            self.workflow is not None
            and getattr(self.workflow, "dataset", None) is not None
        ):
            self.dataset_ = self.workflow.dataset
        else:
            self.dataset_ = None

        if self.workflow is None:
            if self.dataset_ is None:
                raise ValueError(
                    "A dataset must be provided either in __init__ or to fit()."
                )
            self.workflow = RetrievalWorkflow(
                dataset=self.dataset_,
                load_if_exists=self.load_if_exists,
                lancedb_index_dir=self.lancedb_index_dir,
                top_k=self.top_k,
                language=self.language,
                retrieval_mode=self.retrieval_mode,
                dense_weight=self.dense_weight,
                bm25_weight=self.bm25_weight,
                enable_reranking=self.enable_reranking,
            )

        # Ingest or load existing index
        if self.workflow._index is None:
            run_async(lambda: self.workflow.run(ingest=True))

        raw_classes = self.classes if self.classes is not None else [0, 1]
        self.classes_ = np.array(raw_classes)

        # Build topic lookup mapping
        self.topic_map_: dict[str, str] = {}
        if self.dataset_ is not None:
            try:
                topics = self.dataset_.get_topics()
                qid_col = (
                    "qid"
                    if "qid" in topics.columns
                    else "query_id"
                    if "query_id" in topics.columns
                    else None
                )
                q_col = (
                    "query"
                    if "query" in topics.columns
                    else "title"
                    if "title" in topics.columns
                    else "text"
                    if "text" in topics.columns
                    else None
                )
                if qid_col and q_col:
                    self.topic_map_ = {
                        str(row[qid_col]): str(row[q_col])
                        for _, row in topics.iterrows()
                    }
            except Exception as e:
                LOGGER.debug("Failed to build topic map from dataset: %s", e)

        self._retrieval_cache: dict[str, list[RetrievalResult]] = {}
        self.is_fitted_ = True
        return self

    def _ensure_fitted(self, X: Any = None) -> None:
        if not getattr(self, "is_fitted_", False):
            self.fit(X=X)

    def _retrieve_for_query(self, query_str: str) -> list[RetrievalResult]:
        if not query_str:
            return []
        if query_str in self._retrieval_cache:
            return self._retrieval_cache[query_str]
        results = run_async(
            lambda: self.workflow.run(query=query_str)
        )
        if results is None:
            results = []
        self._retrieval_cache[query_str] = results
        return results

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Synchronously retrieve candidate results for a query."""
        self._ensure_fitted()
        return self._retrieve_for_query(query)

    async def aretrieve(self, query: str) -> list[RetrievalResult]:
        """Asynchronously retrieve candidate results for a query."""
        self._ensure_fitted()
        if not query:
            return []
        if query in self._retrieval_cache:
            return self._retrieval_cache[query]
        results = await self.workflow.run(query=query)
        if results is None:
            results = []
        self._retrieval_cache[query] = results
        return results

    def get_embedding(self, text: str) -> np.ndarray:
        """Compute an embedding vector for a given text using the model's embedding model."""
        return np.array(self.embed_model.get_text_embedding(text))

    def embed(self, text: str) -> np.ndarray:
        """Compute an embedding vector for a given text using the model's embedding model."""
        return self.get_embedding(text)

    @staticmethod
    def _score_to_proba(score: float) -> tuple[float, float]:
        """Convert a retrieval or similarity score into [P(class 0), P(class 1)]."""
        if pd.isna(score) or np.isnan(score):
            return 1.0, 0.0
        if 0.0 <= score <= 1.0:
            p1 = float(score)
        elif score < 0.0:
            p1 = 0.0
        else:
            # Handle unbounded positive scores (e.g., BM25 scores > 1.0)
            p1 = float(score / (score + 1.0))
        p1 = float(np.clip(p1, 0.0, 1.0))
        p0 = float(1.0 - p1)
        return p0, p1

    def _extract_query_and_doc(self, item: Any) -> tuple[str, str, str]:
        """Extract (query_text, doc_id, doc_text) from a row, dict, or formatted string."""
        if isinstance(item, pd.Series):
            item = item.to_dict()

        if isinstance(item, dict):
            qid = str(item.get("query_id") or item.get("qid") or "")
            doc_id = str(item.get("doc_id") or item.get("docno") or "")
            query_text = (
                item.get("query")
                or item.get("query_text")
                or self.topic_map_.get(qid, "")
            )
            doc_text = str(
                item.get("document") or item.get("doc_text") or item.get("body") or ""
            )
            if "text" in item and "<query>" in str(item["text"]):
                q_parsed, d_parsed = InformationRetrievalDataset.parse_pair(
                    str(item["text"])
                )
                query_text = query_text or q_parsed
                doc_text = doc_text or d_parsed
            return str(query_text), doc_id, doc_text

        if isinstance(item, str):
            if "<query>" in item:
                q, d = InformationRetrievalDataset.parse_pair(item)
                return q, "", d
            return item, "", ""

        return "", "", ""

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probabilities [P(class 0), P(class 1)] for query-doc pairs in X."""
        self._ensure_fitted()

        if isinstance(X, InformationRetrievalDataset):
            if hasattr(X, "probabilities") and X.probabilities is not None:
                return np.array(X.probabilities, dtype=float)
            X = X.metadata

        # If X is a DataFrame with precomputed cosine similarity (e.g. from to_tabular())
        assert isinstance(X, pd.DataFrame), f"X must be a DataFrame, got {type(X)}"

        if "cosine_sim" in X.columns and not (
            "query_id" in X.columns or "qid" in X.columns
        ):
            sims = np.clip(X["cosine_sim"].to_numpy(dtype=float), 0.0, 1.0)
            return np.column_stack([1.0 - sims, sims])

        # Optimized batch retrieval for DataFrames with query_id/qid and doc_id/docno
        qid_col = (
            "query_id"
            if "query_id" in X.columns
            else "qid"
            if "qid" in X.columns
            else None
        )
        doc_col = (
            "doc_id"
            if "doc_id" in X.columns
            else "docno"
            if "docno" in X.columns
            else None
        )

        if qid_col and doc_col:
            unique_qids = X[qid_col].astype(str).unique()
            qid_doc_scores: dict[str, dict[str, float]] = {}

            for qid in unique_qids:
                query_str = self.topic_map_.get(qid, "")
                if not query_str and "query" in X.columns:
                    match = X[X[qid_col].astype(str) == qid]
                    if len(match) > 0 and pd.notna(match.iloc[0].get("query")):
                        query_str = str(match.iloc[0]["query"])
                results = self._retrieve_for_query(query_str)
                qid_doc_scores[qid] = {str(res.doc_id): res.score for res in results}

            probas = []
            for _, row in X.iterrows():
                qid_val = str(row[qid_col])
                doc_val = str(row[doc_col])
                doc_scores = qid_doc_scores.get(qid_val, {})
                if doc_val in doc_scores:
                    p0, p1 = self._score_to_proba(doc_scores[doc_val])
                else:
                    p0, p1 = 1.0, 0.0
                probas.append([p0, p1])

            return np.array(probas, dtype=float)

        items = [row for _, row in X.iterrows()]

        # For text pairs, lists of strings, or custom dicts
        embed_cache: dict[str, np.ndarray] = {}

        def _get_cached_embedding(text: str) -> np.ndarray:
            if text not in embed_cache:
                embed_cache[text] = self.get_embedding(text)
            return embed_cache[text]

        probas = []
        for item in items:
            query_str, doc_id, doc_text = self._extract_query_and_doc(item)

            if doc_id:
                # We have a doc_id: check retrieval results for query
                results = self._retrieve_for_query(query_str)
                doc_scores = {str(res.doc_id): res.score for res in results}
                if doc_id in doc_scores:
                    p0, p1 = self._score_to_proba(doc_scores[doc_id])
                else:
                    p0, p1 = 1.0, 0.0
            elif doc_text and query_str:
                # Text-based query-doc pair: compute pairwise cosine similarity
                q_emb = _get_cached_embedding(query_str)
                d_emb = _get_cached_embedding(doc_text)
                norm_q = np.linalg.norm(q_emb)
                norm_d = np.linalg.norm(d_emb)
                if norm_q > 0 and norm_d > 0:
                    sim = float(np.dot(q_emb, d_emb) / (norm_q * norm_d))
                else:
                    sim = 0.0
                p0, p1 = self._score_to_proba(sim)
            elif query_str:
                # Standalone query without document: return top-1 retrieval confidence
                results = self._retrieve_for_query(query_str)
                if len(results) > 0:
                    p0, p1 = self._score_to_proba(results[0].score)
                else:
                    p0, p1 = 1.0, 0.0
            else:
                p0, p1 = 1.0, 0.0

            probas.append([p0, p1])

        return np.array(probas, dtype=float)

    def predict(self, X: Any) -> np.ndarray:
        """Predict relevance for query-doc pairs in X."""
        probas = self.predict_proba(X)
        if self.score_threshold is not None:
            pred_indices = (probas[:, 1] >= self.score_threshold).astype(int)
        else:
            pred_indices = np.argmax(probas, axis=1)

        return self.classes_[pred_indices]

    def retrieve_dataframe(
        self,
        dataset: InformationRetrievalDataset,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run retrieval across dataset topics and return a formatted DataFrame.

        The resulting DataFrame contains ['query_id', 'doc_id', 'score', 'rank', 'relevance']
        columns, ready to be passed to dataset.set_predictions().
        """
        self._ensure_fitted()

        if not isinstance(dataset, InformationRetrievalDataset):
            raise ValueError("An InformationRetrievalDataset must be provided.")

        topics = dataset.get_topics()
        qid_col = "qid" if "qid" in topics.columns else "query_id"
        q_col = (
            "query"
            if "query" in topics.columns
            else "title"
            if "title" in topics.columns
            else "text"
        )
        k = top_k or self.top_k
        rows = []

        for _, row in topics.iterrows():
            qid = str(row[qid_col])
            query_str = str(row[q_col])
            results = self._retrieve_for_query(query_str)

            for rank, res in enumerate(results[:k], start=1):
                p0, p1 = self._score_to_proba(res.score)
                rows.append(
                    {
                        "query_id": qid,
                        "doc_id": str(res.doc_id),
                        "score": [p0, p1],
                        "rank": rank,
                        "relevance": 1,
                    }
                )

        return pd.DataFrame(rows)
