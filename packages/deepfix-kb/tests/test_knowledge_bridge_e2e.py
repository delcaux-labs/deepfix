"""End-to-end integration tests for KnowledgeBridge.

These tests verify the complete integration of all KnowledgeBridge components:
- TavilySearchRetriever (web search)
- PerplexitySonarRetriever (AI research)
- HybridRetriever (multi-source orchestration)
- KnowledgeBridge (main interface)
- DSPy tools

Note: These tests require API keys to be configured:
- TAVILY_API_KEY for web search
- OPENROUTER_API_KEY for Perplexity Sonar

Run with: pytest tests/test_knowledge_bridge_e2e.py -v
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepfix_kb import KnowledgeBridge
from deepfix_kb.config import KnowledgeBridgeConfig, PerplexityConfig, TavilyConfig
from deepfix_kb.retrieval import (
    BaseRetriever,
    HybridRetriever,
    PerplexityAPIError,
    PerplexityConfigError,
    PerplexityError,
    PerplexitySonarRetriever,
    RetrievalResult,
    RetrievalStrategy,
    TavilySearchRetriever,
)
from deepfix_kb.tools import create_knowledge_tools


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def mock_tavily_response():
    """Mock Tavily API response."""
    return {
        "results": [
            {
                "title": "PyTorch Learning Rate Scheduling",
                "url": "https://pytorch.org/docs/stable/optim.html",
                "content": "Use cosine annealing or step decay for fine-tuning transformers.",
                "score": 0.95,
            },
            {
                "title": "Hugging Face Training Tips",
                "url": "https://huggingface.co/docs/transformers/training",
                "content": "Start with a small learning rate like 2e-5 for fine-tuning.",
                "score": 0.89,
            },
        ],
        "answer": "For fine-tuning, use learning rates between 1e-5 and 5e-5.",
    }


@pytest.fixture
def mock_perplexity_response():
    """Mock Perplexity DSPy response."""
    mock_prediction = MagicMock()
    mock_prediction.answer = (
        "For training vision transformers (ViT), best practices include:\n\n"
        "1. Use a learning rate of 1e-4 to 3e-4 with cosine decay\n"
        "2. Apply strong data augmentation (RandAugment, MixUp)\n"
        "3. Use AdamW optimizer with weight decay 0.05-0.1\n\n"
        "Source: https://arxiv.org/abs/2010.11929"
    )
    return mock_prediction


@pytest.fixture
def mock_retrieval_result():
    """Create a mock RetrievalResult."""
    return RetrievalResult(
        content="Mock content for testing",
        source="Mock Source",
        source_type="web",
        url="https://example.com",
        relevance_score=0.9,
    )


# ============================================================================
# Configuration Tests
# ============================================================================
class TestKnowledgeBridgeConfig:
    """Tests for configuration models."""

    def test_config_from_env_defaults(self):
        """Test configuration creation with defaults."""
        config = KnowledgeBridgeConfig()

        assert config.default_strategy == "parallel"
        assert config.max_results_per_source == 3
        assert config.max_total_results == 10
        assert config.enable_local_kb is False

    def test_tavily_config_defaults(self):
        """Test Tavily configuration defaults."""
        config = TavilyConfig()

        assert config.search_depth == "basic"
        assert config.default_topic == "general"
        assert config.include_answer is True

    def test_perplexity_config_defaults(self):
        """Test Perplexity configuration defaults."""
        config = PerplexityConfig()

        assert config.model == "sonar"
        assert config.temperature == 0.7
        assert config.max_tokens is None

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("PERPLEXITY_MODEL", "sonar-pro")
        monkeypatch.setenv("KNOWLEDGE_BRIDGE_STRATEGY", "cascading")
        monkeypatch.setenv("KNOWLEDGE_BRIDGE_ENABLE_LOCAL_KB", "true")

        config = KnowledgeBridgeConfig()

        assert config.perplexity.model == "sonar-pro"
        assert config.default_strategy == "cascading"
        assert config.enable_local_kb is True


# ============================================================================
# Retriever Tests
# ============================================================================
class TestTavilySearchRetriever:
    """Tests for Tavily Search retriever."""

    def test_retriever_initialization(self):
        """Test retriever initializes correctly."""
        retriever = TavilySearchRetriever(api_key="test-key")

        assert retriever.source_type == "web"
        assert retriever.is_available is True

    def test_retriever_not_available_without_key(self):
        """Test retriever reports unavailable without API key."""
        with patch.dict(os.environ, {}, clear=True):
            retriever = TavilySearchRetriever(api_key=None)
            # Force clear any cached env var
            retriever._api_key = None
            assert retriever.is_available is False

    @pytest.mark.asyncio
    async def test_retrieve_with_mock(self, mock_tavily_response):
        """Test retrieval with mocked API response."""
        retriever = TavilySearchRetriever(api_key="test-key")

        with patch.object(retriever, "client") as mock_client:
            mock_client.search = AsyncMock(return_value=mock_tavily_response)

            results = await retriever.retrieve(
                "learning rate for fine-tuning", max_results=2
            )

            assert len(results) == 2
            assert results[0].source_type == "web"
            assert "pytorch" in results[0].url.lower()
            assert results[0].relevance_score == 0.95


class TestPerplexitySonarRetriever:
    """Tests for Perplexity Sonar retriever."""

    def test_retriever_initialization(self):
        """Test retriever initializes correctly."""
        retriever = PerplexitySonarRetriever(api_key="test-key", model="sonar-pro")

        assert retriever.source_type == "perplexity"
        assert retriever.model_name == "sonar-pro"
        assert retriever.is_available is True

    def test_retriever_not_available_without_key(self):
        """Test retriever reports unavailable without API key."""
        with patch.dict(os.environ, {}, clear=True):
            retriever = PerplexitySonarRetriever(api_key=None)
            retriever._api_key = None
            assert retriever.is_available is False

    def test_config_error_on_missing_key(self):
        """Test that configuration error is raised for missing API key."""
        retriever = PerplexitySonarRetriever(api_key=None)
        retriever._api_key = None

        with pytest.raises(ValueError, match="API key not provided"):
            _ = retriever.lm


# ============================================================================
# Hybrid Retriever Tests
# ============================================================================
class TestHybridRetriever:
    """Tests for Hybrid retriever orchestration."""

    @pytest.fixture
    def mock_retrievers(self, mock_retrieval_result):
        """Create mock retrievers."""
        tavily = MagicMock(spec=BaseRetriever)
        tavily.source_type = "web"
        tavily.is_available = True
        tavily.retrieve = AsyncMock(return_value=[mock_retrieval_result])

        perplexity = MagicMock(spec=BaseRetriever)
        perplexity.source_type = "perplexity"
        perplexity.is_available = True
        perplexity.retrieve = AsyncMock(return_value=[mock_retrieval_result])

        return tavily, perplexity

    def test_available_sources(self, mock_retrievers):
        """Test available sources reporting."""
        tavily, perplexity = mock_retrievers
        hybrid = HybridRetriever(tavily=tavily, perplexity=perplexity)

        sources = hybrid.available_sources
        assert "web" in sources
        assert "perplexity" in sources

    @pytest.mark.asyncio
    async def test_parallel_retrieval(self, mock_retrievers):
        """Test parallel retrieval from multiple sources."""
        tavily, perplexity = mock_retrievers
        hybrid = HybridRetriever(tavily=tavily, perplexity=perplexity)

        results = await hybrid.retrieve(
            "test query",
            strategy=RetrievalStrategy.PARALLEL,
        )

        # Should get results from both sources
        assert len(results) >= 1
        tavily.retrieve.assert_called_once()
        perplexity.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_source_filtering(self, mock_retrievers):
        """Test filtering to specific sources."""
        tavily, perplexity = mock_retrievers
        hybrid = HybridRetriever(tavily=tavily, perplexity=perplexity)

        await hybrid.retrieve(
            "test query",
            sources=["web"],
        )

        # Should only query Tavily
        tavily.retrieve.assert_called_once()
        perplexity.retrieve.assert_not_called()


# ============================================================================
# KnowledgeBridge Integration Tests
# ============================================================================
class TestKnowledgeBridgeIntegration:
    """Integration tests for KnowledgeBridge."""

    def test_bridge_initialization(self):
        """Test bridge initializes with retrievers."""
        bridge = KnowledgeBridge(
            tavily_api_key="test-tavily",
            openrouter_api_key="test-openrouter",
        )

        # Check retrievers are initialized
        assert bridge._tavily is not None
        assert bridge._perplexity is not None
        assert bridge._hybrid is not None

    @pytest.mark.asyncio
    async def test_query_with_mock(
        self, mock_tavily_response, mock_perplexity_response
    ):
        """Test full query workflow with mocks."""
        bridge = KnowledgeBridge(
            tavily_api_key="test-tavily",
            openrouter_api_key="test-openrouter",
        )

        # Mock the retrievers
        with (
            patch.object(bridge._tavily, "retrieve") as mock_tavily,
            patch.object(bridge._perplexity, "retrieve") as mock_perplexity,
        ):
            mock_tavily.return_value = [
                RetrievalResult(
                    content="Web result content",
                    source="Web Source",
                    source_type="web",
                    url="https://example.com",
                )
            ]
            mock_perplexity.return_value = [
                RetrievalResult(
                    content="AI research result",
                    source="Perplexity Sonar",
                    source_type="perplexity",
                )
            ]

            response = await bridge.query(
                "How to train vision transformers?",
                sources=["web", "perplexity"],
                synthesize=False,
            )

            assert response.query == "How to train vision transformers?"
            assert len(response.results) >= 1

    @pytest.mark.asyncio
    async def test_search_convenience_method(self):
        """Test search() convenience method."""
        bridge = KnowledgeBridge(
            tavily_api_key="test-tavily",
            openrouter_api_key="test-openrouter",
        )

        with patch.object(bridge._tavily, "retrieve") as mock_retrieve:
            mock_retrieve.return_value = [
                RetrievalResult(
                    content="Search result",
                    source="Search Source",
                    source_type="web",
                )
            ]

            results = await bridge.search("test query")

            assert len(results) == 1
            mock_retrieve.assert_called_once()


# ============================================================================
# Tools Tests
# ============================================================================
class TestTools:
    """Tests for DSPy tool wrappers."""

    def test_create_knowledge_tools(self):
        """Test tool creation."""
        bridge = KnowledgeBridge(
            tavily_api_key="test-key",
            openrouter_api_key="test-key",
        )

        tools = create_knowledge_tools(bridge, use_hybrid=True)

        # Should have 1 tool: hybrid_search
        assert len(tools) == 1

    def test_create_tools_without_hybrid(self):
        """Test tool creation without hybrid tool."""
        bridge = KnowledgeBridge(
            tavily_api_key="test-key",
            openrouter_api_key="test-key",
        )

        tools = create_knowledge_tools(bridge, use_hybrid=False)

        # Should have 3 tools
        assert len(tools) == 3


# ============================================================================
# Error Handling Tests
# ============================================================================
class TestErrorHandling:
    """Tests for error handling."""

    def test_perplexity_error_hierarchy(self):
        """Test exception class hierarchy."""
        assert issubclass(PerplexityConfigError, PerplexityError)
        assert issubclass(PerplexityAPIError, PerplexityError)

    def test_api_error_attributes(self):
        """Test PerplexityAPIError attributes."""
        original = ValueError("Original error")
        error = PerplexityAPIError(
            "API call failed",
            status_code=500,
            original_error=original,
        )

        assert str(error) == "API call failed"
        assert error.status_code == 500
        assert error.original_error is original


# ============================================================================
# Live Integration Tests (require API keys)
# ============================================================================
@pytest.mark.skipif(not os.getenv("TAVILY_API_KEY"), reason="TAVILY_API_KEY not set")
class TestLiveTavilyIntegration:
    """Live integration tests for Tavily (requires API key)."""

    @pytest.mark.asyncio
    async def test_live_web_search(self):
        """Test actual web search."""
        retriever = TavilySearchRetriever()

        results = await retriever.retrieve(
            "PyTorch learning rate scheduler",
            max_results=3,
        )

        assert len(results) > 0
        assert all(r.source_type == "web" for r in results)
        assert all(r.url is not None for r in results)
        print(results)


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set"
)
class TestLivePerplexityIntegration:
    """Live integration tests for Perplexity (requires API key)."""

    @pytest.mark.asyncio
    async def test_live_research(self):
        """Test actual AI research."""
        retriever = PerplexitySonarRetriever(model="sonar")

        result = await retriever.research(
            "Best practices for training vision transformers",
            depth="brief",
        )

        assert result is not None
        assert result.source_type == "perplexity"
        assert len(result.content) > 100
        print(result)
