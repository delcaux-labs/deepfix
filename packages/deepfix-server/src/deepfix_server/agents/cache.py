"""
Database-backed cache implementation for LLM calls.

This module provides a standalone cache (no longer dependent on dspy.clients.Cache)
that stores LLM request/response pairs in the database with hit count tracking.
"""

import copy
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import hashlib

LOGGER = logging.getLogger(__name__)

# Lazily initialized when a database engine is available
_SessionLocal = None
_Base = None


def _get_base():
    global _Base
    if _Base is None:
        _Base = declarative_base()
    return _Base


def _get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False)
    return _SessionLocal


class Cache(_get_base()):
    """LLM cache model for database-backed caching"""

    __tablename__ = "llm_request_cache"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    cache_key = Column(String, unique=True, nullable=False, index=True)  # SHA256 hash
    model_name = Column(String, nullable=False, index=True)  # LLM model identifier
    request_json = Column(Text, nullable=False)  # JSON serialized request
    response_json = Column(Text, nullable=False)  # JSON serialized response
    hit_count = Column(Integer, default=0, nullable=False)  # Number of cache hits
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DatabaseCache:
    """Database-backed cache for LLM calls (formerly DSPy cache).

    This cache stores LLM request/response pairs in a PostgreSQL database,
    keyed by a hash of the messages and model name. It tracks cache hit counts
    for analytics and uses database transactions for thread safety.

    No longer subclasses dspy.clients.Cache — now standalone.
    """

    def __init__(self, **kwargs):  # pylint: disable=unused-argument
        """Initialize the database cache.

        Args:
            **kwargs: Additional arguments (for backward compatibility).
        """
        LOGGER.info("Initialized LLM database cache")

    @staticmethod
    def cache_key(request: Dict[str, Any], ignored_args: Optional[list[str]] = None) -> str:
        """Generate a cache key from a request dictionary.

        Uses SHA256 hash of messages + model name, ignoring specified args.
        """
        key_dict = {k: v for k, v in request.items() if k not in (ignored_args or [])}
        key_str = json.dumps(key_dict, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(
        self,
        request: Dict[str, Any],
        ignored_args_for_cache_key: Optional[list[str]] = None,
    ) -> Optional[Any]:
        """Retrieve cached response for a request.

        If a cache entry exists:
        - Increments the hit_count
        - Updates the updated_at timestamp
        - Returns the cached response

        Args:
            request: The LLM request dictionary.
            ignored_args_for_cache_key: Optional list of keys to ignore (not used).

        Returns:
            Cached response object if found, None otherwise.
        """
        try:
            key = self.cache_key(request, ignored_args_for_cache_key)
        except Exception:
            LOGGER.debug(f"Failed to generate cache key for request: {request}")
            return None

        db = _get_session_local()()

        try:
            # Query for cache entry
            cache_entry = db.query(Cache).filter(Cache.cache_key == key).first()

            if cache_entry:
                # Cache hit - increment counter and update timestamp
                cache_entry.hit_count += 1
                cache_entry.updated_at = datetime.now(timezone.utc)
                db.commit()

                # Deserialize and return cached response
                response = json.loads(cache_entry.response_json)
                LOGGER.debug(
                    "Cache HIT for model=%s, hit_count=%s",
                    cache_entry.model_name,
                    cache_entry.hit_count,
                )
                if hasattr(response, "usage"):
                    response.usage = {}
                    response.cache_hit = True
                return response

            # Cache miss
            LOGGER.debug("Cache MISS for key=%s...", key[:16])
            return None

        except Exception as exc:  # pylint: disable=broad-except
            db.rollback()
            LOGGER.warning("Error reading from cache: %s", exc)
            return None
        finally:
            db.close()

    def put(
        self,
        request: Dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: Optional[list[str]] = None,
        enable_memory_cache: bool = True,  # Kept for compatibility
    ) -> None:
        """Store a response in the cache.

        Creates a new cache entry with:
        - cache_key: SHA256 hash of messages + model
        - model_name: Extracted from request
        - request_json: Serialized request
        - response_json: Serialized response
        - hit_count: Initialized to 0

        Args:
            request: The LLM request dictionary.
            value: The response to cache.
            ignored_args_for_cache_key: Optional list of keys to ignore (not used).
            enable_memory_cache: Ignored (kept for compatibility).
        """
        key = self.cache_key(request, ignored_args_for_cache_key)
        model_name = request.get("model", "unknown")

        db = _get_session_local()()
        try:
            # Check if entry already exists (race condition protection)
            existing = db.query(Cache).filter(Cache.cache_key == key).first()
            if existing:
                # Entry already exists, no need to insert again
                LOGGER.debug("Cache entry already exists for key=%s...", key[:16])
                return

            # Serialize request and response
            request_json = json.dumps(request, default=str)
            response_json = json.dumps(value, default=str)

            # Create new cache entry
            cache_entry = Cache(
                cache_key=key,
                model_name=model_name,
                request_json=request_json,
                response_json=response_json,
                hit_count=0,
            )

            db.add(cache_entry)
            db.commit()

            LOGGER.debug(
                "Cached response for model=%s, key=%s...", model_name, key[:16]
            )

        except IntegrityError:
            # Race condition: another thread inserted the same key
            db.rollback()
            LOGGER.debug("Cache entry race condition for key=%s...", key[:16])
        except Exception as exc:  # pylint: disable=broad-except
            db.rollback()
            LOGGER.warning("Error writing to cache: %s", exc)
        finally:
            db.close()
