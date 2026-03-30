"""
Artifact analysis routes.

This module provides the API endpoint for analyzing ML artifacts,
replicating the functionality of LitServe-based deepfix-server.
"""

import json
import logging
import os
import time
import traceback
from typing import Any, Optional

import httpx
from deepfix_core.models import APIRequest, APIResponse, DatasetArtifacts
from fastapi import APIRouter, Depends, HTTPException, Request
from langfuse import get_client, observe
from sqlalchemy.orm import Session

from ..database import get_db
from ..dependencies import get_api_key_user
from ..models import RequestLog
from ..schemas import APIKeyValidationResponse

router = APIRouter()

LOGGER = logging.getLogger(__name__)

DEEPFIX_SERVER_URL = os.getenv("DEEPFIX_SERVER_URL", "http://localhost:4141/api/v1/analyse")


def _serialize_to_json(obj: Any) -> Optional[str]:
    """Serialize an object to JSON string.

    Args:
        obj: Object to serialize.

    Returns:
        JSON string representation, or None if serialization fails.
    """
    if obj is None:
        return None

    try:
        # Handle Pydantic models
        if hasattr(obj, "model_dump"):
            return json.dumps(obj.model_dump(), default=str)
        elif hasattr(obj, "dict"):
            return json.dumps(obj.dict(), default=str)
        # Handle dicts and other JSON-serializable objects
        return json.dumps(obj, default=str)
    except Exception as exc:
        LOGGER.warning(f"Failed to serialize object to JSON: {exc}")
        return None


def _decode_request(request: APIRequest) -> APIRequest:
    """Decode API request into APIRequest validating artifacts.

    Args:
        request: APIRequest containing artifacts and configuration.

    Returns:
        request: Validated APIRequest.

    Raises:
        HTTPException: If request decoding fails (status 400).
    """
    try:
        dataset_artifacts = request.dataset_artifacts
        if isinstance(request.dataset_artifacts, dict):
            dataset_artifacts = DatasetArtifacts.from_dict(request.dataset_artifacts)
        elif request.dataset_artifacts is not None and not isinstance(
            request.dataset_artifacts, DatasetArtifacts
        ):
            raise ValueError("Dataset artifacts must be a DatasetArtifacts object")

        return request
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Error decoding request: {exc}",
        ) from exc


async def _log_request(
    db: Session,
    current_user: APIKeyValidationResponse,
    endpoint: str,
    request: APIRequest,
    response: APIResponse,
    status_code: int,
    duration_ms: float,
) -> None:
    """Log the request to the database.

    Args:
        db: Database session.
        current_user: Current user information.
        endpoint: API endpoint that was called.
        request: The API request.
        response: The API response.
        status_code: HTTP status code.
        duration_ms: Request duration in milliseconds.
    """
    try:
        request_json = _serialize_to_json(request)
        response_json = _serialize_to_json(response)

        log_entry = RequestLog(
            user_id=current_user.user_id,
            user_email=current_user.user_email,
            endpoint=endpoint,
            request_json=request_json,
            response_json=response_json,
            status_code=status_code,
            duration_ms=duration_ms,
        )
        db.add(log_entry)
        db.commit()

        LOGGER.info(
            f"Logged request for user {current_user.user_email} "
            f"to {endpoint} ({duration_ms:.2f}ms)"
        )
    except Exception as exc:
        db.rollback()
        LOGGER.exception(f"Failed to log request/response: {exc}")


@router.post("/analyse", response_model=APIResponse)
@observe()
async def analyse_artifacts(
    request: APIRequest,
    current_user: APIKeyValidationResponse = Depends(get_api_key_user),
    db: Session = Depends(get_db),
) -> APIResponse:
    """Analyze ML artifacts and return diagnostic results.

    This endpoint accepts dataset, training, deepchecks, and model checkpoint
    artifacts, runs analysis through specialized agents, and returns findings
    and recommendations.

    Args:
        request: APIRequest containing artifacts to analyze.
        current_user: Authenticated user from API key validation.
        db: Database session for logging.

    Returns:
        APIResponse with analysis results from all agents.

    Raises:
        HTTPException: If authentication fails (401/403) or analysis fails (500).
    """
    start_time = time.perf_counter()
    endpoint = "/analyse"

    try:
        # 1. Decode request
        _decode_request(request)

        # 2. Forward request to deepfix-server
        async with httpx.AsyncClient() as client:
            server_response = await client.post(
                DEEPFIX_SERVER_URL,
                json=request.model_dump(),
                timeout=300.0,
            )

            if server_response.status_code != 200:
                raise HTTPException(
                    status_code=server_response.status_code,
                    detail=f"Error from analysis server: {server_response.text}"
                )

            out = server_response.json()
            response = APIResponse(**out)

        # 3. Log successful request
        duration_ms = (time.perf_counter() - start_time) * 1000
        await _log_request(
            db=db,
            current_user=current_user,
            endpoint=endpoint,
            request=request,
            response=response,
            status_code=200,
            duration_ms=duration_ms,
        )

        return response

    except HTTPException as exc:
        # Re-raise HTTP exceptions (auth failures, bad requests)
        raise exc
    except Exception as exc:
        LOGGER.exception(f"Analysis failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail=traceback.format_exc(),
        ) from exc


@router.get("/health")
async def analysis_health():
    """Health check endpoint for the analysis service."""
    return {
        "status": "ok",
        "service": "analysis",
        "server_url": DEEPFIX_SERVER_URL,
    }
