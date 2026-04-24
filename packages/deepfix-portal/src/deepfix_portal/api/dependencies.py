"""
FastAPI dependencies for authentication, etc.
"""

import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Optional, Tuple

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordBearer,
)
from sqlalchemy.orm import Session

from .database import SessionLocal, get_db
from .models import APIKey, User
from .schemas import APIKeyValidationResponse
from .utils import verify_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
api_key_scheme = HTTPBearer()

# Lightweight TTL bucket helper for lru_cache-backed lookups
USER_CACHE_TTL_SECONDS = 60
API_KEY_CACHE_TTL_SECONDS = 300


def _user_cache_bucket() -> int:
    return int(time.time() / USER_CACHE_TTL_SECONDS)


def _api_key_cache_bucket() -> int:
    return int(time.time() / API_KEY_CACHE_TTL_SECONDS)


@lru_cache(maxsize=256)
def _cached_user_lookup(user_id: str, _bucket: int) -> Optional[User]:
    """
    Cached user fetch to avoid repeated DB hits on hot paths.
    The TTL bucket parameter ensures periodic refresh.
    """
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        return user
    finally:
        db.close()


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user_id = verify_token(token)
    if user_id is None:
        raise credentials_exception

    user = _cached_user_lookup(user_id, _user_cache_bucket())
    if user is None:
        raise credentials_exception

    return user


async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency to ensure the current user is an admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )
    return current_user


async def verify_service_token(
    x_service_token: Optional[str] = Header(default=None),
) -> None:
    """
    Verify the shared service token for server-to-server requests.
    """
    if not settings.DEEPFIX_PORTAL_SERVICE_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service token is not configured",
        )
    if (
        not x_service_token
        or x_service_token != settings.DEEPFIX_PORTAL_SERVICE_TOKEN
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service token",
        )


@lru_cache(maxsize=256)
def _cached_api_key_lookup(
    key: str, _bucket: int
) -> Optional[Tuple[str, str, bool, str, str, Optional[str], bool]]:
    """
    Cached API key lookup to reduce DB hits on validation path.
    Returns tuple of (key_id, key_name, key_is_active, user_id, user_email, user_name, user_is_active).
    """
    db = SessionLocal()
    try:
        api_key = db.query(APIKey).filter(APIKey.key == key).first()
        if not api_key:
            return None

        user = db.query(User).filter(User.id == api_key.user_id).first()
        if not user:
            return None

        return (
            api_key.id,
            api_key.name,
            api_key.is_active,
            api_key.user_id,
            user.email,
            user.name,
            user.is_active,
        )
    finally:
        db.close()


async def get_api_key_user(
    auth: HTTPAuthorizationCredentials = Depends(api_key_scheme),
    db: Session = Depends(get_db),
) -> APIKeyValidationResponse:
    """
    Dependency to validate Bearer token API key and return user info.
    Used for SDK/API authentication via API keys.
    """
    if auth.scheme.lower() != "bearer" or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    cached = _cached_api_key_lookup(auth.credentials, _api_key_cache_bucket())
    if not cached:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    (
        key_id,
        key_name,
        key_is_active,
        user_id,
        user_email,
        user_name,
        user_is_active,
    ) = cached

    if not key_is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is inactive",
        )
    if not user_is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key owner is inactive or missing",
        )

    # Update last_used on the live DB object
    api_key = db.query(APIKey).filter(APIKey.id == key_id).first()
    if api_key:
        api_key.last_used = datetime.now(timezone.utc)
        db.commit()

    return APIKeyValidationResponse(
        key_id=key_id,
        key_name=key_name,
        key_is_active=key_is_active,
        user_id=user_id,
        user_email=user_email,
        user_name=user_name,
        user_is_active=user_is_active,
    )
