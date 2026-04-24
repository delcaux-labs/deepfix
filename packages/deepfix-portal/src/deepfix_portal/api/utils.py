"""
Utility functions for authentication, password hashing, etc.
"""

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import settings

# Use SHA256 truncation for passwords longer than 72 bytes (bcrypt limit)
pwd_context = CryptContext(
    schemes=["bcrypt"], deprecated="auto", bcrypt__truncate_error=True
)


def get_secret_key() -> str:
    """
    Get the secret key from settings
    """
    return settings.DEEPFIX_PORTAL_SECRET_KEY


def get_algorithm() -> str:
    """
    Get the algorithm from settings
    """
    return settings.DEEPFIX_PORTAL_ALGORITHM


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt
    For passwords longer than 72 bytes, pre-hash with SHA256
    """
    # Bcrypt has a 72-byte limit, so pre-hash long passwords
    if len(password.encode("utf-8")) > 72:
        password = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash
    For passwords longer than 72 bytes, pre-hash with SHA256
    """
    # Bcrypt has a 72-byte limit, so pre-hash long passwords
    if len(plain_password.encode("utf-8")) > 72:
        plain_password = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, get_secret_key(), algorithm=get_algorithm())
    return encoded_jwt


def verify_token(token: str) -> Optional[str]:
    """
    Verify and decode a JWT token
    Returns user ID if valid, None otherwise
    """

    try:
        payload = jwt.decode(token, get_secret_key(), algorithms=[get_algorithm()])
        user_id: str = payload.get("sub")
        return user_id
    except JWTError:
        return None


def generate_verification_token() -> str:
    """
    Generate a secure random token for email verification
    Returns a URL-safe random token
    """
    return secrets.token_urlsafe(32)


def create_verification_token(user_id: str) -> str:
    """
    Create a JWT token for email verification
    Token expires in 24 hours
    Returns the encoded JWT token
    """
    expire = datetime.now(timezone.utc) + timedelta(hours=24)
    to_encode = {
        "sub": user_id,
        "type": "email_verification",
        "exp": expire,
    }
    encoded_jwt = jwt.encode(to_encode, get_secret_key(), algorithm=get_algorithm())
    return encoded_jwt


def verify_verification_token(token: str) -> Optional[str]:
    """
    Verify and decode an email verification token
    Returns user ID if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, get_secret_key(), algorithms=[get_algorithm()])
        token_type = payload.get("type")
        if token_type != "email_verification":
            return None
        user_id: str = payload.get("sub")
        return user_id
    except JWTError:
        return None
