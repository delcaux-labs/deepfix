"""
SQLAlchemy database models
"""

import uuid

from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, text
from sqlalchemy.sql import func

from .database import Base

class RequestLog(Base):
    """Request/Response log from deepfix-server.

    Stores successful API requests with user identity, request/response payloads,
    and timing information.

    Attributes:
        id: Unique identifier for the log entry.
        user_id: ID of the user who made the request.
        user_email: Email of the user who made the request.
        endpoint: API endpoint that was called.
        request_json: JSON-serialized request payload.
        response_json: JSON-serialized response payload.
        status_code: HTTP status code of the response.
        duration_ms: Request duration in milliseconds.
        created_at: Timestamp when the log entry was created.
    """

    __tablename__ = "request_logs"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(
            f"req_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4()}"
        ),
    )
    user_id = Column(String, nullable=False, index=True)
    user_email = Column(String, nullable=False)
    endpoint = Column(String, nullable=False)
    request_json = Column(Text, nullable=True)
    response_json = Column(Text, nullable=True)
    response_summary = Column(Text, nullable=True)
    status_code = Column(Integer, nullable=True)
    duration_ms = Column(Float, nullable=True)
    remote_job_id = Column(String, nullable=True, index=True)
    job_status = Column(String, default="PENDING")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))


class User(Base):
    """User model"""

    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(Text, unique=True, nullable=False, index=True)
    email = Column(Text, unique=True, nullable=False, index=True)
    password = Column(Text, nullable=False)  # Should be hashed
    name = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    email_verification_token = Column(Text, nullable=True)
    email_verification_token_expires = Column(DateTime(timezone=True), nullable=True)
    password_reset_token = Column(Text, nullable=True)
    password_reset_token_expires = Column(DateTime(timezone=True), nullable=True)


class APIKey(Base):
    """API Key model"""

    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)  # Foreign key to users.id
    key = Column(String, unique=True, nullable=False, index=True)  # The actual API key
    name = Column(Text, nullable=True)  # Optional name for the key
    last_used = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)
