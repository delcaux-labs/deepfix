"""
Database configuration and session management
"""

import os

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .config import settings

# Enable defensive checks for stale/broken connections (e.g., Postgres idle
# timeouts). This prevents "Software caused connection abort" errors from
# bubbling up as 500s on first use after an idle period.
POOL_PRE_PING_DEFAULT = True
POOL_RECYCLE_SECONDS_DEFAULT = 1800  # 30 minutes


def _build_engine():
    is_sqlite = settings.DATABASE_URL.startswith("sqlite")

    # SQLite requires a special flag for multithreading; other engines do not
    connect_args = {"check_same_thread": False} if is_sqlite else {}

    # For networked databases, add pool guards to avoid stale connections
    pool_kwargs = {}
    if not is_sqlite:
        pool_kwargs["pool_pre_ping"] = POOL_PRE_PING_DEFAULT
        pool_kwargs["pool_recycle"] = settings.DB_POOL_RECYCLE

    return create_engine(
        settings.DATABASE_URL,
        echo=False,
        connect_args=connect_args,
        **pool_kwargs,
    )


# Create engine with safe defaults
engine = _build_engine()

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """
    Dependency to get database session
    Use this in route handlers: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    except OperationalError as exc:
        db.rollback()
        raise HTTPException(
            status_code=503, detail="Database connection lost. Please retry."
        ) from exc
    finally:
        try:
            db.close()
        except OperationalError:
            # Connection is already dead; nothing more to do here
            pass
