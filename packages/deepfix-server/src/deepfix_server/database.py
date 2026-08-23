"""
Database configuration and session management for deepfix-server.
"""

from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# Global engine and session factory
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None

# Base class for models
Base = declarative_base()

def ensure_schema_compatibility(engine: Engine) -> None:
    """Ensure SQLite tables have all columns added in newer models."""
    try:
        with engine.connect() as conn:
            # Check fix_jobs table
            result = conn.exec_driver_sql("PRAGMA table_info(fix_jobs)")
            columns = {row[1] for row in result.fetchall()}
            if columns:
                expected_columns = {
                    "dataset_uri": "TEXT",
                    "model_uri": "TEXT",
                    "phase": "VARCHAR",
                    "events_data": "TEXT",
                    "intermediate_metrics_data": "TEXT",
                }
                for col_name, col_type in expected_columns.items():
                    if col_name not in columns:
                        conn.exec_driver_sql(f"ALTER TABLE fix_jobs ADD COLUMN {col_name} {col_type}")
                conn.commit()
    except Exception:
        pass


def init_database(database_url: str, database_echo: bool = False) -> None:
    """Initialize the database engine and session factory.

    Args:
        database_url: Database connection URL (SQLAlchemy format).
        database_echo: Whether to echo SQL statements for debugging.
    """
    global _engine, _SessionLocal
    _engine = create_engine(database_url, echo=database_echo)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    Base.metadata.create_all(bind=_engine)
    ensure_schema_compatibility(_engine)


def get_engine() -> Optional[Engine]:
    """Get the database engine.

    Returns:
        The SQLAlchemy engine, or None if not initialized.
    """
    return _engine


def get_session() -> Session:
    """Create a new database session.

    Returns:
        A new SQLAlchemy session.

    Raises:
        RuntimeError: If the database has not been initialized.
    """
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _SessionLocal()


def get_db():
    """Dependency to get database session.

    Yields:
        A database session that will be closed after use.

    Raises:
        RuntimeError: If the database has not been initialized.
    """
    db = get_session()
    try:
        yield db
    finally:
        db.close()
