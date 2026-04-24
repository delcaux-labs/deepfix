import uuid
from datetime import datetime

from deepfix_core.models import AnalysisJobStatus
from sqlalchemy import Column, DateTime, Enum, String, Text

from .database import Base


class AnalysisJob(Base):
    """Model to track background analysis jobs."""

    __tablename__ = "analysis_jobs"

    id = Column(
        String,
        primary_key=True,
        default=lambda: (
            f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        ),
    )
    status = Column(
        Enum(AnalysisJobStatus), nullable=False, default=AnalysisJobStatus.PENDING
    )
    request_data = Column(Text, nullable=True)
    result_data = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
