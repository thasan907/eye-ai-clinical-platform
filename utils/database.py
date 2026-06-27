"""
utils/database.py
SQLAlchemy async database setup + table definitions.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DATABASE_URL

# ── Engine & session ─────────────────────────────────────────────
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


# ── Tables ───────────────────────────────────────────────────────

class ScanRecord(Base):
    """One row per uploaded retinal image scan."""
    __tablename__ = "scans"

    id              = Column(Integer, primary_key=True, index=True)
    patient_id      = Column(String(64), index=True)
    image_filename  = Column(String(256))
    eye_side        = Column(String(10))          # "left" | "right"

    # AI results
    predicted_label = Column(String(64))
    confidence      = Column(Float)
    severity_level  = Column(Integer)
    recommended_action = Column(String(256))

    # Quality check
    quality_ok      = Column(Boolean, default=True)
    quality_reason  = Column(String(256))

    # Agent summary
    agent_summary   = Column(Text)               # Plain-English report

    # Meta
    flagged_urgent  = Column(Boolean, default=False)
    created_at      = Column(DateTime, default=datetime.utcnow)
    reviewed_by     = Column(String(64), nullable=True)


class Patient(Base):
    """Patient demographics and risk factors."""
    __tablename__ = "patients"

    id          = Column(Integer, primary_key=True, index=True)
    patient_id  = Column(String(64), unique=True, index=True)
    name        = Column(String(128))
    age         = Column(Integer)
    diabetic    = Column(Boolean, default=False)
    hba1c       = Column(Float, nullable=True)    # Blood sugar level
    notes       = Column(Text, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)


# ── Helpers ──────────────────────────────────────────────────────

async def init_db():
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency: yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
