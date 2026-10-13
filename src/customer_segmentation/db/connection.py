"""Database connection and session management for BCIP."""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()

DEFAULT_DATABASE_URL = (
    "postgresql://bcip_admin:bcip_secure_pass_2026@localhost:5435/bcip_db"
)


def get_database_url() -> str:
    """Returns database URL from environment or default."""
    url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    # Ensure psycopg2 driver prefix if plain postgresql:// is provided
    if url.startswith("postgresql://") and not url.startswith("postgresql+psycopg2://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def get_engine() -> Engine:
    """Returns a singleton SQLAlchemy engine."""
    global _engine, _SessionFactory
    if _engine is None:
        db_url = get_database_url()
        _engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        _SessionFactory = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def get_session_factory() -> sessionmaker:
    """Returns the sessionmaker instance."""
    if _SessionFactory is None:
        get_engine()
    assert _SessionFactory is not None
    return _SessionFactory


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for transactional DB session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_db_connection() -> dict[str, bool | str]:
    """Checks database connectivity and returns status."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();")).scalar()
            return {"connected": True, "version": str(result)}
    except Exception as exc:  # noqa: BLE001
        return {"connected": False, "error": str(exc)}
