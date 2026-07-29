import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool


def _build_database_url() -> str:
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    db = os.environ.get("APP_DB_NAME", "mem0_app")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


# evercompliant POC patch: with Aurora Serverless v2 auto-pause (min 0 ACU), idle
# pooled connections keep the cluster awake forever. MEM0_DB_NULLPOOL=true opens a
# connection per session instead — slower per-request, but lets Aurora pause.
_USE_NULLPOOL = os.environ.get("MEM0_DB_NULLPOOL", "").lower() in {"1", "true", "yes", "on"}

engine = create_engine(
    _build_database_url(),
    pool_pre_ping=True,
    **({"poolclass": NullPool} if _USE_NULLPOOL else {}),
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency that yields a SQLAlchemy session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
