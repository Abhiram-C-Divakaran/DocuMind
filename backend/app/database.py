from typing import AsyncGenerator, Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from backend.app.config import settings
from backend.app.logger import logger

connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG and not settings.DATABASE_URL.startswith("sqlite"),
    connect_args=connect_args,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

_sync_engine: Optional[Engine] = None
_SyncSessionLocal: Optional[sessionmaker] = None


def build_sync_session_factory(sync_url: Optional[str] = None) -> sessionmaker:
    url = sync_url or settings.sync_database_url
    sync_kwargs: dict = {"future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        sync_kwargs["connect_args"] = {"check_same_thread": False}
    eng = create_engine(url, **sync_kwargs)
    return sessionmaker(bind=eng, class_=Session, expire_on_commit=False, autocommit=False, autoflush=False)


def get_sync_session() -> Session:
    global _sync_engine, _SyncSessionLocal
    if _SyncSessionLocal is None:
        url = settings.sync_database_url
        sync_kwargs: dict = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            sync_kwargs["connect_args"] = {"check_same_thread": False}
        _sync_engine = create_engine(url, **sync_kwargs)
        _SyncSessionLocal = sessionmaker(
            bind=_sync_engine,
            class_=Session,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
    return _SyncSessionLocal()


def reset_sync_engine() -> None:
    global _sync_engine, _SyncSessionLocal
    if _sync_engine is not None:
        _sync_engine.dispose()
    _sync_engine = None
    _SyncSessionLocal = None


Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session rollback due to error: {e}")
            raise
        finally:
            await session.close()


