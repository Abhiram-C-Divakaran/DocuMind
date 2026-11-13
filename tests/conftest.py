import asyncio
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import AsyncGenerator, Tuple
from unittest.mock import patch

from PIL import Image
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.config import settings
from backend.app.core.security import create_access_token, hash_password, rate_limiter
from backend.app.database import Base, get_db, reset_sync_engine
from backend.app.main import app
from backend.app.models.user import User, UserRole
from backend.app.services.document_service import DocumentService
from backend.app.services.storage import storage_service

# Minimal valid binary PDF payload
MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 300 144]/Parent 2 0 R>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n168\n%%EOF\n"
)


def _make_minimal_image_bytes(fmt: str) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(buf, format=fmt)
    return buf.getvalue()


MINIMAL_PNG_BYTES = _make_minimal_image_bytes("PNG")
MINIMAL_JPEG_BYTES = _make_minimal_image_bytes("JPEG")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def _reset_rate_limiter_and_fast_dispatch():
    rate_limiter.reset_memory()
    with (
        patch.object(
            DocumentService,
            "_dispatch_celery_job",
            return_value=("celery-test-task", None),
        ),
        patch.object(
            DocumentService,
            "_revoke_celery_task",
            return_value=None,
        ),
    ):
        yield
    rate_limiter.reset_memory()


@pytest_asyncio.fixture(scope="function")
async def test_db_session() -> AsyncGenerator[AsyncSession, None]:
    db_dir = tempfile.mkdtemp()
    db_path = Path(db_dir) / "test_documind.db"
    async_url = f"sqlite+aiosqlite:///{db_path}"
    sync_url = f"sqlite:///{db_path}"

    orig_sync_url = settings.SYNC_DATABASE_URL
    settings.SYNC_DATABASE_URL = sync_url
    reset_sync_engine()

    engine = create_async_engine(
        async_url,
        connect_args={"check_same_thread": False},
        future=True,
    )
    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()
    reset_sync_engine()
    settings.SYNC_DATABASE_URL = orig_sync_url
    shutil.rmtree(db_dir, ignore_errors=True)


async def _seed_user(
    session: AsyncSession,
    email: str,
    role: UserRole,
    password: str = "StrongPass123!",
) -> Tuple[User, str]:
    user = User(
        email=email,
        full_name=f"Test {role.value}",
        password_hash=hash_password(password),
        role=role.value,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    token, _ = create_access_token(subject=user.id, email=user.email, role=user.role)
    return user, token


@pytest_asyncio.fixture(scope="function")
async def test_users(test_db_session: AsyncSession) -> dict:
    admin, admin_token = await _seed_user(test_db_session, "admin@test.local", UserRole.ADMIN)
    reviewer, reviewer_token = await _seed_user(test_db_session, "reviewer@test.local", UserRole.REVIEWER)
    other_reviewer, other_token = await _seed_user(test_db_session, "other@test.local", UserRole.REVIEWER)
    viewer, viewer_token = await _seed_user(test_db_session, "viewer@test.local", UserRole.VIEWER)
    return {
        "admin": (admin, admin_token),
        "reviewer": (reviewer, reviewer_token),
        "other": (other_reviewer, other_token),
        "viewer": (viewer, viewer_token),
    }


@pytest_asyncio.fixture(scope="function")
async def anon_client(test_db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    temp_dir = tempfile.mkdtemp()
    original_upload_dir = storage_service.base_dir
    storage_service.base_dir = Path(temp_dir)

    async def override_get_db():
        yield test_db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

    app.dependency_overrides.clear()
    storage_service.base_dir = original_upload_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest_asyncio.fixture(scope="function")
async def client(
    anon_client: AsyncClient, test_users: dict
) -> AsyncGenerator[AsyncClient, None]:
    _, token = test_users["reviewer"]
    anon_client.headers.update({"Authorization": f"Bearer {token}"})
    yield anon_client


@pytest_asyncio.fixture(scope="function")
async def admin_client(
    test_db_session: AsyncSession, test_users: dict
) -> AsyncGenerator[AsyncClient, None]:
    _, token = test_users["admin"]
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"Authorization": f"Bearer {token}"},
    ) as ac:
        yield ac


@pytest_asyncio.fixture(scope="function")
async def viewer_client(
    test_db_session: AsyncSession, test_users: dict
) -> AsyncGenerator[AsyncClient, None]:
    _, token = test_users["viewer"]
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"Authorization": f"Bearer {token}"},
    ) as ac:
        yield ac
