import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import hash_password, verify_password
from backend.app.models.document import Document, DocumentStatus
from backend.app.models.user import RefreshToken, User, UserRole
from tests.conftest import MINIMAL_PDF_BYTES, MINIMAL_PNG_BYTES


@pytest.mark.asyncio
async def test_argon2_password_hashing_and_verification():
    hashed = hash_password("S3cur3P@ssword!")
    assert hashed.startswith("$argon2")
    assert verify_password("S3cur3P@ssword!", hashed) is True
    assert verify_password("WrongPass", hashed) is False


@pytest.mark.asyncio
async def test_unauthenticated_requests_rejected_401(anon_client: AsyncClient):
    protected_gets = [
        "/api/v1/auth/me",
        "/api/v1/documents",
        "/api/v1/documents/metrics/summary",
        "/api/v1/reviews",
        "/api/v1/reviews/metrics",
        "/api/v1/jobs",
        "/api/v1/jobs/metrics",
        "/api/v1/users",
    ]
    for path in protected_gets:
        res = await anon_client.get(path)
        assert res.status_code == 401, f"Expected 401 on {path}, got {res.status_code}"

    up_res = await anon_client.post(
        "/api/v1/documents",
        files={"file": ("a.pdf", MINIMAL_PDF_BYTES, "application/pdf")},
    )
    assert up_res.status_code == 401


@pytest.mark.asyncio
async def test_login_refresh_rotation_replay_detection_and_logout(
    anon_client: AsyncClient, test_users: dict, test_db_session: AsyncSession
):
    # Bad password
    bad = await anon_client.post(
        "/api/v1/auth/login",
        json={"email": "reviewer@test.local", "password": "wrong"},
    )
    assert bad.status_code == 401

    # Valid login
    login_res = await anon_client.post(
        "/api/v1/auth/login",
        json={"email": "reviewer@test.local", "password": "StrongPass123!"},
    )
    assert login_res.status_code == 200
    tokens = login_res.json()
    assert tokens["token_type"] == "bearer"
    assert tokens["user"]["email"] == "reviewer@test.local"
    assert tokens["user"]["role"] == "REVIEWER"

    # Access /me with bearer token
    me_res = await anon_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "reviewer@test.local"

    # Refresh token rotates pair
    rot_res = await anon_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert rot_res.status_code == 200
    new_tokens = rot_res.json()
    assert new_tokens["refresh_token"] != tokens["refresh_token"]

    # Replay of already-rotated old refresh token triggers 401 + revokes active sessions
    replay_res = await anon_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert replay_res.status_code == 401

    # Previously valid rotated token is now also revoked because of replay defense
    after_replay = await anon_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": new_tokens["refresh_token"]},
    )
    assert after_replay.status_code == 401

    # Fresh login + logout revocation
    login2 = await anon_client.post(
        "/api/v1/auth/login",
        json={"email": "reviewer@test.local", "password": "StrongPass123!"},
    )
    tok2 = login2.json()
    logout_res = await anon_client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": tok2["refresh_token"]},
        headers={"Authorization": f"Bearer {tok2['access_token']}"},
    )
    assert logout_res.status_code == 200
    assert logout_res.json()["revoked_tokens"] >= 1

    revoked_use = await anon_client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": tok2["refresh_token"]},
    )
    assert revoked_use.status_code == 401


@pytest.mark.asyncio
async def test_no_unrestricted_admin_self_registration(anon_client: AsyncClient):
    res = await anon_client.post(
        "/api/v1/auth/register",
        json={"email": "attacker@test.local", "password": "StrongPass123!"},
    )
    assert res.status_code == 403
    assert res.json()["error"]["code"] == "REGISTRATION_DISABLED"


@pytest.mark.asyncio
async def test_rbac_viewer_read_only_and_admin_endpoints(
    client: AsyncClient,
    viewer_client: AsyncClient,
    admin_client: AsyncClient,
):
    # Viewer cannot upload, manage users, or trigger job recovery / feedback export
    up_res = await viewer_client.post(
        "/api/v1/documents",
        files={"file": ("v.png", MINIMAL_PNG_BYTES, "image/png")},
    )
    assert up_res.status_code == 403

    for forbidden_call in (
        viewer_client.get("/api/v1/users"),
        viewer_client.post("/api/v1/jobs/recover"),
        viewer_client.post("/api/v1/reviews/feedback-export"),
        client.get("/api/v1/users"),
        client.post("/api/v1/jobs/recover"),
        client.post("/api/v1/reviews/feedback-export"),
    ):
        resp = await forbidden_call
        assert resp.status_code == 403

    # Reviewer creates a document; viewer/admin access check
    rev_up = await client.post(
        "/api/v1/documents",
        files={"file": ("rev.png", MINIMAL_PNG_BYTES, "image/png")},
        data={"extraction_method": "PLACEHOLDER"},
    )
    assert rev_up.status_code == 202
    doc_id = rev_up.json()["document"]["id"]

    # Viewer cannot mutate review on any document
    patch_rev = await viewer_client.patch(
        f"/api/v1/documents/{doc_id}/review",
        json={"corrected_data": {"vendor_name": "Hacked"}},
    )
    assert patch_rev.status_code == 403

    # Admin can create users and list users
    create_res = await admin_client.post(
        "/api/v1/users",
        json={
            "email": "newop@test.local",
            "password": "OperatorPassword123!",
            "full_name": "New Operator",
            "role": "REVIEWER",
        },
    )
    assert create_res.status_code == 201
    assert create_res.json()["role"] == "REVIEWER"

    users_res = await admin_client.get("/api/v1/users")
    assert users_res.status_code == 200
    assert any(u["email"] == "newop@test.local" for u in users_res.json()["items"])


@pytest.mark.asyncio
async def test_cross_account_ownership_isolation_and_legacy_unowned_docs(
    anon_client: AsyncClient,
    client: AsyncClient,
    admin_client: AsyncClient,
    test_users: dict,
    test_db_session: AsyncSession,
):
    # Upload by reviewer1 (`client`)
    up = await client.post(
        "/api/v1/documents",
        files={"file": ("owner1.png", MINIMAL_PNG_BYTES, "image/png")},
        data={"extraction_method": "PLACEHOLDER"},
    )
    assert up.status_code == 202
    doc_id = up.json()["document"]["id"]
    job_id = up.json()["job"]["id"]

    # Legacy pre-auth unowned document (owner_id = NULL)
    legacy = Document(
        owner_id=None,
        original_filename="legacy_pre4a.pdf",
        stored_filename="legacy_pre4a.pdf",
        file_path="storage/uploads/legacy_pre4a.pdf",
        file_size_bytes=123,
        content_type="application/pdf",
        status=DocumentStatus.COMPLETED.value,
    )
    test_db_session.add(legacy)
    await test_db_session.commit()
    await test_db_session.refresh(legacy)

    _, other_token = test_users["other"]
    other_headers = {"Authorization": f"Bearer {other_token}"}

    # Reviewer2 cannot inspect Reviewer1's document, status, preview, retry, review, or job
    for call in (
        anon_client.get(f"/api/v1/documents/{doc_id}", headers=other_headers),
        anon_client.get(f"/api/v1/documents/{doc_id}/status", headers=other_headers),
        anon_client.get(f"/api/v1/documents/{doc_id}/preview/meta", headers=other_headers),
        anon_client.get(f"/api/v1/documents/{doc_id}/review", headers=other_headers),
        anon_client.get(f"/api/v1/jobs/{job_id}", headers=other_headers),
        anon_client.post(f"/api/v1/documents/{doc_id}/retry", headers=other_headers),
        anon_client.delete(f"/api/v1/documents/{doc_id}", headers=other_headers),
        anon_client.get(f"/api/v1/documents/{legacy.id}", headers=other_headers),
    ):
        res = await call
        assert res.status_code == 403

    # Reviewer2's list shows neither Reviewer1's doc nor unowned legacy doc
    list_other = await anon_client.get("/api/v1/documents", headers=other_headers)
    assert list_other.status_code == 200
    assert list_other.json()["total"] == 0

    # Admin can inspect both Reviewer1's doc and legacy pre-4A doc
    assert (await admin_client.get(f"/api/v1/documents/{doc_id}")).status_code == 200
    assert (await admin_client.get(f"/api/v1/documents/{legacy.id}")).status_code == 200


@pytest.mark.asyncio
async def test_login_rate_limiting_returns_429(anon_client: AsyncClient, test_users: dict):
    from backend.app.config import settings

    orig = settings.LOGIN_RATE_LIMIT_PER_MINUTE
    settings.LOGIN_RATE_LIMIT_PER_MINUTE = 3
    try:
        for _ in range(3):
            r = await anon_client.post(
                "/api/v1/auth/login",
                json={"email": "reviewer@test.local", "password": "badpassword"},
            )
            assert r.status_code == 401
        blocked = await anon_client.post(
            "/api/v1/auth/login",
            json={"email": "reviewer@test.local", "password": "StrongPass123!"},
        )
        assert blocked.status_code == 429
        assert blocked.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    finally:
        settings.LOGIN_RATE_LIMIT_PER_MINUTE = orig

