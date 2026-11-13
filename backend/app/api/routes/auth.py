from fastapi import APIRouter, Depends, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_client_ip, get_current_user, get_db, require_roles
from backend.app.config import settings
from backend.app.core.exceptions import PermissionDeniedError, RateLimitExceededError
from backend.app.core.security import rate_limiter
from backend.app.models.user import User, UserRole
from backend.app.schemas.auth import (
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterViewerRequest,
    TokenResponse,
    UserCreateRequest,
    UserListResponse,
    UserResponse,
    UserUpdateRequest,
)
from backend.app.services.auth_service import auth_service

router = APIRouter()


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    client_ip = get_client_ip(request)
    allowed_ip, _ = rate_limiter.check_and_consume(
        f"login:ip:{client_ip}",
        max_requests=settings.LOGIN_RATE_LIMIT_PER_MINUTE,
        window_seconds=60,
    )
    allowed_acct, _ = rate_limiter.check_and_consume(
        f"login:email:{payload.email}",
        max_requests=settings.LOGIN_RATE_LIMIT_PER_MINUTE,
        window_seconds=60,
    )
    if not allowed_ip or not allowed_acct:
        raise RateLimitExceededError()

    user = await auth_service.authenticate_user(db, payload.email, payload.password)
    return await auth_service.issue_tokens(db, user)


@router.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_viewer(
    payload: RegisterViewerRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Restricted optional registration endpoint.
    Disabled by default; even when enabled via ALLOW_VIEWER_SELF_REGISTRATION=true,
    it only assigns the lowest-privilege VIEWER role and never ADMIN or REVIEWER.
    """
    if not settings.ALLOW_VIEWER_SELF_REGISTRATION:
        raise PermissionDeniedError(
            "Self-registration is disabled. Contact an administrator to provision an account.",
            error_code="REGISTRATION_DISABLED",
        )
    client_ip = get_client_ip(request)
    allowed, _ = rate_limiter.check_and_consume(
        f"register:ip:{client_ip}",
        max_requests=max(3, settings.LOGIN_RATE_LIMIT_PER_MINUTE // 2),
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitExceededError()

    user = await auth_service.create_user(
        db,
        UserCreateRequest(
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name,
            role=UserRole.VIEWER,
        ),
    )
    return UserResponse.model_validate(user)


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh_tokens(
    payload: RefreshRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    client_ip = get_client_ip(request)
    allowed, _ = rate_limiter.check_and_consume(
        f"refresh:ip:{client_ip}",
        max_requests=settings.LOGIN_RATE_LIMIT_PER_MINUTE * 2,
        window_seconds=60,
    )
    if not allowed:
        raise RateLimitExceededError()

    return await auth_service.rotate_refresh_token(db, payload.refresh_token)


@router.post("/auth/logout")
async def logout(
    payload: LogoutRequest = LogoutRequest(),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    revoked_count = await auth_service.revoke_refresh_token(
        db,
        user=current_user,
        raw_refresh_token=payload.refresh_token,
    )
    return {"status": "logged_out", "revoked_tokens": revoked_count}


@router.get("/auth/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return UserResponse.model_validate(current_user)


@router.get("/users", response_model=UserListResponse)
async def list_users(
    current_user: User = Depends(require_roles(UserRole.ADMIN)),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    users = list(result.scalars().all())
    return UserListResponse(
        items=[UserResponse.model_validate(u) for u in users],
        total=len(users),
    )


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user_admin(
    payload: UserCreateRequest,
    current_user: User = Depends(require_roles(UserRole.ADMIN)),
    db: AsyncSession = Depends(get_db),
):
    user = await auth_service.create_user(db, payload)
    return UserResponse.model_validate(user)


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user_admin(
    user_id: str,
    payload: UserUpdateRequest,
    current_user: User = Depends(require_roles(UserRole.ADMIN)),
    db: AsyncSession = Depends(get_db),
):
    user = await auth_service.update_user(db, user_id, payload, acting_admin=current_user)
    return UserResponse.model_validate(user)

