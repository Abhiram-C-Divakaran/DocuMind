import secrets
from typing import Optional

from fastapi import APIRouter, Body, Depends, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_client_ip, get_current_user, get_db, require_roles
from backend.app.config import settings
from backend.app.core.exceptions import AuthenticationError, PermissionDeniedError, RateLimitExceededError
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


def _is_cookie_secure(request: Request) -> bool:
    if settings.COOKIE_SECURE is not None:
        return bool(settings.COOKIE_SECURE)
    proto = (request.headers.get("X-Forwarded-Proto") or "").strip().lower()
    return settings.is_production or request.url.scheme == "https" or proto == "https"


def _set_auth_cookies(request: Request, response: Response, refresh_token: str) -> None:
    secure_flag = _is_cookie_secure(request)
    samesite_val = (settings.COOKIE_SAMESITE or "lax").lower()
    max_age = settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    csrf_token = secrets.token_urlsafe(32)

    response.set_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        value=refresh_token,
        max_age=max_age,
        httponly=True,
        secure=secure_flag,
        samesite=samesite_val,
        path=f"{settings.API_V1_STR}/auth",
    )
    response.set_cookie(
        key=settings.CSRF_COOKIE_NAME,
        value=csrf_token,
        max_age=max_age,
        httponly=False,
        secure=secure_flag,
        samesite=samesite_val,
        path="/",
    )


def _clear_auth_cookies(request: Request, response: Response) -> None:
    secure_flag = _is_cookie_secure(request)
    samesite_val = (settings.COOKIE_SAMESITE or "lax").lower()
    response.delete_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        path=f"{settings.API_V1_STR}/auth",
        secure=secure_flag,
        samesite=samesite_val,
    )
    response.delete_cookie(
        key=settings.CSRF_COOKIE_NAME,
        path="/",
        secure=secure_flag,
        samesite=samesite_val,
    )


def _verify_cookie_csrf(request: Request) -> None:
    csrf_cookie = request.cookies.get(settings.CSRF_COOKIE_NAME) or ""
    csrf_header = request.headers.get("X-CSRF-Token") or ""
    if not csrf_cookie or not csrf_header or not secrets.compare_digest(csrf_cookie, csrf_header):
        raise PermissionDeniedError(
            "CSRF validation failed for cookie-authenticated session request.",
            error_code="CSRF_INVALID",
        )


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
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
    tokens = await auth_service.issue_tokens(db, user)
    if tokens.refresh_token:
        _set_auth_cookies(request, response, tokens.refresh_token)
    if (request.headers.get("X-Auth-Transport") or "").strip().lower() == "cookie":
        tokens.refresh_token = None
    return tokens


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
    request: Request,
    response: Response,
    payload: Optional[RefreshRequest] = Body(default=None),
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

    body_token = payload.refresh_token if payload else None
    cookie_token = request.cookies.get(settings.REFRESH_COOKIE_NAME)
    if not body_token and cookie_token:
        _verify_cookie_csrf(request)
        raw_refresh = cookie_token
        cookie_mode = True
    elif body_token:
        raw_refresh = body_token
        cookie_mode = (request.headers.get("X-Auth-Transport") or "").strip().lower() == "cookie"
    else:
        raise AuthenticationError("Refresh token missing.", error_code="MISSING_REFRESH_TOKEN")

    try:
        tokens = await auth_service.rotate_refresh_token(db, raw_refresh)
    except AuthenticationError:
        _clear_auth_cookies(request, response)
        raise

    if tokens.refresh_token:
        _set_auth_cookies(request, response, tokens.refresh_token)
    if cookie_mode:
        tokens.refresh_token = None
    return tokens


@router.post("/auth/logout")
async def logout(
    request: Request,
    response: Response,
    payload: Optional[LogoutRequest] = Body(default=None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    body_token = payload.refresh_token if payload else None
    cookie_token = request.cookies.get(settings.REFRESH_COOKIE_NAME)
    if not body_token and cookie_token:
        _verify_cookie_csrf(request)
    raw_token = body_token or cookie_token

    revoked_count = await auth_service.revoke_refresh_token(
        db,
        user=current_user,
        raw_refresh_token=raw_token,
    )
    _clear_auth_cookies(request, response)
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

