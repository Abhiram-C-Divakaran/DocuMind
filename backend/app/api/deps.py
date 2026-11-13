from typing import Callable, Optional

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.exceptions import AuthenticationError, PermissionDeniedError
from backend.app.core.security import decode_token
from backend.app.database import get_db
from backend.app.models.document import Document
from backend.app.models.user import User, UserRole
from backend.app.services.auth_service import auth_service

_bearer_scheme = HTTPBearer(auto_error=False)


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Bearer authentication token is required.")

    try:
        payload = decode_token(credentials.credentials, expected_type="access")
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Access token has expired.", error_code="TOKEN_EXPIRED")
    except jwt.PyJWTError:
        raise AuthenticationError("Invalid access token.", error_code="INVALID_TOKEN")

    user_id = payload.get("sub")
    if not user_id:
        raise AuthenticationError("Invalid token subject.", error_code="INVALID_TOKEN")

    user = await auth_service.get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise AuthenticationError("User account is disabled or missing.", error_code="ACCOUNT_DISABLED")
    return user


def require_roles(*allowed_roles: UserRole) -> Callable[[User], User]:
    allowed_values = {role.value for role in allowed_roles}

    async def _role_dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_values:
            raise PermissionDeniedError(
                f"Role '{current_user.role}' is not authorized to perform this action."
            )
        return current_user

    return _role_dependency


def authorize_document_access(document: Document, user: User) -> None:
    """
    Enforces account ownership isolation:
    - ADMIN may inspect any document, including legacy pre-Phase-4A documents where owner_id is NULL.
    - Non-admin accounts (REVIEWER, VIEWER) may only access documents they own (document.owner_id == user.id).
      Unowned legacy records are never accessible to non-admin users.
    """
    if user.role == UserRole.ADMIN.value:
        return
    if document.owner_id is not None and document.owner_id == user.id:
        return
    raise PermissionDeniedError("You do not have access to this document.")


__all__ = [
    "get_db",
    "AsyncSession",
    "get_current_user",
    "require_roles",
    "authorize_document_access",
    "get_client_ip",
]
