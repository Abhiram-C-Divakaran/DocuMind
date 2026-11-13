from datetime import datetime, timezone
from typing import Optional

import jwt
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.core.exceptions import (
    AuthenticationError,
    ConflictError,
    DocuMindException,
    PermissionDeniedError,
)
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    hash_token,
    verify_password,
)
from backend.app.logger import logger
from backend.app.models.user import RefreshToken, User, UserRole
from backend.app.schemas.auth import TokenResponse, UserCreateRequest, UserResponse, UserUpdateRequest


class AuthService:
    async def get_user_by_id(self, db: AsyncSession, user_id: str) -> Optional[User]:
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_user_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        normalized = email.strip().lower()
        result = await db.execute(select(User).where(User.email == normalized))
        return result.scalar_one_or_none()

    async def create_user(
        self,
        db: AsyncSession,
        payload: UserCreateRequest,
    ) -> User:
        existing = await self.get_user_by_email(db, payload.email)
        if existing is not None:
            raise ConflictError(
                f"User with email '{payload.email}' already exists.",
                error_code="USER_ALREADY_EXISTS",
            )
        user = User(
            email=payload.email.strip().lower(),
            full_name=(payload.full_name or "").strip() or None,
            password_hash=hash_password(payload.password),
            role=payload.role.value,
            is_active=True,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        logger.info("User account created", extra={"event": "user_created", "user_id": user.id})
        return user

    async def update_user(
        self,
        db: AsyncSession,
        target_user_id: str,
        payload: UserUpdateRequest,
        acting_admin: User,
    ) -> User:
        user = await self.get_user_by_id(db, target_user_id)
        if not user:
            raise DocuMindException(status_code=404, detail="User not found.", error_code="USER_NOT_FOUND")

        if user.id == acting_admin.id and payload.is_active is False:
            raise PermissionDeniedError("Administrators cannot deactivate their own account.")
        if user.id == acting_admin.id and payload.role is not None and payload.role != UserRole.ADMIN:
            raise PermissionDeniedError("Administrators cannot demote their own active admin role.")

        if payload.full_name is not None:
            user.full_name = payload.full_name.strip() or None
        if payload.role is not None:
            user.role = payload.role.value
        if payload.is_active is not None:
            user.is_active = payload.is_active
        if payload.password is not None:
            user.password_hash = hash_password(payload.password)

        user.updated_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(user)
        return user

    async def authenticate_user(self, db: AsyncSession, email: str, password: str) -> User:
        user = await self.get_user_by_email(db, email)
        if not user or not verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid email or password.", error_code="INVALID_CREDENTIALS")
        if not user.is_active:
            raise AuthenticationError("User account is inactive.", error_code="ACCOUNT_DISABLED")
        user.last_login_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(user)
        return user

    async def issue_tokens(self, db: AsyncSession, user: User) -> TokenResponse:
        access_token, _ = create_access_token(
            subject=user.id,
            email=user.email,
            role=user.role,
        )
        refresh_token, jti, refresh_exp = create_refresh_token(subject=user.id)
        db.add(
            RefreshToken(
                user_id=user.id,
                jti=jti,
                token_hash=hash_token(refresh_token),
                expires_at=refresh_exp,
            )
        )
        await db.commit()
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.model_validate(user),
        )

    async def rotate_refresh_token(self, db: AsyncSession, raw_refresh_token: str) -> TokenResponse:
        try:
            payload = decode_token(raw_refresh_token, expected_type="refresh")
        except jwt.PyJWTError:
            raise AuthenticationError("Invalid or expired refresh token.", error_code="INVALID_REFRESH_TOKEN")

        jti = payload.get("jti")
        user_id = payload.get("sub")
        if not jti or not user_id:
            raise AuthenticationError("Invalid refresh token payload.", error_code="INVALID_REFRESH_TOKEN")

        token_hash_val = hash_token(raw_refresh_token)
        result = await db.execute(
            select(RefreshToken).where(
                RefreshToken.jti == jti,
                RefreshToken.token_hash == token_hash_val,
            )
        )
        stored = result.scalar_one_or_none()
        if not stored:
            raise AuthenticationError("Refresh token unrecognized.", error_code="INVALID_REFRESH_TOKEN")

        now = datetime.now(timezone.utc)
        if stored.revoked_at is not None:
            # Refresh token replay detected: revoke all active sessions for this user immediately
            await db.execute(
                update(RefreshToken)
                .where(RefreshToken.user_id == stored.user_id, RefreshToken.revoked_at.is_(None))
                .values(revoked_at=now)
            )
            await db.commit()
            logger.warning(
                "Revoked refresh token replay detected; invalidated all user sessions",
                extra={"event": "refresh_token_replay", "user_id": stored.user_id},
            )
            raise AuthenticationError("Refresh token has been revoked.", error_code="TOKEN_REVOKED")

        expires_at = stored.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            raise AuthenticationError("Refresh token has expired.", error_code="TOKEN_EXPIRED")

        user = await self.get_user_by_id(db, stored.user_id)
        if not user or not user.is_active:
            raise AuthenticationError("User account unavailable.", error_code="ACCOUNT_DISABLED")

        new_access, _ = create_access_token(subject=user.id, email=user.email, role=user.role)
        new_refresh, new_jti, new_exp = create_refresh_token(subject=user.id)

        stored.revoked_at = now
        stored.replaced_by_jti = new_jti
        db.add(
            RefreshToken(
                user_id=user.id,
                jti=new_jti,
                token_hash=hash_token(new_refresh),
                expires_at=new_exp,
            )
        )
        await db.commit()

        return TokenResponse(
            access_token=new_access,
            refresh_token=new_refresh,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.model_validate(user),
        )

    async def revoke_refresh_token(
        self,
        db: AsyncSession,
        user: User,
        raw_refresh_token: Optional[str] = None,
    ) -> int:
        now = datetime.now(timezone.utc)
        if raw_refresh_token:
            token_hash_val = hash_token(raw_refresh_token)
            res = await db.execute(
                update(RefreshToken)
                .where(
                    RefreshToken.user_id == user.id,
                    RefreshToken.token_hash == token_hash_val,
                    RefreshToken.revoked_at.is_(None),
                )
                .values(revoked_at=now)
            )
        else:
            res = await db.execute(
                update(RefreshToken)
                .where(RefreshToken.user_id == user.id, RefreshToken.revoked_at.is_(None))
                .values(revoked_at=now)
            )
        await db.commit()
        return int(res.rowcount or 0)


auth_service = AuthService()

