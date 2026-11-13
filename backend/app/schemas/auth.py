from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.models.user import UserRole


def _validate_email_format(value: str) -> str:
    cleaned = value.strip().lower()
    if "@" not in cleaned or "." not in cleaned.split("@")[-1] or len(cleaned) < 5 or " " in cleaned:
        raise ValueError("Must be a valid email address.")
    return cleaned


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=1, max_length=256)

    @field_validator("email")
    @classmethod
    def check_email(cls, v: str) -> str:
        return _validate_email_format(v)


class RegisterViewerRequest(BaseModel):
    """Restricted public registration schema (only enabled when ALLOW_VIEWER_SELF_REGISTRATION=true)."""
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=256)
    full_name: Optional[str] = Field(default=None, max_length=120)

    @field_validator("email")
    @classmethod
    def check_email(cls, v: str) -> str:
        return _validate_email_format(v)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=10)


class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = None


class UserCreateRequest(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=8, max_length=256)
    full_name: Optional[str] = Field(default=None, max_length=120)
    role: UserRole = UserRole.REVIEWER

    @field_validator("email")
    @classmethod
    def check_email(cls, v: str) -> str:
        return _validate_email_format(v)


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = Field(default=None, max_length=120)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(default=None, min_length=8, max_length=256)


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    last_login_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserListResponse(BaseModel):
    items: List[UserResponse]
    total: int


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

