import os
import secrets
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_INSECURE_SECRET_PLACEHOLDERS = {
    "changeme",
    "change-me",
    "secret",
    "dev-secret",
    "documind-secret",
    "your-secret-key",
    "supersecret",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PROJECT_NAME: str = "DocuMind Document Intelligence"
    API_V1_STR: str = "/api/v1"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    CORS_ORIGINS: Union[str, List[str]] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    DATABASE_URL: str = "sqlite+aiosqlite:///./documind.db"
    SYNC_DATABASE_URL: Optional[str] = None
    UPLOAD_DIR: str = "storage/uploads"
    MAX_FILE_SIZE_MB: int = 15

    # Async Worker & Broker Configuration (Celery + Redis)
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    CELERY_TASK_ALWAYS_EAGER: bool = False
    CELERY_TASK_SOFT_TIME_LIMIT: int = 120
    CELERY_TASK_TIME_LIMIT: int = 180
    CELERY_WORKER_MAX_TASKS_PER_CHILD: int = 50
    STALE_JOB_TIMEOUT_SECONDS: int = 300
    MAX_JOB_ATTEMPTS: int = 3
    MAX_CONCURRENT_GPU_INFERENCE: int = 1
    PRELOAD_LAYOUTLMV3_ON_WORKER_START: bool = False

    # Authentication & RBAC Security Configuration
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    LOGIN_RATE_LIMIT_PER_MINUTE: int = 10
    ALLOW_VIEWER_SELF_REGISTRATION: bool = False
    FLOWER_BASIC_AUTH: Optional[str] = None

    # Upload & Media Hardening Bounds
    MAX_IMAGE_PIXELS: int = 40_000_000
    MAX_PDF_PAGES_HARD_LIMIT: int = 50
    RETENTION_DAYS: int = 90
    LOG_FORMAT: str = "json"

    # ML & OCR Pipeline Configuration
    DEFAULT_EXTRACTION_METHOD: str = "TESSERACT_BASELINE"
    ENABLE_ML_INFERENCE: bool = True
    TESSERACT_CMD: Optional[str] = None
    OCR_PDF_DPI: int = 300
    OCR_MAX_PAGES: int = 5
    OCR_CONFIDENCE_THRESHOLD: float = 0.60
    DEFAULT_DATE_FORMAT_ORDER: str = "DMY"
    LAYOUTLMV3_CHECKPOINT_DIR: str = "checkpoints/layoutlmv3-sroie-best"

    # Human-in-the-Loop Review & Preview Configuration
    REVIEW_CONFIDENCE_THRESHOLD: float = 0.75
    PREVIEW_MAX_PDF_PAGES: int = 20
    PREVIEW_RENDER_DPI: int = 150
    DEFAULT_REVIEWER_ID: str = "local-anonymous"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def sync_database_url(self) -> str:
        if self.SYNC_DATABASE_URL:
            return self.SYNC_DATABASE_URL
        url = self.DATABASE_URL
        if url.startswith("postgresql+asyncpg://"):
            return "postgresql+psycopg2://" + url[len("postgresql+asyncpg://") :]
        if url.startswith("sqlite+aiosqlite://"):
            return "sqlite://" + url[len("sqlite+aiosqlite://") :]
        return url

    def validate_runtime_security(self) -> str:
        """
        Ensures a valid JWT signing secret is configured. In production, weak or missing
        secrets fail startup immediately. In local development/test, an ephemeral secret
        is generated for the process lifetime when JWT_SECRET_KEY is not explicitly set.
        """
        env = (self.ENVIRONMENT or "development").strip().lower()
        secret = (self.JWT_SECRET_KEY or "").strip()

        if env == "production":
            if not secret or len(secret) < 32 or secret.lower() in _INSECURE_SECRET_PLACEHOLDERS:
                raise RuntimeError(
                    "JWT_SECRET_KEY must be set to a cryptographically strong string (>= 32 chars) in production."
                )
            if self.DEBUG:
                raise RuntimeError("DEBUG must be False in production.")
            return secret

        if not secret:
            secret = secrets.token_urlsafe(48)
            self.JWT_SECRET_KEY = secret
        elif secret.lower() in _INSECURE_SECRET_PLACEHOLDERS or len(secret) < 16:
            raise RuntimeError(
                "Configured JWT_SECRET_KEY is too short or matches a known weak placeholder."
            )
        return secret


settings = Settings()

