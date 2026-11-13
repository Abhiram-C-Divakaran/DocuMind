import os
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
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
    UPLOAD_DIR: str = "storage/uploads"
    MAX_FILE_SIZE_MB: int = 15

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


settings = Settings()
