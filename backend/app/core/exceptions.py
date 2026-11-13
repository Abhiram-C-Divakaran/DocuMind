from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class DocuMindException(HTTPException):
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = "INTERNAL_ERROR",
        extra: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.extra = extra or {}


class FileValidationError(DocuMindException):
    def __init__(self, detail: str, extra: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="INVALID_FILE",
            extra=extra,
        )


class FileSizeLimitExceededError(DocuMindException):
    def __init__(self, detail: str, max_size_mb: int):
        super().__init__(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=detail,
            error_code="FILE_TOO_LARGE",
            extra={"max_size_mb": max_size_mb},
        )


class DocumentNotFoundError(DocuMindException):
    def __init__(self, document_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' was not found.",
            error_code="DOCUMENT_NOT_FOUND",
            extra={"document_id": document_id},
        )


class StorageError(DocuMindException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="STORAGE_ERROR",
        )


class ConflictError(DocuMindException):
    def __init__(self, detail: str, error_code: str = "CONFLICT", extra: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code=error_code,
            extra=extra,
        )


class ReviewNotFoundError(DocuMindException):
    def __init__(self, document_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review record for document '{document_id}' was not found.",
            error_code="REVIEW_NOT_FOUND",
            extra={"document_id": document_id},
        )


class JobNotFoundError(DocuMindException):
    def __init__(self, job_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Processing job '{job_id}' was not found.",
            error_code="JOB_NOT_FOUND",
            extra={"job_id": job_id},
        )


class AuthenticationError(DocuMindException):
    def __init__(self, detail: str = "Authentication required.", error_code: str = "UNAUTHORIZED"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code=error_code,
        )


class PermissionDeniedError(DocuMindException):
    def __init__(self, detail: str = "Insufficient permissions for this resource.", error_code: str = "FORBIDDEN"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code=error_code,
        )


class RateLimitExceededError(DocuMindException):
    def __init__(self, detail: str = "Too many authentication requests. Please retry later."):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED",
        )

