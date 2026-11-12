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
