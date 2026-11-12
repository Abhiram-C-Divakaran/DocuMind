import os
import uuid
from pathlib import Path
from typing import Tuple
from backend.app.config import settings
from backend.app.core.exceptions import StorageError
from backend.app.logger import logger


class StorageService:
    def __init__(self, base_dir: str = settings.UPLOAD_DIR):
        self.base_dir = Path(base_dir).resolve()
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create storage directory {self.base_dir}: {e}")
            raise StorageError(f"Storage directory initialization failed: {e}")

    def save_file(self, file_bytes: bytes, extension: str) -> Tuple[str, str, str]:
        """
        Saves file bytes with a secure UUIDv4 filename.
        Returns:
            Tuple of (file_id, stored_filename, full_file_path)
        """
        file_id = str(uuid.uuid4())
        normalized_ext = extension if extension.startswith(".") else f".{extension}"
        stored_filename = f"{file_id}{normalized_ext}"
        destination = self.base_dir / stored_filename

        try:
            with open(destination, "wb") as f:
                f.write(file_bytes)
            logger.info(f"Saved file {stored_filename} ({len(file_bytes)} bytes) at {destination}")
            return file_id, stored_filename, str(destination)
        except Exception as e:
            logger.error(f"Failed to save file to {destination}: {e}")
            raise StorageError(f"Failed to persist file: {e}")

    def get_file_path(self, stored_filename: str) -> Path:
        target = (self.base_dir / stored_filename).resolve()
        # Security check: prevent directory traversal
        if not str(target).startswith(str(self.base_dir)):
            raise StorageError("Access denied: Invalid file path traversal detected.")
        if not target.exists():
            raise StorageError(f"File {stored_filename} does not exist.")
        return target

    def delete_file(self, stored_filename: str) -> bool:
        try:
            target = self.get_file_path(stored_filename)
            if target.exists():
                target.unlink()
                logger.info(f"Deleted file {stored_filename}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Error deleting file {stored_filename}: {e}")
            return False


storage_service = StorageService()

