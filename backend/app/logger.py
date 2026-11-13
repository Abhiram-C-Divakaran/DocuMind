import contextvars
import json
import logging
import os
import re
import sys
from typing import Any, Dict, Optional

request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(password|token|secret|authorization|raw_text|ocr_text|structured_data)",
    re.IGNORECASE,
)


def _sanitize_extra(data: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for k, v in data.items():
        if _SENSITIVE_KEY_PATTERN.search(k):
            sanitized[k] = "[REDACTED]"
        else:
            sanitized[k] = v
    return sanitized


class StructuredFormatter(logging.Formatter):
    _EXTRA_FIELDS = (
        "request_id",
        "document_id",
        "job_id",
        "extraction_method",
        "status",
        "duration_ms",
        "error_category",
        "user_id",
        "event",
    )

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        ctx_req_id = request_id_var.get()
        if ctx_req_id:
            log_obj["request_id"] = ctx_req_id

        extra: Dict[str, Any] = {}
        for field in self._EXTRA_FIELDS:
            val = getattr(record, field, None)
            if val is not None:
                extra[field] = val
        if extra:
            log_obj.update(_sanitize_extra(extra))

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        if os.environ.get("LOG_FORMAT", "json").lower() == "text":
            rid = f" req={log_obj['request_id']}" if "request_id" in log_obj else ""
            return f"[{log_obj['timestamp']}] [{log_obj['level']}] [{log_obj['logger']}]{rid} {log_obj['message']}"

        return json.dumps(log_obj, default=str)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger("documind")
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
        log.addHandler(handler)
        log.setLevel(level)
    return log


logger = setup_logging()


