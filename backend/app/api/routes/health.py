from typing import Any, Dict

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.database import get_db
from ml.extractors.layoutlmv3_extractor import get_cached_layoutlmv3_extractor
from ml.ocr.engine import get_tesseract_status

router = APIRouter()


def _check_redis_status() -> str:
    try:
        import redis

        client = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=0.5, socket_timeout=0.5)
        try:
            client.ping()
            return "connected"
        finally:
            client.close()
    except Exception:
        return "unavailable"


def _check_worker_status() -> str:
    try:
        from backend.app.worker.celery_app import celery_app

        insp = celery_app.control.inspect(timeout=0.5)
        pong = insp.ping() if insp else None
        return "available" if pong else "unavailable"
    except Exception:
        return "unavailable"


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    db_status = "disconnected"
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    redis_status = _check_redis_status()
    worker_status = _check_worker_status() if redis_status == "connected" else "unavailable"
    tess_ok, _, _ = get_tesseract_status()
    lm_info = get_cached_layoutlmv3_extractor().get_readiness_info()

    overall = "healthy" if db_status == "connected" else "degraded"
    return {
        "status": overall,
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "database": db_status,
        "redis": redis_status,
        "worker": worker_status,
        "tesseract_available": bool(tess_ok),
        "layoutlmv3_ready": bool(lm_info.get("ready", False)),
        "layoutlmv3_is_production_ready": bool(lm_info.get("is_production_ready", False)),
    }
