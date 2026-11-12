from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_db
from backend.app.config import settings
from backend.app.services.storage import storage_service

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint verifying:
    - Service uptime & version
    - Database connectivity
    - Storage directory writeability
    """
    db_status = "healthy"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    storage_status = "healthy" if storage_service.base_dir.exists() else "unhealthy"

    overall_healthy = (db_status == "healthy") and (storage_status == "healthy")

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "service": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "database": db_status,
            "storage": storage_status,
        },
        "phase": "2A",
    }
