from backend.app.api.routes.auth import router as auth_router
from backend.app.api.routes.documents import router as documents_router
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.jobs import router as jobs_router
from backend.app.api.routes.models import router as models_router
from backend.app.api.routes.reviews import router as reviews_router

__all__ = [
    "auth_router",
    "health_router",
    "documents_router",
    "jobs_router",
    "models_router",
    "reviews_router",
]
