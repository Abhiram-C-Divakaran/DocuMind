import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import (
    auth_router,
    documents_router,
    health_router,
    jobs_router,
    models_router,
    reviews_router,
)
from backend.app.config import settings
from backend.app.core.exceptions import DocuMindException
from backend.app.database import Base, engine
from backend.app.logger import logger, request_id_var


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_runtime_security()
    logger.info("Initializing DocuMind application...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables verified.")
    yield
    logger.info("Shutting down DocuMind application...")
    await engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Multimodal Document Intelligence Platform API for receipts and invoices.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-CSRF-Token", "X-Auth-Transport"],
)


# Request ID, security response headers & structured timing middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_var.set(req_id)
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        request_id_var.reset(token)

    response.headers["X-Request-ID"] = req_id
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    if request.url.path.startswith(f"{settings.API_V1_STR}/auth"):
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("Pragma", "no-cache")
    if request.url.path.startswith("/api/"):
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        )

    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code}",
        extra={
            "event": "http_request",
            "request_id": req_id,
            "status": str(response.status_code),
            "duration_ms": duration_ms,
        },
    )
    return response


# Central Error Handlers
@app.exception_handler(DocuMindException)
async def documind_exception_handler(request: Request, exc: DocuMindException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.detail,
                "details": exc.extra,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        errors.append(f"{loc}: {err.get('msg', 'Invalid value')}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Input validation failed.",
                "details": {"errors": errors},
            }
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled server error on {request.method} {request.url.path}",
        extra={"event": "unhandled_exception", "error_category": type(exc).__name__},
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred processing your request.",
                "details": {},
            }
        },
    )


# Mount routers
app.include_router(health_router)
app.include_router(auth_router, prefix=settings.API_V1_STR, tags=["Authentication & Users"])
app.include_router(documents_router, prefix=settings.API_V1_STR)
app.include_router(jobs_router, prefix=f"{settings.API_V1_STR}/jobs", tags=["Processing Jobs"])
app.include_router(models_router, prefix=settings.API_V1_STR)
app.include_router(reviews_router, prefix=settings.API_V1_STR)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "docs": "/docs",
        "health": "/health",
        "api_v1": settings.API_V1_STR,
    }

