# DocuMind Security Architecture & Controls

This document details the implemented authentication, authorization, session-handling, upload-validation, rate-limiting, preview-isolation, and operational security controls in DocuMind.

---

## 1. Authentication & Session Architecture

- **Password Hashing**: Passwords are hashed using Argon2 (`argon2.PasswordHasher(time_cost=2, memory_cost=65536, parallelism=2, hash_len=32, salt_len=16)`) in [security.py](file:///c:/Users/Abhiram/Documents/Projects/Doc/backend/app/core/security.py).
- **Short-Lived Access Tokens**: Stateless HS256 JWT access tokens (`JWT_SECRET`, validated against empty or weak/default values in production) are kept exclusively in browser JS memory (`accessToken` variable in [authStorage.ts](file:///c:/Users/Abhiram/Documents/Projects/Doc/frontend/src/api/authStorage.ts)) — never persisted in `localStorage` or `sessionStorage`.
- **HttpOnly Refresh Cookies + Double-Submit CSRF**:
  - Browser clients authenticate with `X-Auth-Transport: cookie`.
  - `/api/v1/auth/login` and `/api/v1/auth/refresh` issue a rotating refresh token inside an `HttpOnly`, `SameSite=lax` cookie (`documind_refresh_token`) path-scoped to `/api/v1/auth` (with `Secure=true` in production / HTTPS).
  - `/api/v1/auth/refresh` and `/api/v1/auth/logout` validate the matching non-HttpOnly `documind_csrf_token` cookie against the `X-CSRF-Token` request header whenever cookie transport is used.
- **Refresh Token Rotation & Replay Detection**:
  - Raw refresh tokens are hashed with SHA-256 (`hash_token`) before database storage (`RefreshToken`).
  - Reusing a revoked or previously rotated refresh token triggers family/session revocation to mitigate token theft.

## 2. Role-Based Access Control (RBAC) & Resource Ownership

- Three explicit roles: `ADMIN`, `REVIEWER`, and `VIEWER`.
- **Ownership Isolation**: Document reads, downloads, and status queries enforce `document.owner_id == current_user.id` unless `current_user.role == ADMIN`.
- **Mutation Guards**:
  - Upload (`POST /api/v1/documents`) and review mutations (`PATCH /review`, `POST /review/approve|reject|reopen`) require `ADMIN` or `REVIEWER`.
  - User provisioning/status management (`GET|POST|PATCH /api/v1/users`) and job queue ops (`POST /api/v1/jobs/{id}/retry`) require `ADMIN`.
  - `VIEWER` accounts are strictly read-only across documents and reviews.

## 3. Rate Limiting & Proxy Trust Boundaries

- **Fail-Closed Production Rate Limiter**: `SlidingWindowRateLimiter` uses Redis ZSET sliding windows across login, upload, and API traffic. When `RATE_LIMIT_FAIL_CLOSED` is enabled (automatically true in production, or explicitly configured via env), Redis outages trigger `503 Service Unavailable` (`ServiceUnavailableError`) after periodic recovery probes rather than silently bypassing limits.
- **Trusted Proxy Header Control**: `get_client_ip` in [deps.py](file:///c:/Users/Abhiram/Documents/Projects/Doc/backend/app/api/deps.py) ignores `X-Forwarded-For` and `X-Real-IP` unless `TRUST_PROXY_HEADERS=true` is explicitly enabled behind a hardened reverse proxy.

## 4. Document Upload & Preview Isolation

- **Chunked Streaming Size Limits**: Uploads stream in chunks and abort immediately with HTTP 413 if `MAX_UPLOAD_SIZE_BYTES` is exceeded.
- **Signature & Decompression Protections**:
  - Extension + MIME + magic-byte signature validation (`PDF`, `PNG`, `JPEG`, `TIFF` / `WEBP`).
  - Image pixel limits reject decompression bombs (`Image.MAX_IMAGE_PIXELS` / explicit pixel checks).
  - PDF page-count ceilings reject oversized documents before expensive rendering.
- **Filename & Storage Traversal Guards**: Client filenames are sanitized, stored under generated UUID names, and `StorageService.get_file_path` rejects path traversal (`../`, nested paths) with `StorageError`.
- **Controlled Document Preview**:
  - Multi-page PDFs are rasterized server-side page-by-page to PNG images (`/api/v1/documents/{id}/file?page=N`) rather than serving raw potentially active PDF/HTML payloads directly in-origin.
  - Responses enforce `X-Content-Type-Options: nosniff` and sanitized `Content-Disposition: inline`.

## 5. HTTP Security Headers, Production Docs & Log Redaction

- **Response Headers** ([main.py](file:///c:/Users/Abhiram/Documents/Projects/Doc/backend/app/main.py) & [nginx.conf](file:///c:/Users/Abhiram/Documents/Projects/Doc/frontend/nginx.conf)):
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `Referrer-Policy: no-referrer`
  - `Permissions-Policy: camera=(), microphone=(), geolocation=()`
  - Strict API `Content-Security-Policy` (`default-src 'none'; frame-ancestors 'none'`) and `object-src 'none'` on the frontend static origin.
  - `Cache-Control: no-store` and `Pragma: no-cache` on `/api/v1/auth/*` responses.
- **Production Hardening**: OpenAPI (`/docs`, `/redoc`, `/openapi.json`) is disabled when `is_production` is true; `docker-compose.prod.yml` uses `ports: !reset []` to strip host exposure of FastAPI and Flower and binds Nginx to `127.0.0.1`.
- **Structured Log Redaction**: [logger.py](file:///c:/Users/Abhiram/Documents/Projects/Doc/backend/app/logger.py) filters keys matching `(password|token|secret|authorization|raw_text|ocr_text|structured_data)` and replaces their values with `[REDACTED]`.

