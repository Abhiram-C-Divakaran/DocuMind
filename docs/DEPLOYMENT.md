# DocuMind Production Deployment & Operations Runbook

This guide describes deploying and operating the DocuMind multi-service stack on a Linux host/VPS with Docker Compose, HTTPS TLS termination, database/upload backups, and operational recovery procedures.

---

## 1. Architecture Overview

| Service | Purpose | Network Exposure (Production) |
| --- | --- | --- |
| `frontend` | React static UI + Nginx `/api` reverse proxy | Bound to `127.0.0.1:3000` (behind host TLS reverse proxy) |
| `backend` | FastAPI API, Auth/RBAC, Preview Renderer, Outbox Dispatcher | Internal Docker network only (`ports: !reset []`) |
| `worker` | Celery worker running Tesseract OCR / LayoutLMv3 extraction | Internal Docker network only |
| `beat` | Celery Beat periodic stale-job & outbox recovery scheduler | Internal Docker network only |
| `postgres` | PostgreSQL persistence (documents, jobs, reviews, audit history) | Internal Docker network only |
| `redis` | Celery broker, result backend, and fail-closed rate limiter | Internal Docker network only (AOF + `--requirepass`) |
| `flower` | Optional Celery task dashboard (dev/ops profile) | Internal Docker network only (`ports: !reset []`) |

---

## 2. Environment Configuration

Copy [.env.example](file:///c:/Users/Abhiram/Documents/Projects/Doc/.env.example) to `.env` on the server and set strong random credentials:

```bash
POSTGRES_PASSWORD="<strong-random-postgres-password>"
REDIS_PASSWORD="<strong-random-redis-password>"
JWT_SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
CORS_ORIGINS="https://documind.example.com"
ENVIRONMENT="production"
DEBUG="false"
COOKIE_SECURE="true"
TRUST_PROXY_HEADERS="true"
RATE_LIMIT_FAIL_CLOSED="true"
FRONTEND_BIND_IP="127.0.0.1"
FRONTEND_PORT="3000"
```

Production safeguards enforced at startup ([config.py](file:///c:/Users/Abhiram/Documents/Projects/Doc/backend/app/config.py)):
- Weak, default, or short (`< 32` bytes) `JWT_SECRET_KEY` values fail fast.
- Wildcard `CORS_ORIGINS` is rejected in production.
- `/docs`, `/redoc`, and `/openapi.json` are disabled automatically when `ENVIRONMENT=production`.

---

## 3. Starting the Stack & Initial Admin Provisioning

1. Validate Compose manifests before starting:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml config -q
   ```
2. Build and launch services:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
   ```
3. Run database migrations (also invoked by `backend/docker-entrypoint.sh`):
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml exec backend alembic upgrade head
   ```
4. Provision the first `ADMIN` user via CLI (no hardcoded admin credentials ship in migrations):
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml exec backend \
     python -m backend.app.cli create-admin --email admin@example.com --password "<strong-password>"
   ```

---

## 4. HTTPS TLS Termination (Host Reverse Proxy)

`docker-compose.prod.yml` binds Nginx solely to `127.0.0.1:3000` so external traffic must terminate TLS on a host proxy before reaching `frontend`:

Example Caddy configuration (`/etc/caddy/Caddyfile`):

```caddyfile
documind.example.com {
    reverse_proxy 127.0.0.1:3000 {
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-Proto {scheme}
    }
}
```

Ensure host firewall rules (`ufw` / security groups) expose only TCP `80` and `443` publicly. Keep `5432`, `6379`, `8000`, `5555`, and `3000` closed to external networks.

---

## 5. Resource Sizing Guidelines

`docker-compose.prod.yml` configures initial limits:
- `postgres`: `1.5` CPU / `1G` RAM
- `redis`: `0.5` CPU / `512M` RAM
- `worker`: `${WORKER_CPU_LIMIT:-2.0}` CPU / `${WORKER_MEMORY_LIMIT:-4G}` RAM (increase if enabling LayoutLMv3 transformer weights alongside Tesseract OCR).

Adjust `worker` concurrency (`--concurrency=2` or `--concurrency=1` for GPU/CPU LayoutLMv3 inference) based on host memory so PyTorch/Tesseract child processes do not OOM.

---

## 6. Backup & Restore Procedures

### Database Backup (`pg_dump`)
```bash
mkdir -p backups
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T postgres \
  pg_dump -U "${POSTGRES_USER:-documind}" -Fc "${POSTGRES_DB:-documind}" > "backups/documind_$(date +%F_%H%M).dump"
```

### Document Storage Backup
Back up the persistent upload volume (`storage/uploads` / `uploads_data`) alongside each database snapshot so uploaded PDFs/images and PostgreSQL rows remain consistent:
```bash
tar -czf "backups/uploads_$(date +%F_%H%M).tar.gz" storage/uploads
```

### Database Restore (`pg_restore`)
Stop write traffic (`backend`, `worker`, `beat`), restore into PostgreSQL, and restart services:
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml stop backend worker beat
cat backups/documind_latest.dump | docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T postgres \
  pg_restore -U "${POSTGRES_USER:-documind}" -d "${POSTGRES_DB:-documind}" --clean --if-exists
docker compose -f docker-compose.yml -f docker-compose.prod.yml start backend worker beat
```

---

## 7. Operational Recovery Runbook

1. **Redis Broker Outage / Recovery**:
   - Uploads uploaded during a brief Redis outage remain durably persisted in PostgreSQL (`ProcessingJob` + `OutboxEvent`).
   - Once Redis recovers, Celery Beat triggers stale-outbox recovery automatically, or an Admin can click **Run Outbox & Stale Recovery** in **Admin & Jobs Ops** (`POST /api/v1/jobs/recover-stale`).
   - API rate limiting fails closed with HTTP `503` in production while Redis is unreachable; recovery resumes normal operation without restarting FastAPI.
2. **Stale Running Jobs (Worker Crash / OOM)**:
   - Jobs stuck in `PROCESSING` past `JOB_STALE_TIMEOUT_SECONDS` are reset/recovered by Beat (`recover_stale_jobs_and_outbox_sync`) or manual admin trigger.
3. **Importing Validated LayoutLMv3 Checkpoints**:
   - Validate external `.zip` / checkpoint bundles against Zip-Slip and missing config/model files before worker activation using `ml/training/import_checkpoint.py`.

