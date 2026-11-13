#!/usr/bin/env sh
set -eu

# Ensure storage directory exists when bind-mounted
mkdir -p "${UPLOAD_DIR:-/app/storage/uploads}"

if [ "${RUN_MIGRATIONS:-false}" = "true" ]; then
    echo "Running Alembic database migrations..."
    alembic -c backend/alembic.ini upgrade head
fi

exec "$@"

