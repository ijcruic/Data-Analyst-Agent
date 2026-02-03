#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_DIR="$ROOT_DIR/tests"
DATA_DIR="$COMPOSE_DIR/data"

trap 'cd "$COMPOSE_DIR" && docker-compose down --remove-orphans' EXIT

cd "$COMPOSE_DIR"
docker-compose up --build -d

echo "Waiting for services to report healthy..."
python3 scripts/wait_for_health.py

echo "Running smoke checks..."
python3 scripts/smoke_test.py

echo "All checks passed."
