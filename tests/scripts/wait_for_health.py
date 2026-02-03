#!/usr/bin/env python3
import os
import time
import requests

SERVICES = {
    "web": "http://localhost:8000",
    "agent": "http://localhost:8002/health",
    "mcp": "http://localhost:4242",
    "minio": "http://localhost:9000/minio/health/live",
}

TIMEOUT = 180
SLEEP = 5

start = time.time()

while True:
    pending = []
    for name, url in SERVICES.items():
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code >= 400:
                pending.append(name)
        except requests.RequestException:
            pending.append(name)

    if not pending:
        print("All services are healthy.")
        break

    if time.time() - start > TIMEOUT:
        raise SystemExit(f"Timed out waiting for services: {pending}")

    print(f"Waiting for services: {pending}")
    time.sleep(SLEEP)
