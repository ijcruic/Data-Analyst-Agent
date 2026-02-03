# System Test Environment

Local docker-compose stack for exercising the web UI, agent service, and MCP server end to end.

## Directory Layout

```
tests/
├── docker-compose.yml
├── data/
│   ├── customer_churn_dataset-training-master.csv
│   └── customer_churn_dataset-testing-master.csv
└── scripts/
    ├── run_e2e.sh
    ├── smoke_test.py
    └── wait_for_health.py
```

- `docker-compose.yml` builds the services from the project root.
- `data/` holds sample datasets mounted into the containers at runtime.
- `scripts/` contains helper tooling for automated smoke tests.

## Manual Workflow

```bash
cd tests
docker-compose up --build
```

After the services report ready:

1. Visit `http://localhost:8000` for the Streamlit UI.
2. Inspect MinIO via `http://localhost:9001` (login `minioadmin` / `minioadmin`).
3. Tear down the stack with `docker-compose down`.

## Smoke Test

The `scripts/run_e2e.sh` helper mirrors the production flow:

- Waits for all services to become healthy.
- Uploads TSV and JSON files directly to MinIO using the same credentials the UI will use in production.
- Calls MCP tools (`register_external_data`, `list_available_files`, `get_file_schema`, `analyze_session_data`) to verify multi-format loading.
- (Optional) Extend the script to exercise `publish_artifact` once new outputs are generated.

> Prerequisites: create a virtual environment and install `requests` + `boto3` before running the script.
