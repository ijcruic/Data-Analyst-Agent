# Web UI Service

Streamlit frontend that lets analysts upload data, issue natural-language requests, and review results from the agent service.

## Features

- Upload CSV (and other configured) files to seed a session.
- Submit analysis prompts to the agent service and render responses.
- Display generated plots and provide download links for artifacts via the agent proxy.
- Optional integration with MinIO for durable session storage.

## Quick Start

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py --server.port=8000 --server.address=0.0.0.0
```

The application reads runtime options from `config.yaml`. Adjust service URLs, upload constraints, and storage backends there.

## Configuration Highlights

- `agent.api_url`: base URL of the agent-service FastAPI instance.
- `uploads.allowed_extensions`, `uploads.max_size_mb`: file-type and size controls.
- `storage.minio`: enables MinIO sessions when endpoint and credentials are provided.
- `sessions.ttl_seconds`: automatic session clean-up interval.

## Deployment Notes

- Expose port `8000` (or the configured Streamlit port) to reach the UI.
- The Docker image bundles `sources.list` and `pip.conf` to support air-gapped builds.
- Mount persistent storage at `/tmp/uploads` if you need to retain local uploads between restarts when MinIO is disabled.
