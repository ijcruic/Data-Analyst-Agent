# Configuration Schema Proposal

Phase 0 catalog of configuration inputs for the **web UI**, **agent service**, and **MCP server**. Each service will load defaults from `config.yaml` while permitting environment-variable overrides for deployment-specific values.

---

## Web UI (`web-ui/config.yaml`)

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `agent.api_url` | string | `http://localhost:8002` | Remote endpoint for `/analyze`. Override via `AGENT_URL`. |
| `uploads.allowed_extensions` | list[string] | `["csv", "xlsx", "json", "parquet"]` | Controls uploader filter. |
| `uploads.max_size_mb` | int | `50` | Upper bound for single file upload. |
| `uploads.storage_root` | string | `/tmp/uploads` | Local scratch directory (UI may use for buffering before pushing to MinIO). |
| `sessions.ttl_seconds` | int | `3600` | Used for automatic cleanup of session artifacts. |
| `storage.minio.endpoint` | string | `http://localhost:9000` | Local dev MinIO endpoint; override with hosted endpoint in prod. |
| `storage.minio.region` | string | `us-east-1` | Region-style label for MinIO. |
| `storage.minio.access_key` | string | `minioadmin` | Default dev credential; override via secrets for hosted service. |
| `storage.minio.secret_key` | string | `minioadmin` | Default dev credential; override via secrets for hosted service. |
| `storage.minio.bucket` | string | `data-analyst` | Bucket containing session artifacts. |
| `storage.minio.secure` | bool | `false` | Toggle HTTPS when pointing at hosted MinIO. |
| `storage.minio.presign_expiry_seconds` | int | `600` | Duration for signed download URLs. |
| `telemetry.matomo.site_id` | int | `145` | Required Matomo telemetry identifier. |
| `telemetry.matomo.url` | string | `"https://matomo.tools.gap.ic.gov/matomo.php"` | Endpoint reachable from cluster. |
| `telemetry.matomo.ssl_verify` | bool | `false` | Closed-network clusters may require skipping SSL verification. |
| `ui.theme` | string | `light` | Optional Streamlit theme override. |

Derived environment variables:
- `AGENT_URL`
- `WEB_UI_AGENT_URL`
- `UPLOAD_ROOT`
- `WEB_UI_UPLOAD_ROOT`
- `WEB_UI_MINIO_ENDPOINT`, `WEB_UI_MINIO_ACCESS_KEY`, `WEB_UI_MINIO_SECRET_KEY`, `WEB_UI_MINIO_BUCKET`, `WEB_UI_MINIO_SECURE`
- `MATOMO_SITE_ID`, `MATOMO_URL`, `MATOMO_SSL_VERIFY` (if we mirror keys individually)

Additional considerations:
- Toggle to control whether downloadable artifacts appear (`downloads.enabled`).
- Optional flag to expose debug information in development (`debug.enabled`).
- Defaults target local development (MinIO running in docker-compose). In production, override `storage.minio.*` fields to point at the hosted MinIO service and enable TLS via `secure: true`.
- If MinIO credentials are omitted, the UI falls back to storing files on the local filesystem (`uploads.storage_root`).

---

## Agent Service (`agent-service/config.yaml`)

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `server.port` | int | `8002` | Mirrors `AGENT_PORT`. |
| `server.host` | string | `0.0.0.0` | Useful when deploying behind service mesh. |
| `llm.provider` | enum | `"openai"` | OpenAI-compatible backend identifier. |
| `llm.api_base` | string | `https://api.openai.com/v1` | Base URL for the OpenAI-compatible API endpoint. |
| `llm.model` | string | `"gpt-4o"` | Override via `OPENAI_MODEL`. |
| `llm.temperature` | float | `0.7` | Mirrors `AGENT_TEMPERATURE`. |
| `llm.api_key` | string | _none_ | Required; fallback to `OPENAI_API_KEY`. Consider support for K8s secrets. |
| `mcp.endpoints` | list[string] | `["http://mcp-server:4242/mcp"]` | Supersedes `MCP_URL` comma list. The agent connects to *all* listed endpoints. |
| `mcp.connect_timeout_seconds` | int | `30` | Controls retries. |
| `mcp.max_retries` | int | `5` | Align with current logic. |
| `storage.minio.endpoint` | string | `http://localhost:9000` | Local dev MinIO endpoint; override with hosted endpoint. |
| `storage.minio.access_key` | string | `minioadmin` | Dev credential; override via secrets. |
| `storage.minio.secret_key` | string | `minioadmin` | Dev credential; override via secrets. |
| `storage.minio.bucket` | string | `data-analyst` | Shared bucket for session artifacts. |
| `storage.minio.secure` | bool | `false` | Use `true` for hosted HTTPS MinIO. |
| `telemetry.matomo` | object | same keys as web UI | Enables consistent analytics. |
| `logging.level` | string | `INFO` | Logging verbosity. |
| `logging.json` | bool | `false` | Toggle JSON logs; set to `true` when shipping to centralized logging. |

Environment variable mapping:
- `AGENT_PORT`, `OPENAI_MODEL`, `OPENAI_API_KEY`, `OPENAI_API_BASE`, `AGENT_TEMPERATURE`, `MCP_URL`.
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`, `MINIO_SECURE`.
- Future: `MATOMO_SITE_ID`, etc.

Security considerations:
- Encourage referencing API keys via K8s secrets; config loader should support `from_env` or file references.

---

## MCP Server (`mcp-server/config.yaml`)

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `server.host` | string | `0.0.0.0` | Bind address. |
| `server.port` | int | `4242` | Expose via service/ingress in Kubernetes. |
| `storage.backend` | enum | `"minio"` | Primary backend; set to `"filesystem"` for legacy/local-only workflows. |
| `storage.filesystem.root` | string | `/tmp/uploads` | Legacy local storage path used until MinIO support lands. |
| `storage.minio.endpoint` | string | `http://localhost:9000` | Local dev endpoint; override with hosted service. |
| `storage.minio.region` | string | `us-east-1` | Optional region label. |
| `storage.minio.access_key` | string | `minioadmin` | Dev credential; override via secrets. |
| `storage.minio.secret_key` | string | `minioadmin` | Dev credential; override via secrets. |
| `storage.minio.bucket` | string | `data-analyst` | Bucket containing session prefixed objects. |
| `storage.minio.secure` | bool | `false` | Use `true` for hosted HTTPS MinIO. |
| `storage.minio.session_prefix` | string | `sessions/` | Prefix for per-session artifacts. |
| `storage.minio.presign_expiry_seconds` | int | `600` | Signed URL validity window. |
| `storage.minio.public_endpoint` | string | `http://localhost:9000` | Optional public URL used when generating presigned links. |
| `cleanup.ttl_seconds` | int | `3600` | Align with session-only retention policy. |
| `cleanup.interval_seconds` | int | `300` | Background janitor frequency. |
| `execution.timeout_seconds` | int | `180` | Matches current sandbox limit. |
| `execution.memory_limit_mb` | int | `512` | Document enforcement mechanism. |
| `execution.allowed_modules` | list[string] | `["pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn"]` | For sandbox policy. |
| `telemetry.matomo` | object | same as other services | Optional server-side analytics. |
| `results.cache_enabled` | bool | `true` | Controls in-memory result retention. |
| `results.max_entries` | int | `100` | Guard against unbounded memory use. |

Environment variables to support:
- `MCP_PORT`, `MCP_HOST`.
- Filesystem root override (`UPLOAD_ROOT`) for backward compatibility.
- MinIO connection parameters (`MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_BUCKET`, etc.).

Operational notes:
- Local development currently relies on filesystem storage; MinIO service will be added to docker-compose in Phase 2 while production points to the hosted MinIO endpoint.
- Need a mechanism to signal session cleanup externally (e.g., called by web-ui).

---

## Shared Configuration Patterns

1. **Loading order:** defaults → `config.yaml` → environment variables → command-line flags (if any).
2. **Validation:** Schema using `pydantic` or `voluptuous` per service; fail fast with descriptive errors.
3. **Secrets:** Encourage referencing sensitive values via environment variables or mounted secret files; avoid committing to Git.
4. **Matomo analytics:** Provide a shared helper to read Matomo settings from config/environment so instrumentation stays consistent.
5. **Kubernetes deployment:** Expect config files to be mounted via ConfigMaps; secrets via Secret volumes or environment variables.

---

## Action Items

1. Implement config loader utilities per service (Phase 1).
2. Update Docker compose and sample configs to demonstrate overrides.
3. Document configuration usage in `README` once loaders are in place.
