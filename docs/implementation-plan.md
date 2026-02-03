# Data Analyst Agent – Implementation Plan

## 1. Current State Overview
- **UI (`web-ui/app.py`)** uploads CSV files directly to `/tmp/uploads/<session_id>` and interacts with the agent service over HTTP. Chat history renders text responses and base64 plots inline.
- **Agent service (`agent-service/`)** relies on environment variables for runtime configuration, instantiates a LangGraph ReAct agent, and assumes CSV-only tooling exposed by the MCP server.
- **MCP server (`mcp-server/server.py`)** exposes five tools focused on CSV workflows. Data access is limited to CSV files on the shared volume, and code execution always preloads data with `pd.read_csv`.
- **Shared configuration** is currently handled only through `.env` variables and Docker compose settings; there is no per-service config file.
- **File handling**: uploads are supported through Streamlit, but there is no mechanism to download generated artifacts or expose images beyond inline base64 blobs.

## 2. Goals and Constraints
1. Introduce a `config.yaml` for each service to capture settings such as telemetry (Matomo), model endpoints, API credentials, and service-specific parameters while retaining environment-variable overrides for deployment flexibility.
2. Introduce an object-storage-backed data layer (MinIO) to replace local shared volumes and cover richer data workflows:
   - ingest uploads beyond CSV (any format pandas can read),
   - support downloads/exports of generated artifacts,
   - make images accessible for inline rendering in Streamlit (via pre-signed URLs or static hosting).
3. Allow the agent to dynamically decide how to load data (e.g., `read_excel`, `read_json`, custom parsing) instead of relying on hardcoded `pd.read_csv` inside MCP tooling.
4. Restructure local test assets (docker compose + sample CSVs) into a dedicated test fixture directory to support automated end-to-end validation.
5. Identify and schedule robustness improvements (logging, error reporting, validation, tests) that raise confidence in production-like scenarios.

## 3. Proposed Work Breakdown

### Phase 0 – Discovery & Architecture
- Document required configuration fields for each service (Matomo IDs, model providers, API keys, storage endpoints).
- Define the MinIO-based storage architecture: tenancy model, bucket layout, session key structure, retention policy, and credential distribution. Capture design details in ADR-0001 and supporting docs.
- Confirm security and compliance expectations for file handling (retention, size limits, content scanning) to inform design.

### Phase 1 – Configuration System
- Define `config.yaml` schema per service (base structure + optional overrides). Intent is to have the important environment variables in the config so that when we move the services from development to production, we should only have to update the configs.
- Implement lightweight config loaders:
  - Parse YAML at startup, merge with environment variables.
  - Provide validation and defaults.
- Update Docker compose and service entrypoints to mount or copy configs and surface configuration errors early.
- Document the configuration contract in `README`/new docs section.

### Phase 2 – Data Storage & Transfer Enhancements
- ✅ Provisioned MinIO for local development via docker-compose; production deployment will rely on hosted MinIO (documented in ADR-0001).
- ✅ Built shared storage client utilities in the MCP server (`storage.py`) with filesystem fallback and MinIO support.
- ✅ Refactored MCP tools to surface file metadata and generate pre-signed download URLs.
- ✅ Updated Streamlit UI to upload through MinIO, surface signed download links, and maintain session-scoped asset metadata.
- ✅ Implemented automatic cleanup helper aligned with session TTL (via `cleanup_expired_sessions` Python helper).

### Phase 3 – Dynamic Data Loading in MCP
- Introduce a dedicated MCP tool (e.g. `register_external_data`) that downloads user-provided URLs into per-session temp storage and returns local paths + metadata.
- Refactor `analyze_session_data` to accept those local paths (or legacy filenames) and optional loader hints for pandas.
- Update agent prompts/tool descriptions so the model first registers URLs and then calls `analyze_session_data` with the returned paths.
- UI enhancements:
  - Surface pre-signed upload URLs in chat history.
  - Allow the user (or agent) to pass those links back to the agent for registration.
- Harden sandboxing: restrict builtins, limit filesystem exposure, and guard against long-running code. Handles keep the agent code agnostic to storage details.

### Phase 4 – UI & Agent Updates
- Reflect new tool signatures and storage locations in agent logic (prompting, error handling, schema retrieval).
- Adapt UI workflows:
  - Broaden upload accept types (Excel, JSON, Parquet, images?).
  - Display richer metadata returned from MCP.
  - Provide shareable pre-signed URLs immediately after upload.
  - Treat returned analysis markdown (plots, download links) as first-class chat content.
  - Encourage use of `publish_artifact` so generated files are shareable via markdown links.
- Add user feedback for unsupported formats or failed conversions.
- Surface Matomo analytics configuration (site id, endpoint, SSL verify toggle) through the new UI config loader and ensure events respect the closed-network deployment constraints.

### Phase 5 – Robustness & QA
- Add structured logging and request tracing across services.
- Implement automated tests:
  - Unit tests for config parsing and storage helpers.
  - Integration tests simulating end-to-end analysis on multiple file types.
- Improve error surfacing to the UI (clear messages, recovery guidance).
- Update documentation (setup, config, troubleshooting) and ensure sample datasets cover new formats.
- ✅ Relocated local docker-compose orchestration plus sample datasets into `tests/` for end-to-end validation (includes MinIO startup).
- Add automated coverage for:
  - register→analyze workflow using remote URLs.
  - Artifact downloads / markdown rendering.

## 4. Confirmed Constraints & Decisions
- **Deployment target:** Kubernetes cluster running in a closed network; development occurs locally on an open network. Plan assumes Helm/Kustomize-friendly configs and storage options compatible with both environments.
- **Data retention:** Uploaded data and generated artifacts are session-scoped only; clean up once the session ends or the user leaves. No cross-session persistence required.
- **Data size/compliance:** No additional constraints beyond technical limits; we can focus on operational safeguards (timeouts, size caps) in code.
- **Access control:** No authentication/authorization requirements for downloads at this stage; keep design modular so auth can be added later.
- **Analytics:** Matomo is the required telemetry provider. Expect config keys such as `matomo_site_id: 145`, `matomo_url: "https://matomo.tools.gap.ic.gov/matomo.php"`, `matomo_ssl_verify: False`. Config loaders should validate and surface these values.

## 5. Next Steps
1. Prepare end-to-end tests covering multi-format uploads and MinIO-backed workflows (tests harness scaffolding complete; expand smoke test coverage).
2. Integrate the cleanup tool into automated housekeeping (cronjob or UI trigger) for production deployments.
3. Document loader_hint usage with sample prompts/tests so agents consistently apply the new capability.
