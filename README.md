# Agentic AI Data Analyst

A functional data analyst powered by LLM reasoning and dynamic code execution.

## Quick Start (3 Steps)

### 1. Setup Environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Start Services
```bash
# From the repo root
cd tests
docker-compose up --build
# Wait for: "✅ All services ready"
```

This launches:
- Streamlit UI (port `8000`)
- Agent service (port `8002`)
- MCP server (port `4242`)
- **MinIO object storage** (API `9000`, console `9001`) with default credentials `minioadmin / minioadmin`

### 3. Open Browser
```
http://localhost:8000
```

**That's it!** Upload a CSV and start analyzing.

---

## System Architecture

- **UI** (port 8000): Streamlit web interface with a chat interface.
- **Agent** (port 8002): FastAPI service with LangGraph reasoning.
- **MCP** (port 4242): Data analysis tools with dynamic Python execution.

### System Diagram

```mermaid
flowchart LR
    U[User Browser] -->|upload file / prompt| UI[Streamlit UI]
    UI -->|session upload URL| MinIO[(MinIO Object Storage)]
    MinIO -->|presigned dataset URL| UI
    UI -->|chat request + URLs| Agent[Agent Service]
    Agent -->|register_external_data / analyze| MCP[MCP Server]
    MCP -->|download dataset| Cache[/tmp/mcp_external_cache/<session>]
    MCP -->|agent code writes plots/files| Outputs[/tmp/mcp_outputs/<session>]
    MCP -->|publish_artifact| Agent
    Agent -->|cache artifact| AgentArtifacts[/tmp/agent_artifacts/<session>]
    Agent -->|structured response + links| UI
    UI -->|render answers/plots| U
```

All services communicate via HTTP/JSON with session-based isolation.

### Data Workflow

1. **Upload** – The Streamlit UI streams user files directly into MinIO under `sessions/<session_id>/…` and surfaces presigned download URLs in the chat sidebar.
2. **Registration** – The agent passes those URLs to the MCP server’s `register_external_data` tool; the MCP server downloads each asset into `/tmp/mcp_external_cache/<session>` and returns concrete filesystem paths plus metadata.
3. **Analysis** – The agent crafts Python for `analyze_session_data`, referencing the cached paths. The code executes inside the MCP sandbox, which exposes `paths` and an `output_dir` (`/tmp/mcp_outputs/<session>`).
4. **Artifact Publishing** – When the agent saves derived datasets or images, it calls `publish_artifact`. The MCP server exposes the artifact as an MCP resource; the agent service immediately fetches the bytes, caches them under `/tmp/agent_artifacts/<session>`, and records a proxy URL.
5. **Response Delivery** – The agent packages the analysis summary and artifact metadata back to the UI. Streamlit rewrites any `resource://` links to the agent proxy endpoint, renders plots inline, and offers download buttons backed by the FastAPI `/artifacts/...` route.
6. **Cleanup** – Operational scripts can invoke the `cleanup_expired_sessions` helper to purge both `/tmp/mcp_external_cache/<session>` and `/tmp/mcp_outputs/<session>`, as well as the corresponding MinIO session objects, after the TTL expires.

## MCP Tools

| Tool | Purpose |
|------|---------|
| `list_available_files(session_id)` | Discover uploaded session assets (name/size/links) |
| `get_file_schema(filename, session_id)` | Inspect data structure |
| `register_external_data(assets, session_id)` | Cache external dataset URLs and return local paths + metadata |
| `analyze_session_data(paths, code, session_id)` | Execute analysis code across cached datasets using the provided paths |
| `publish_artifact(path, session_id, mime_type?)` | Register saved outputs as MCP resources for downstream download/rendering |

## Directory Structure

```
├── README.md
├── .env.example
├── docker-compose.yml
│
├── mcp-server/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── server.py          # 5 core tools
│
├── agent-service/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── agent.py           # LangGraph + MCP integration
│   └── main.py            # FastAPI wrapper
│
└── web-ui/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py             # Streamlit interface
```

## Requirements

- **Docker** + Docker Compose
- **OpenAI API key** (set in .env)
- **4GB+ RAM** recommended

## Configuration

Each service now loads its runtime settings from a local `config.yaml`:

| Service | Config file | Notes |
|---------|-------------|-------|
| Web UI | `web-ui/config.yaml` | Controls agent endpoint, upload limits, Streamlit options, and MinIO connection info. |
| Agent service | `agent-service/config.yaml` | Defines LLM model/temperature/base URL/API key, MCP endpoints, logging, and MinIO credentials. |
| MCP server | `mcp-server/config.yaml` | Sets server host/port, storage backend, execution limits, and Matomo telemetry. |

Environment variables override the YAML values at runtime (see `docs/configuration-schema.md` for the full key mapping). For local development the defaults point to `http://localhost` services. When deploying to the hosted Kubernetes cluster, mount your production-ready configs (or use env vars) to supply the managed MinIO endpoint and Matomo site settings.

## Data Storage

- Uploaded datasets are stored in MinIO under `sessions/<session_id>/<filename>` (docker-compose provisions a local MinIO instance).
- The Streamlit UI surfaces signed download links (default expiry 10 minutes). If MinIO is unavailable, uploads fall back to the local filesystem.
- Kubernetes deployments should target the hosted MinIO service described in ADR-0001; configure credentials via secrets.
- Session data is ephemeral. Use the `cleanup_expired_sessions` helper in `mcp-server/server.py` (e.g., via CronJob/Helm hook) to purge MCP cache directories and invoke storage cleanup. The agent-service can periodically clear `/tmp/agent_artifacts/<session>` using the same session identifiers if long-term retention is not required.

## Code Sandboxing

Execution environment includes only safe libraries:
- pandas, numpy, scipy
- scikit-learn
- matplotlib, seaborn

Limits: 180-second timeout, 512MB memory

## Troubleshooting

**Port already in use:**
```bash
docker-compose down
docker-compose up --build
```

**Agent won't connect:**
- Wait 30 seconds for startup
- Check: `docker-compose ps`
- Check: `docker-compose logs agent-service`

**File upload fails:**
```bash
mkdir -p /tmp/uploads
chmod 777 /tmp/uploads
```

**API key error:**
```bash
# Verify .env has correct format
cat .env
# Should be: OPENAI_API_KEY=sk-xxxxx
# NOT: OPENAI_API_KEY="sk-xxxxx"
```

**MinIO console:**
```
http://localhost:9001  # login with minioadmin / minioadmin
```

---

## Documentation

- **SETUP_AND_DEMO.md** – Complete setup and interactive walkthrough (start here)
- **docs/implementation-plan.md** – Roadmap and phased deliverables
- **docs/configuration-schema.md** – Detailed configuration keys and overrides
- **docs/adr/0001-storage-backend.md** – Storage architecture decision (filesystem → MinIO)
- **docs/operations-notes.md** – Cleanup automation and loader-hint guidance
- **tests/README.md** – Docker compose instructions for local end-to-end testing

## Testing

- `tests/docker-compose.yml` spins up the full stack against a local MinIO instance (mirroring the hosted service).
- `tests/scripts/run_e2e.sh` uploads sample data via the S3 API, exercises MCP tools (including loader hints), and cleans up session artifacts.

### Manual Workflow

1. Ensure `.env` at the repo root contains a valid OpenAI-compatible API key (and `OPENAI_API_BASE` if you use a proxy).
2. From the repo root:
   ```bash
   cd tests
   docker-compose up --build
   ```
3. When the compose logs show all services ready, browse to `http://localhost:8000`, upload a dataset such as `customer_churn_dataset-training-master.csv`, and chat with the agent.
4. Keep the compose logs visible to monitor interactions; press `Ctrl+C` to stop and run `docker-compose down` to clean up.

Tip: after uploading, the chat sidebar prints the pre-signed dataset URLs. When you ask the agent a question, it will first call `register_external_data` with those URLs and then `analyze_session_data` using the local paths it receives.

When the agent saves derived outputs (CSV, images, etc.) under `/tmp/mcp_outputs/<session>`, it uses `publish_artifact` to return shareable download/display links that Streamlit renders in the chat.

> Optional: create a virtual environment and install `requests` + `boto3` if you plan to run `./scripts/run_e2e.sh` or other helper scripts.

## Teaching Points

✅ Service composition (3 independent microservices)
✅ MCP protocol (tool standardization)
✅ ReAct agent pattern (reasoning + tool calling)
✅ Code sandboxing (safe dynamic execution)
✅ Session management (multi-user isolation)
✅ Docker containerization
✅ Async patterns (Python asyncio)
✅ LLM integration (Claude/GPT)
