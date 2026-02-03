# MCP Server Service

Server process that exposes Model Context Protocol (MCP) tools used by the agent service. It fetches datasets, runs scripted analyses, and publishes artifacts that downstream components can retrieve.

## Overview

- **Protocol:** MCP over HTTP (`fastmcp`).
- **Listen address:** configurable via `config.yaml` (`server.host`, `server.port`) – defaults to `0.0.0.0:4242`.
- **Responsibilities:** ingest uploaded data, execute Python analysis snippets, list cached files, and expose generated artifacts.

## Exposed Tools

| Tool | Description |
|------|-------------|
| `register_external_data(assets, session_id)` | Downloads remote files into the session cache. |
| `list_available_files(session_id)` | Lists datasets registered for a session. |
| `get_file_schema(filename, session_id)` | Describes column schema for a cached file. |
| `analyze_session_data(paths, code, session_id)` | Executes Python analysis code against session files. |
| `publish_artifact(path, session_id, mime_type?)` | Publishes generated artifacts for download. |

Session data and artifacts are stored under `/tmp/mcp_external_cache` and `/tmp/mcp_outputs` inside the container.

## Quick Start

```bash
python3 -m pip install -r requirements.txt
python3 server.py
```

The server loads configuration from `config.yaml` and connects to the configured storage backend (local filesystem or MinIO).

## Configuration Highlights

- `server.host` / `server.port`: network settings for MCP HTTP transport.
- `storage`: location and credentials for artifact and upload persistence.
- `cleanup.ttl_seconds`: TTL for automatic pruning of stale session data.

## Deployment Notes

- Ensure `config.yaml` contains production credentials (e.g., MinIO access keys).
- Expose port `4242` (or the configured value) so the agent can reach the MCP server.
- The Docker image copies `sources.list` and `pip.conf` to support air-gapped deployments.
