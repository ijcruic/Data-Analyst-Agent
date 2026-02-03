# Operations Notes

## Session Cleanup

- Use the helper `cleanup_expired_sessions` in `mcp-server/server.py` from operational scripts (CronJob, pre-stop hook) to prune cached inputs/outputs older than `cleanup.ttl_seconds`.
- This helper is no longer exposed as an MCP tool; invoke it via a lightweight Python script that imports and calls the function.
- Ensure `cleanup.ttl_seconds` is aligned across UI/agent/MCP configs so all services share the same retention expectation.

## Artifact Delivery

- The agent-service now downloads MCP `publish_artifact` resources immediately and stores them under `/tmp/agent_artifacts/<session>`. The FastAPI `/artifacts/{session}/{artifact}` endpoint streams those files to the UI.
- Deployments should monitor disk usage for `/tmp/agent_artifacts` and add a scheduled cleanup (e.g., on session expiration) if long-running sessions are expected.
- Since artifacts are served directly by the agent, MinIO presigned URLs are no longer used in responses; UI downloads should reference the FastAPI proxy instead.

## Loader Hint Usage

- Agent prompt now reminds the LLM to include `loader_hints` when calling `analyze_session_data`.
- For evaluations or regression tests, include prompts that explicitly mention non-CSV formats (e.g., TSV requiring a tab separator, Excel sheet selection).
- The `tests/scripts/run_e2e.sh` smoke test uploads TSV/JSON files, calls `register_external_data`, and exercises `analyze_session_data`. Extend this script (or additional tests) to cover other formats and artifact publishing.

## Hosted MinIO

- Production deployments target the managed MinIO service; credentials and endpoints are supplied via Kubernetes Secrets/ConfigMaps.
- Local docker-compose spins up MinIO solely for testing. Ensure runbooks emphasize that production uses the hosted instance and the stack must be configured accordingly.
