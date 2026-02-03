"""
Data Analysis MCP Server.

This process implements the "mcp-server" component. It exposes core
tools that the agent-service can call via the Model Context Protocol (MCP)
to analyze uploaded data.

Tools:
- register_external_data(assets, session_id)
- list_available_files(session_id)
- get_file_schema(filename, session_id)
- analyze_session_data(paths, code, session_id)
- publish_artifact(path, session_id, mime_type?)

All tools operate on session-scoped files cached under `/tmp/mcp_external_cache`
and `/tmp/mcp_outputs`, allowing the agent to work with concrete local paths.
"""

from __future__ import annotations

import base64
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, Dict, List
from urllib.parse import urlparse

import requests
from fastmcp import FastMCP
from mcp.types import BlobResourceContents

from config import load_config
from storage import StorageError, build_storage

# ============================================================================
# Configuration
# ============================================================================

CONFIG = load_config()
SERVER_HOST = CONFIG.get("server", {}).get("host", "0.0.0.0")
SERVER_PORT = int(CONFIG.get("server", {}).get("port", 4242))
CLEANUP_CONFIG = CONFIG.get("cleanup", {})
CLEANUP_TTL_SECONDS = int(CLEANUP_CONFIG.get("ttl_seconds", 3600))

try:
    STORAGE = build_storage(CONFIG)
except StorageError as exc:
    raise RuntimeError(f"Failed to initialize storage backend: {exc}") from exc

# Create FastMCP server
mcp = FastMCP("DataAnalystServer", host=SERVER_HOST, port=SERVER_PORT)

# Cache for registered external data: session_id -> list of metadata dicts
_registered_assets: dict[str, list[dict]] = {}
_session_temp_root = Path("/tmp/mcp_external_cache")
_session_temp_root.mkdir(parents=True, exist_ok=True)

_artifact_root = Path("/tmp/mcp_outputs")
_artifact_root.mkdir(parents=True, exist_ok=True)

# Session-scoped index of published artifacts accessible via MCP resources.
# Structure: {session_id: {artifact_name: {"path": str, "mime_type": str, ...}}}
_published_artifacts: dict[str, dict[str, dict[str, Any]]] = {}


def _load_dataframe(local_path: Path, **kwargs):
    import pandas as pd

    ext = local_path.suffix.lower()
    readers = {
        (".csv", ".txt", ".data"): lambda path, **kw: pd.read_csv(path, **kw),
        (".tsv",): lambda path, **kw: pd.read_csv(path, sep="\t", **kw),
        (".xls", ".xlsx"): lambda path, **kw: pd.read_excel(path, **kw),
        (".json",): lambda path, **kw: pd.read_json(path, **kw),
        (".parquet",): lambda path, **kw: pd.read_parquet(path, **kw),
        (".feather", ".ft"): lambda path, **kw: pd.read_feather(path, **kw),
        (".pkl", ".pickle"): lambda path, **kw: pd.read_pickle(path, **kw),
    }

    for exts, reader in readers.items():
        if ext in exts:
            return reader(local_path, **kwargs)

    # Fallback to read_csv for unknown extensions that may still be text/tabular.
    return pd.read_csv(local_path, **kwargs)


def _session_cache_dir(session_id: str) -> Path:
    path = _session_temp_root / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_output_dir(session_id: str) -> Path:
    path = _artifact_root / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _store_registered_asset(
    session_id: str,
    source_url: str,
    filename: str,
    loader_hints: Optional[dict] = None,
) -> dict:
    """Download the remote asset into the session cache and return metadata."""
    session_dir = _session_cache_dir(session_id)
    base_name = filename or urlparse(source_url).path.split("/")[-1] or "dataset"
    safe_name = base_name
    counter = 1
    while (session_dir / safe_name).exists():
        stem = Path(base_name).stem
        suffix = Path(base_name).suffix
        safe_name = f"{stem}_{counter}{suffix}"
        counter += 1

    destination = session_dir / safe_name

    response = requests.get(source_url, stream=True, timeout=60)
    response.raise_for_status()
    with destination.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return {
        "filename": safe_name,
        "path": destination,
        "source_url": source_url,
        "loader_hints": loader_hints or {},
        "registered_at": datetime.now(timezone.utc),
    }


def _find_registered_asset(session_id: str, identifier: str) -> Optional[dict]:
    """Locate cached asset metadata by alias, filename, path, or URL."""
    assets = _registered_assets.get(session_id, [])
    for meta in assets:
        if identifier in (
            meta.get("alias"),
            meta["filename"],
            str(meta["path"]),
            meta["source_url"],
        ):
            return meta
    return None


@mcp.tool()
def register_external_data(assets: List[Dict[str, Any]], session_id: str) -> Dict[str, Any]:
    """Download remote datasets into the session cache.

    Each asset descriptor should at minimum supply a `url`. Optional fields
    include `filename` (to override the cached name) and `loader_hints`
    (metadata passed back to callers but not otherwise interpreted).

    Returns a dictionary with `datasets` (metadata for each cached file,
    including its absolute `path`) and `errors` (any download failures).
    """
    if not assets:
        return {"datasets": [], "errors": []}

    datasets = []
    errors = []

    session_assets = _registered_assets.setdefault(session_id, [])

    for asset in assets:
        url = asset.get("url")
        if not url:
            errors.append({"error": "Missing url field", "asset": asset})
            continue

        filename = asset.get("filename") or urlparse(url).path.split("/")[-1] or "dataset"
        loader_hints = asset.get("loader_hints")

        try:
            metadata = _store_registered_asset(session_id, url, filename, loader_hints)
            alias = str(len(session_assets) + 1)
            metadata["alias"] = alias
            session_assets.append(metadata)
            datasets.append(
                {
                    "alias": alias,
                    "path": str(metadata["path"]),
                    "filename": metadata["filename"],
                    "source_url": metadata["source_url"],
                    "loader_hints": metadata["loader_hints"],
                }
            )
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})

    return {"datasets": datasets, "errors": errors}


@mcp.tool()
def list_available_files(session_id: str) -> List[Dict[str, Any]]:
    """Return metadata about all datasets cached for a session.

    Args:
        session_id: Unique identifier for the chat/session.

    Returns:
        List of dictionaries describing each cached dataset (alias, filename,
        absolute path, source URL, registration timestamp, loader hints, and
        a user-facing download URL).
    """
    session_dir = _session_cache_dir(session_id)
    assets = _registered_assets.get(session_id, [])
    result: list[dict[str, Any]] = []

    for meta in assets:
        path_obj = Path(meta["path"])
        if not path_obj.exists():
            continue
        result.append(
            {
                "alias": meta["alias"],
                "filename": meta["filename"],
                "path": str(path_obj),
                "source_url": meta["source_url"],
                "registered_at": meta["registered_at"].isoformat(),
                "loader_hints": meta["loader_hints"],
                "download_url": str(path_obj),
            }
        )

    # Include any files that might exist in the cache directory even if the
    # in-memory metadata was lost (e.g., server restart mid-session).
    for file_path in session_dir.iterdir():
        if not file_path.is_file():
            continue
        existing = next((item for item in result if item["path"] == str(file_path)), None)
        if existing:
            continue
        result.append(
            {
                "alias": None,
                "filename": file_path.name,
                "path": str(file_path),
                "source_url": None,
                "registered_at": None,
                "loader_hints": {},
                "download_url": str(file_path),
            }
        )

    return result


@mcp.tool()
def get_file_schema(filename: str, session_id: str) -> Dict[str, Any]:
    """Inspect a cached dataset and return column/sample metadata.

    Args:
        filename: Alias, cached filename, local path, or original URL referencing
            the dataset to inspect.
        session_id: Unique identifier for the chat/session.

    Returns:
        Dictionary containing columns, dtypes, shape, sample rows, and a
        download URL. Returns an `error` entry if the dataset cannot be
        resolved.
    """
    metadata = _find_registered_asset(session_id, filename)

    local_path: Optional[Path] = None

    if metadata is not None:
        local_path = Path(metadata["path"])
    else:
        candidate = Path(filename)
        if not candidate.is_absolute():
            candidate = (_session_cache_dir(session_id) / candidate).resolve()
        try:
            session_root = _session_cache_dir(session_id).resolve()
            candidate = candidate.resolve()
            if session_root not in candidate.parents and candidate != session_root:
                return {"error": f"Access to path outside session cache is not allowed: {candidate}"}
        except Exception:
            return {"error": f"Invalid path: {filename}"}
        local_path = candidate

    if not local_path.exists():
        return {"error": f"File not found: {local_path}"}

    try:
        df = _load_dataframe(local_path)
    except Exception as exc:
        return {"error": f"Failed to load {local_path}: {exc}"}

    response = {
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": list(df.shape),
        "sample": df.head(3).to_dict(orient="records"),
        "download_url": str(local_path),
    }

    if metadata is not None:
        response["source_url"] = metadata["source_url"]
        response["loader_hints"] = metadata.get("loader_hints", {})

    return response


@mcp.tool()
def analyze_session_data(
    paths: List[str],
    code: str,
    session_id: str,
) -> Dict[str, Any]:
    """Execute agent-supplied Python code in a sandbox.
    
    The code can use pandas, numpy, and other libraries. To return a result
    directly to your thought process, assign it to a variable named `result`.
    To create a file, save it to the provided `output_dir`. The `paths` list
    contains the absolute paths to the session's datasets.
    """

    if not paths:
        return {"error": "No dataset paths specified for analysis."}

    import pandas as pd
    import numpy as np
    import scipy as sp
    import sklearn
    import matplotlib
    matplotlib.use("Agg")  # ensure headless rendering remains safe if used
    import matplotlib.pyplot as plt
    import seaborn as sns

    resolved_paths: List[str] = []
    for identifier in paths:
        meta = _find_registered_asset(session_id, identifier)
        if meta is not None:
            path_obj = Path(meta["path"])
        else:
            candidate = Path(identifier)
            if not candidate.is_absolute():
                candidate = (_session_cache_dir(session_id) / candidate)
            path_obj = candidate.expanduser().resolve()
            session_root = _session_cache_dir(session_id).resolve()
            if session_root not in path_obj.parents and path_obj != session_root:
                return {"error": f"Path {path_obj} is outside the allowed session cache."}

        if not path_obj.exists():
            return {"error": f"Data file not found: {path_obj}"}

        resolved_paths.append(str(path_obj))

    output_dir = _session_output_dir(session_id)

    local_vars = {
        "paths": resolved_paths,
        "output_dir": str(output_dir),
        "pd": pd,
        "np": np,
        "sp": sp,
        "sklearn": sklearn,
        "plt": plt,
        "sns": sns,
    }

    try:
        exec(code, {}, local_vars)
        result = local_vars.get("result", "✅ Code ran but no variable `result` was set.")
        return {"text": str(result)}
    except Exception as exc:
        return {"error": f"Code execution failed: {exc}"}
    finally:
        plt.close("all")


@mcp.tool()
def publish_artifact(
    session_id: str,
    path: str,
    mime_type: str = "application/octet-stream",
) -> Dict[str, Any]:
    """Expose a local file as an MCP resource so downstream clients can retrieve it.

    Args:
        session_id: Unique identifier for the chat/session that owns the file.
        path: Absolute path or path relative to `/tmp/mcp_outputs/<session>`
            pointing at the file to publish.
        mime_type: MIME type hint recorded with the resource metadata.

    Returns:
        Dictionary containing the canonical internal path (`path`), original
        filename, MIME type, size in bytes, and the `resource://` URI that
        callers can pass back to the MCP client.
    """

    output_dir = _session_output_dir(session_id)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    candidate = candidate.resolve()

    if not candidate.exists():
        return {"error": f"Artifact not found: {candidate}"}

    if output_dir not in candidate.parents and candidate.parent != output_dir:
        dest = output_dir / candidate.name
        shutil.copy2(candidate, dest)
        candidate = dest

    artifact_name = candidate.name
    session_records = _published_artifacts.setdefault(session_id, {})
    if artifact_name in session_records:
        stem = Path(artifact_name).stem
        suffix = Path(artifact_name).suffix
        counter = 1
        unique_name = artifact_name
        while unique_name in session_records:
            unique_name = f"{stem}_{counter}{suffix}"
            counter += 1
        artifact_name = unique_name

    file_stat = candidate.stat()
    metadata = {
        "path": str(candidate),
        "filename": candidate.name,
        "mime_type": mime_type or "application/octet-stream",
        "size": file_stat.st_size,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    session_records[artifact_name] = metadata

    uri = f"resource://artifacts/{session_id}/{artifact_name}"

    response: Dict[str, Any] = {
        "path": str(candidate),
        "filename": metadata["filename"],
        "mime_type": metadata["mime_type"],
        "size": metadata["size"],
        "uri": uri,
    }
    return response


@mcp.resource("resource://artifacts/{session_id}/{artifact_name}")
def get_published_artifact(session_id: str, artifact_name: str) -> BlobResourceContents:
    """Expose published artifacts as MCP resources."""
    session_records = _published_artifacts.get(session_id, {})
    metadata = session_records.get(artifact_name)
    if metadata is None:
        raise FileNotFoundError(f"Artifact not found: {artifact_name}")

    file_path = Path(metadata["path"])
    if not file_path.exists():
        raise FileNotFoundError(f"Artifact file missing: {file_path}")

    data = file_path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")

    return BlobResourceContents(
        uri=f"resource://artifacts/{session_id}/{artifact_name}",
        blob=encoded,
        mimeType=metadata.get("mime_type") or "application/octet-stream",
    )


def cleanup_expired_sessions(
    session_id: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Remove cached inputs/outputs that exceed the retention window.

    This helper is not exposed as an MCP tool; invoke it from operational
    scripts (e.g. on shutdown or via a CronJob) to prune cached data.
    """
    ttl = int(ttl_seconds) if ttl_seconds is not None else CLEANUP_TTL_SECONDS
    now = datetime.now(timezone.utc)
    removed: list[str] = []

    def _prune(session: str) -> None:
        STORAGE.delete_session(session)
        cache_dir = _session_temp_root / session
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        output_dir = _artifact_root / session
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        _registered_assets.pop(session, None)
        _published_artifacts.pop(session, None)

    if session_id:
        _prune(session_id)
        removed.append(session_id)
    else:
        try:
            sessions = STORAGE.list_sessions()
        except Exception as exc:
            return {"error": f"Failed to enumerate sessions: {exc}"}

        for record in sessions:
            age_seconds = (now - record.last_modified).total_seconds()
            if age_seconds >= ttl:
                _prune(record.session_id)
                removed.append(record.session_id)

    return {
        "removed_sessions": removed,
        "ttl_seconds": ttl,
        "timestamp": now.isoformat(),
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Starting Data Analyst MCP Server on 0.0.0.0:4242")
    print("Available tools: register_external_data, list_available_files, get_file_schema, analyze_session_data, publish_artifact")
    mcp.run(transport="streamable-http")
