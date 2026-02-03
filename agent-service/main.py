"""
FastAPI service exposing the data analyst agent over HTTP.

This module is the "agent-service" entrypoint. It wraps the DataAnalystAgent
defined in agent.py and provides a simple JSON API that the Streamlit UI
(web-ui/app.py) can call:

- GET  /health   → basic health check
- POST /analyze  → run a natural‑language analysis request
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import get_agent, reset_agent
from config import load_config

# ============================================================================
# Configuration
# ============================================================================

CONFIG = load_config()
SERVER_HOST = CONFIG.get("server", {}).get("host", "0.0.0.0")
SERVER_PORT = int(CONFIG.get("server", {}).get("port", 8002))
LOG_LEVEL = CONFIG.get("logging", {}).get("level", "info")

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Data Analysis Agent", version="1.0")

# ============================================================================
# Pydantic Models
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request body for the /analyze endpoint."""
    query: str
    session_id: str

class AnalysisResponse(BaseModel):
    """Response body for the /analyze endpoint."""
    text: str
    plot_base64: Optional[str] = None
    result_id: str
    artifacts: Optional[List[Dict[str, Any]]] = None

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """
    Returns a health check response.

    This endpoint can be used to check if the service is running.
    """
    return {"status": "healthy"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """
    Analyzes data based on a user query.

    Args:
        request: The analysis request.

    Returns:
        The analysis response.
    """
    try:
        # Generate a unique ID for the result
        result_id = str(uuid.uuid4())[:8]

        # Get the agent
        agent = await get_agent()

        # Analyze the data
        result = await agent.analyze(request.query, request.session_id)
        artifacts = result.get("artifacts") or []
        normalized_artifacts: List[Dict[str, Any]] = []
        for entry in artifacts:
            entry_copy = dict(entry)
            proxy_path = entry_copy.get("proxy_path")
            if proxy_path and not proxy_path.startswith("/"):
                entry_copy["proxy_path"] = f"/{proxy_path}"
            normalized_artifacts.append(entry_copy)
        # Return the analysis response
        return AnalysisResponse(
            text=result.get("text", ""),
            plot_base64=result.get("plot"),
            result_id=result_id,
            artifacts=normalized_artifacts,
        )
    except (asyncio.TimeoutError, TimeoutError) as e:
        print(f"⏱️ Timeout error during analysis: {str(e)}")
        print("🔄 Resetting agent due to timeout...")
        # Reset the agent to allow recovery from timeout state
        reset_agent()
        raise HTTPException(
            status_code=504,
            detail="Analysis request timed out. Please try again."
        )
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/artifacts/{session_id}/{artifact_id}")
async def download_artifact(session_id: str, artifact_id: str):
    """
    Serve cached artifacts fetched from the MCP server.
    """
    agent = await get_agent()
    metadata = agent.get_cached_artifact_metadata(session_id, artifact_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Artifact not found")

    path = metadata.get("path")
    if not path:
        raise HTTPException(status_code=404, detail="Artifact missing path")

    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")

    mime_type = metadata.get("mime_type") or "application/octet-stream"
    filename = metadata.get("filename") or file_path.name
    
    return FileResponse(
        file_path,
        media_type=mime_type,
        filename=filename,
    )

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print(f"🚀 Starting Data Analysis Agent on {SERVER_HOST}:{SERVER_PORT}...")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level=LOG_LEVEL.lower())
