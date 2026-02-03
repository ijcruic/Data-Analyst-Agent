#!/usr/bin/env python3
"""
Streamlit Web UI for the Agentic AI Data Analyst.
"""

from __future__ import annotations

import base64
import io
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse

import requests
import streamlit as st
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from config import load_config

# ============================================================================
# Configuration
# ============================================================================

CONFIG = load_config()

AGENT_URL: str = CONFIG.get("agent", {}).get("api_url", "http://localhost:8002")
ALLOWED_EXTENSIONS: list[str] = CONFIG.get("uploads", {}).get("allowed_extensions", ["csv"])
MAX_UPLOAD_MB: int = CONFIG.get("uploads", {}).get("max_size_mb", 50)
UPLOAD_ROOT = Path(CONFIG.get("uploads", {}).get("storage_root", "/tmp/uploads"))
SESSION_TTL_SECONDS: int = CONFIG.get("sessions", {}).get("ttl_seconds", 3600)

STORAGE_CONFIG = CONFIG.get("storage", {})
MINIO_CONFIG = STORAGE_CONFIG.get("minio", {})
MINIO_ENABLED = all(MINIO_CONFIG.get(key) for key in ("endpoint", "access_key", "secret_key", "bucket"))
MINIO_SESSION_PREFIX = MINIO_CONFIG.get("session_prefix", "sessions/").strip("/")
if MINIO_SESSION_PREFIX:
    MINIO_SESSION_PREFIX += "/"
MINIO_PRESIGN_EXPIRY = int(MINIO_CONFIG.get("presign_expiry_seconds", 600))

_MINIO_CLIENT: Optional[Any] = None
_MINIO_INIT_ERROR: Optional[str] = None

def _create_s3_client(config: Dict[str, Any]) -> Any:
    endpoint = config.get("endpoint", "").strip()
    if not endpoint:
        raise ValueError("Missing MinIO endpoint.")

    has_scheme = endpoint.startswith("http://") or endpoint.startswith("https://")
    secure_default = endpoint.startswith("https://")
    secure_flag = bool(config.get("secure", secure_default))

    if has_scheme:
        endpoint_url = endpoint
        use_ssl = endpoint.startswith("https://")
    else:
        use_ssl = secure_flag
        endpoint_url = f"https://{endpoint}" if use_ssl else f"http://{endpoint}"

    return boto3.resource(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
        region_name=config.get("region"),
        use_ssl=use_ssl,
        verify=False,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    ).meta.client


if MINIO_ENABLED:
    try:
        _MINIO_CLIENT = _create_s3_client(MINIO_CONFIG)
    except Exception as exc:  # pragma: no cover
        _MINIO_CLIENT = None
        _MINIO_INIT_ERROR = str(exc)
        MINIO_ENABLED = False

SESSION_ID_KEY = "session_id"


# ============================================================================
# Helpers
# ============================================================================


def _ensure_session_dir(session_id: str) -> Path:
    session_dir = UPLOAD_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _is_allowed_extension(filename: str, allowed: Iterable[str]) -> bool:
    ext = Path(filename).suffix.lower().lstrip(".")
    return ext in {item.lower() for item in allowed}


def _validate_file_size(upload_size_bytes: int) -> bool:
    if MAX_UPLOAD_MB <= 0:
        return True
    size_mb = upload_size_bytes / (1024 * 1024)
    return size_mb <= MAX_UPLOAD_MB


def _minio_object_name(session_id: str, filename: str) -> str:
    return f"{MINIO_SESSION_PREFIX}{session_id}/{filename}" if MINIO_SESSION_PREFIX else f"{session_id}/{filename}"


def _store_file(session_id: str, filename: str, data: bytes, content_type: Optional[str]) -> Optional[str]:
    if MINIO_ENABLED and _MINIO_CLIENT:
        object_name = _minio_object_name(session_id, filename)
        data_stream = io.BytesIO(data)
        data_stream.seek(0)
        try:
            _MINIO_CLIENT.put_object(
                Bucket=MINIO_CONFIG["bucket"],
                Key=object_name,
                Body=data_stream,
                ContentLength=len(data),
                ContentType=content_type or "application/octet-stream",
            )
            return _MINIO_CLIENT.generate_presigned_url(
                "get_object",
                Params={"Bucket": MINIO_CONFIG["bucket"], "Key": object_name},
                ExpiresIn=MINIO_PRESIGN_EXPIRY,
            )
        except ClientError as exc:
            raise RuntimeError(f"Failed to store {filename} in MinIO: {exc}") from exc

    session_dir = _ensure_session_dir(session_id)
    filepath = session_dir / filename
    with filepath.open("wb") as handle:
        handle.write(data)
    return None


def _agent_proxy_url(proxy_path: str) -> str:
    """Build an absolute URL for an artifact served by the agent service."""
    base = AGENT_URL.strip()
    if "://" not in base:
        base = f"http://{base}"
    base = base.rstrip("/")
    if not proxy_path.startswith("/"):
        proxy_path = "/" + proxy_path
    return f"{base}{proxy_path}"


def _resolve_proxy(artifact: Dict[str, Any]) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    proxy_url = artifact.get("proxy_url")
    proxy_path = artifact.get("proxy_path")
    if proxy_url is None and proxy_path:
        proxy_url = _agent_proxy_url(proxy_path)
    if not proxy_url:
        return None, None, None

    try:
        response = requests.get(proxy_url, timeout=20)
        response.raise_for_status()
        content_type = response.headers.get("content-type")
        return response.content, content_type, proxy_url
    except requests.RequestException:
        return None, None, proxy_url


def _render_figure(figure: Any, index: int) -> None:
    if isinstance(figure, str):
        st.image(figure, caption=f"Result {index + 1}")
        return

    if not isinstance(figure, dict):
        st.write(str(figure))
        return

    proxy_binary, proxy_mime, proxy_url = _resolve_proxy(figure)
    alt_text = figure.get("alt_text", f"Result {index + 1}")

    url = figure.get("url")
    if url and url.startswith("data:"):
        st.image(url, caption=alt_text)
    elif proxy_binary and proxy_mime and proxy_mime.startswith("image/"):
        encoded = base64.b64encode(proxy_binary).decode("utf-8")
        st.image(f"data:{proxy_mime};base64,{encoded}", caption=alt_text)
    elif proxy_url:
        st.image(proxy_url, caption=alt_text)
    elif url:
        st.image(url, caption=alt_text)
    else:
        st.write(f"Image available via resource URI: `{figure.get('uri', 'unknown')}`")


def _render_artifact(artifact: Dict[str, Any]) -> None:
    display_type = artifact.get("display_type", "download")
    filename = artifact.get("filename", "artifact")
    mime = artifact.get("mime_type", "application/octet-stream")

    proxy_binary, proxy_mime, proxy_url = _resolve_proxy(artifact)
    if proxy_mime and (mime == "application/octet-stream" or not mime):
        mime = proxy_mime

    if display_type == "image":
        data_ref = artifact.get("data_base64") or artifact.get("url")
        if data_ref and isinstance(data_ref, str):
            if data_ref.startswith("data:"):
                st.image(data_ref, caption=filename)
            else:
                st.image(f"data:{mime};base64,{data_ref}", caption=filename)
            return
        if proxy_binary and proxy_mime and proxy_mime.startswith("image/"):
            encoded = base64.b64encode(proxy_binary).decode("utf-8")
            st.image(f"data:{proxy_mime};base64,{encoded}", caption=filename)
            return
        if proxy_url:
            st.image(proxy_url, caption=filename)
            return
        if artifact.get("url"):
            st.image(artifact["url"], caption=filename)
            return
        st.write(f"Image available at {artifact.get('uri', 'unknown resource')}")
        return

    if display_type == "text":
        text_content = artifact.get("text")
        if not text_content and proxy_binary:
            try:
                text_content = proxy_binary.decode("utf-8")
            except UnicodeDecodeError:
                text_content = None
        if text_content:
            st.markdown(f"**{filename}**")
            st.code(text_content, language="text")
        elif proxy_url:
            st.markdown(f"[View {filename}]({proxy_url})")
        elif artifact.get("url"):
            st.markdown(f"[View {filename}]({artifact['url']})")
        else:
            st.markdown(f"*No text content available for {filename}*")
        return

    data_base64 = artifact.get("data_base64")
    if data_base64:
        try:
            binary = base64.b64decode(data_base64)
            st.download_button(
                label=f"Download {filename}",
                data=binary,
                file_name=filename,
                mime=mime,
            )
            return
        except Exception:
            st.warning(f"Failed to decode artifact {filename}; showing resource URI instead.")

    if proxy_binary:
        st.download_button(
            label=f"Download {filename}",
            data=proxy_binary,
            file_name=filename,
            mime=mime,
        )
    elif proxy_url:
        st.markdown(f"[Download {filename}]({proxy_url})")
    elif artifact.get("url"):
        st.markdown(f"[Download {filename}]({artifact['url']})")
    elif artifact.get("uri"):
        st.markdown(f"Resource available: `{artifact['uri']}`")


# ============================================================================
# Streamlit UI
# ============================================================================


def main() -> None:
    st.set_page_config(page_title="Data Analyst", page_icon="📊", layout="wide")

    if SESSION_ID_KEY not in st.session_state:
        st.session_state[SESSION_ID_KEY] = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.uploaded_assets = []

    session_id = st.session_state[SESSION_ID_KEY]

    st.markdown("# 📊 Data Analysis Agent")
    st.markdown(f"**Session**: `{session_id}`")
    st.divider()

    page = st.sidebar.selectbox("Select Page", ["📤 Upload", "💬 Chat"])

    if page == "📤 Upload":
        display_upload_page(session_id)
    else:
        display_chat_page(session_id)

    st.divider()
    st.caption(f"Session: {session_id}")


def display_upload_page(session_id: str) -> None:
    st.header("Upload Dataset")
    st.markdown(
        "Upload one or more data files to analyze. "
        f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}."
    )

    if _MINIO_INIT_ERROR:
        st.warning(f"MinIO not available: {_MINIO_INIT_ERROR}")

    uploaded_files = st.file_uploader(
        "Choose data files",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if not _validate_file_size(uploaded_file.size):
                st.error(
                    f"`{uploaded_file.name}` exceeds the {MAX_UPLOAD_MB}MB upload limit."
                )
                continue

            if not _is_allowed_extension(uploaded_file.name, ALLOWED_EXTENSIONS):
                st.error(f"`{uploaded_file.name}` is not an allowed file type.")
                continue

            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            if not file_bytes:
                st.error(f"`{uploaded_file.name}` is empty.")
                continue

            try:
                download_url = _store_file(
                    session_id,
                    uploaded_file.name,
                    file_bytes,
                    uploaded_file.type,
                )
            except RuntimeError as exc:
                st.error(str(exc))
                continue

            st.success(f"Uploaded: `{uploaded_file.name}`")
            asset_record = {
                "name": uploaded_file.name,
                "size_bytes": len(file_bytes),
                "download_url": download_url,
            }
            for idx, existing in enumerate(st.session_state.uploaded_assets):
                if existing["name"] == uploaded_file.name:
                    st.session_state.uploaded_assets[idx] = asset_record
                    break
            else:
                st.session_state.uploaded_assets.append(asset_record)

            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix == ".csv":
                import pandas as pd

                df = pd.read_csv(io.BytesIO(file_bytes))
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                st.dataframe(df.head(10), use_container_width=True)
                st.divider()
            elif suffix in {".xlsx", ".xls"}:
                st.info("Excel preview not yet supported.")
                st.divider()
            else:
                st.info("Preview is currently available for CSV files only.")
                st.divider()

            if download_url:
                st.markdown(
                    f"[Download `{uploaded_file.name}`]({download_url}) "
                    f"(expires in {MINIO_PRESIGN_EXPIRY} seconds)"
                )
            elif MINIO_ENABLED:
                st.info("Download link unavailable; verify MinIO configuration.")

    if st.session_state.get("uploaded_assets"):
        st.subheader("Uploaded Files")
        assets = st.session_state.uploaded_assets
        table_data = [
            {
                "Name": item["name"],
                "Size (MB)": round(item["size_bytes"] / (1024 * 1024), 2),
                "Download URL": item["download_url"] or "Not available",
            }
            for item in assets
        ]
        st.dataframe(table_data, use_container_width=True)


def display_chat_page(session_id: str) -> None:
    st.header("Analyze Data")

    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            if isinstance(content, str):
                st.markdown(content)
            else:
                st.markdown(str(content))

            if message.get("figures"):
                for i, figure in enumerate(message["figures"]):
                    try:
                        _render_figure(figure, i)
                    except Exception as exc:
                        st.error(f"Failed to load image: {exc}")

            if message.get("artifacts"):
                for artifact in message["artifacts"]:
                    if isinstance(artifact, dict):
                        _render_artifact(artifact)

    prompt = st.chat_input("What would you like to know?")
    if not prompt:
        return

    asset_context = [
        f"{asset['name']}: {asset['download_url']}"
        for asset in st.session_state.uploaded_assets
        if asset.get("download_url")
    ]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if asset_context:
        prompt_to_agent = (
            "Available dataset URLs (first call `register_external_data` with them; it will return local paths you can load directly with pandas):\n"
            + "\n".join(f"- {line}" for line in asset_context)
            + "\n\nUser query:\n"
            + prompt
        )
    else:
        prompt_to_agent = prompt

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    f"{AGENT_URL}/analyze",
                    json={"query": prompt_to_agent, "session_id": session_id},
                    timeout=300,
                )
                if response.status_code == 200:
                    result = response.json()
                    artifacts = result.get("artifacts") or []
                    # Ensure proxy URLs are computed with the configured agent base.
                    for artifact in artifacts:
                        proxy_path = artifact.get("proxy_path")
                        if proxy_path:
                            if not proxy_path.startswith("/"):
                                proxy_path = f"/{proxy_path}"
                                artifact["proxy_path"] = proxy_path
                            proxy_url = artifact.get("proxy_url") or _agent_proxy_url(proxy_path)
                            artifact["proxy_url"] = proxy_url

                    text_output = result.get("text", "")
                    if isinstance(text_output, str) and artifacts:
                        for artifact in artifacts:
                            uri = artifact.get("uri")
                            proxy_url = artifact.get("proxy_url")
                            if uri and proxy_url and uri in text_output:
                                text_output = text_output.replace(uri, proxy_url)

                    if text_output:
                        st.markdown(text_output)
                    if result.get("figures"):
                        for i, figure in enumerate(result["figures"]):
                            try:
                                _render_figure(figure, i)
                            except Exception as exc:
                                st.error(f"Failed to load image: {exc}")
                    if artifacts:
                        for artifact in artifacts:
                            if isinstance(artifact, dict):
                                _render_artifact(artifact)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": text_output,
                            "figures": result.get("figures", []),
                            "artifacts": artifacts,
                        }
                    )
                else:
                    st.error(f"Error from agent-service: {response.text}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"Error from agent-service: {response.text}",
                        }
                    )
            except requests.exceptions.ConnectionError:
                msg = "Agent service not running (expected on port 8002)."
                st.error(msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg}
                )
            except Exception as exc:
                msg = f"Unexpected error: {exc}"
                st.error(msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg}
                )


if __name__ == "__main__":
    main()
