#!/usr/bin/env python3
"""
Lightweight end-to-end smoke test for the MCP workflow.

Flow:
1. Upload tabular + JSON fixtures to MinIO under the session prefix.
2. Generate presigned URLs that the MCP server (running in Docker) can fetch.
3. Drive the MCP server via the official Streamable HTTP client (JSON-RPC).
4. Validate registration, listing, schema inspection, and dynamic analysis.
"""

from __future__ import annotations

import ast
import io
import json
import os
import uuid
import textwrap
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import anyio
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "data-analyst")
MINIO_PUBLIC_ENDPOINT = os.getenv("MINIO_PUBLIC_ENDPOINT", "localhost:9000")
MINIO_INTERNAL_ENDPOINT = os.getenv("MINIO_INTERNAL_ENDPOINT", "minio:9000")
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")

MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:4242/mcp")

SESSION_ID = f"smoke-{uuid.uuid4().hex[:8]}"
TSV_OBJECT = f"sessions/{SESSION_ID}/sample.tsv"
JSON_OBJECT = f"sessions/{SESSION_ID}/products.json"


def _create_s3_client(endpoint: str) -> Any:
    endpoint = endpoint.strip()
    has_scheme = endpoint.startswith("http://") or endpoint.startswith("https://")
    if has_scheme:
        endpoint_url = endpoint
        use_ssl = endpoint.startswith("https://")
    else:
        endpoint_url = f"http://{endpoint}"
        use_ssl = False

    resource = boto3.resource(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name=MINIO_REGION,
        use_ssl=use_ssl,
        verify=False,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    return resource.meta.client


def _ensure_bucket_exists(client: Any, bucket: str) -> None:
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchBucket"}:
            client.create_bucket(Bucket=bucket)
        else:
            raise


def _upload_fixture_objects() -> Dict[str, str]:
    """Upload sample datasets to MinIO and return presigned URLs for the MCP server."""
    public_client = _create_s3_client(MINIO_PUBLIC_ENDPOINT)
    _ensure_bucket_exists(public_client, MINIO_BUCKET)

    tsv_data = "category\tvalue\nA\t10\nB\t15\n"
    public_client.put_object(
        Bucket=MINIO_BUCKET,
        Key=TSV_OBJECT,
        Body=io.BytesIO(tsv_data.encode("utf-8")),
        ContentLength=len(tsv_data),
        ContentType="text/tab-separated-values",
    )

    json_data = '[{"sku": "SKU-1", "price": 12.5}, {"sku": "SKU-2", "price": 7.5}]'
    public_client.put_object(
        Bucket=MINIO_BUCKET,
        Key=JSON_OBJECT,
        Body=io.BytesIO(json_data.encode("utf-8")),
        ContentLength=len(json_data),
        ContentType="application/json",
    )

    # Generate presigned URLs using the container-visible hostname.
    internal_client = _create_s3_client(MINIO_INTERNAL_ENDPOINT)

    return {
        "tsv_url": internal_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": MINIO_BUCKET, "Key": TSV_OBJECT},
            ExpiresIn=3600,
        ),
        "json_url": internal_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": MINIO_BUCKET, "Key": JSON_OBJECT},
            ExpiresIn=3600,
        ),
    }


@asynccontextmanager
async def _open_mcp_session() -> ClientSession:
    async with streamable_http_client(MCP_BASE_URL) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session


def _first_text_block(result: Any) -> str:
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("Tool response did not include a text content block.")


def _json_from_result(result: Any) -> Dict[str, Any] | List[Any]:
    return json.loads(_first_text_block(result))


async def _run_async(tsv_url: str, json_url: str) -> None:
    async with _open_mcp_session() as session:
        registration = await session.call_tool(
            "register_external_data",
            {
                "session_id": SESSION_ID,
                "assets": [
                    {"url": tsv_url, "filename": "sample.tsv", "loader_hints": {"sep": "\t"}},
                    {"url": json_url, "filename": "products.json"},
                ],
            },
        )
        registration_data = _json_from_result(registration)
        datasets = registration_data.get("datasets", [])
        if len(datasets) != 2 or registration_data.get("errors"):
            raise SystemExit(f"Registration failed: {registration_data}")

        paths = [item["path"] for item in datasets]

        listing = _json_from_result(
            await session.call_tool("list_available_files", {"session_id": SESSION_ID})
        )
        if len(listing) < 2:
            raise SystemExit(f"Expected two cached files, got: {listing}")

        schema_tsv = _json_from_result(
            await session.call_tool(
                "get_file_schema", {"session_id": SESSION_ID, "filename": "sample.tsv"}
            )
        )
        if schema_tsv.get("shape") != [2, 2]:
            raise SystemExit(f"Unexpected TSV schema: {schema_tsv}")

        schema_json = _json_from_result(
            await session.call_tool(
                "get_file_schema", {"session_id": SESSION_ID, "filename": "products.json"}
            )
        )
        if schema_json.get("shape") != [2, 2]:
            raise SystemExit(f"Unexpected JSON schema: {schema_json}")

        analysis_code = textwrap.dedent(
            """
            import pandas as pd

            path = paths[0]
            df = pd.read_csv(path, sep="\\t")
            result = {
                "row_count": int(df.shape[0]),
                "total_value": float(df["value"].sum()),
                "categories": df["category"].tolist(),
            }
            """
        )
        analysis = _json_from_result(
            await session.call_tool(
                "analyze_session_data",
                {"session_id": SESSION_ID, "paths": [paths[0]], "code": analysis_code},
            )
        )
        if "error" in analysis:
            raise SystemExit(f"TSV analysis failed: {analysis['error']}")
        summary = ast.literal_eval(analysis.get("text", "{}"))
        if summary.get("row_count") != 2 or summary.get("total_value") != 25:
            raise SystemExit(f"Unexpected TSV analysis result: {summary}")

        analysis_json_code = textwrap.dedent(
            """
            import pandas as pd

            path = paths[0]
            df = pd.read_json(path)
            result = {
                "count": int(df.shape[0]),
                "prices": df["price"].tolist(),
            }
            """
        )
        analysis_json_result = _json_from_result(
            await session.call_tool(
                "analyze_session_data",
                {"session_id": SESSION_ID, "paths": [paths[1]], "code": analysis_json_code},
            )
        )
        if "error" in analysis_json_result:
            raise SystemExit(f"JSON analysis failed: {analysis_json_result['error']}")
        summary_json = ast.literal_eval(analysis_json_result.get("text", "{}"))
        if summary_json.get("count") != 2 or summary_json.get("prices") != [12.5, 7.5]:
            raise SystemExit(f"Unexpected JSON analysis result: {summary_json}")


def run_smoke_test() -> None:
    print(f"Using session_id={SESSION_ID}")
    print("Uploading fixtures to MinIO...")
    urls = _upload_fixture_objects()
    print("Driving MCP workflow...")
    anyio.run(_run_async, urls["tsv_url"], urls["json_url"])
    print("✅ Smoke test completed successfully.")


if __name__ == "__main__":
    run_smoke_test()
