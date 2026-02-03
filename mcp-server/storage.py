from __future__ import annotations

import io
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Union
import tempfile

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class StorageError(Exception):
    """Raised when storage operations fail."""


@dataclass
class FileRecord:
    """Metadata describing a stored file."""

    name: str
    size: int
    last_modified: datetime
    url: Optional[str] = None


@dataclass
class SessionRecord:
    """Metadata describing a session's last activity."""

    session_id: str
    last_modified: datetime


class BaseStorage:
    """Abstract storage interface."""

    def list_files(self, session_id: str) -> List[FileRecord]:
        raise NotImplementedError

    def read_file(self, session_id: str, filename: str) -> BinaryIO:
        raise NotImplementedError

    def get_local_path(self, session_id: str, filename: str) -> Optional[Path]:
        """Return a filesystem path if available (filesystem backend only)."""
        return None

    def materialize_file(self, session_id: str, filename: str) -> tuple[Path, bool]:
        """Ensure the file exists on the local filesystem and return the path."""
        path = self.get_local_path(session_id, filename)
        if path is None:
            raise StorageError("materialize_file not implemented for this backend")
        return path, False

    def generate_presigned_url(self, session_id: str, filename: str, expiry: int) -> Optional[str]:
        return None

    def store_file(self, session_id: str, filename: str, data: BinaryIO, size: int) -> None:
        raise NotImplementedError

    def delete_session(self, session_id: str) -> None:
        raise NotImplementedError

    def list_sessions(self) -> List[SessionRecord]:
        raise NotImplementedError

    def upload_artifact(self, session_id: str, local_path: Path, mime_type: str) -> Optional[str]:
        raise NotImplementedError


class FilesystemStorage(BaseStorage):
    """Filesystem-backed storage for local development."""

    def __init__(self, root: Path, presign_expiry: int = 600):
        self.root = root
        self.presign_expiry = presign_expiry
        self.root.mkdir(parents=True, exist_ok=True)
        self.outputs_root = self.root / "outputs"
        self.outputs_root.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_files(self, session_id: str) -> List[FileRecord]:
        session_dir = self._session_dir(session_id)
        records: List[FileRecord] = []
        for path in session_dir.iterdir():
            if path.is_file():
                stat = path.stat()
                records.append(
                    FileRecord(
                        name=path.name,
                        size=stat.st_size,
                        last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                    )
                )
        return records

    def read_file(self, session_id: str, filename: str) -> BinaryIO:
        path = self._session_dir(session_id) / filename
        if not path.exists():
            raise StorageError(f"File not found: {path}")
        return path.open("rb")

    def get_local_path(self, session_id: str, filename: str) -> Optional[Path]:
        path = self._session_dir(session_id) / filename
        return path if path.exists() else None

    def materialize_file(self, session_id: str, filename: str) -> tuple[Path, bool]:
        path = self.get_local_path(session_id, filename)
        if path is None:
            raise StorageError(f"File not found: {filename}")
        return path, False

    def generate_presigned_url(self, session_id: str, filename: str, expiry: int) -> Optional[str]:
        # Filesystem backend does not support presigned URLs out of the box.
        return None

    def store_file(self, session_id: str, filename: str, data: BinaryIO, size: int) -> None:
        path = self._session_dir(session_id) / filename
        data.seek(0)
        with path.open("wb") as handle:
            handle.write(data.read())

    def delete_session(self, session_id: str) -> None:
        session_dir = self._session_dir(session_id)
        for path in session_dir.glob("*"):
            if path.is_file():
                path.unlink()
        try:
            session_dir.rmdir()
        except OSError:
            pass

    def list_sessions(self) -> List[SessionRecord]:
        records: List[SessionRecord] = []
        for path in self.root.iterdir():
            if not path.is_dir():
                continue
            session_id = path.name
            latest = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            for file in path.glob("**/*"):
                try:
                    mtime = datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
                    if mtime > latest:
                        latest = mtime
                except FileNotFoundError:
                    continue
            records.append(SessionRecord(session_id=session_id, last_modified=latest))
        return records

    def upload_artifact(self, session_id: str, local_path: Path, mime_type: str) -> Optional[str]:
        dest_dir = self.outputs_root / session_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / local_path.name
        if local_path.resolve() != dest.resolve():
            shutil.copy2(local_path, dest)
        return str(dest)


def _create_s3_client(
    endpoint: str,
    access_key: str,
    secret_key: str,
    region: Optional[str],
    secure: bool,
) -> Any:
    """Create a boto3 S3 client configured for MinIO compatibility."""
    endpoint = endpoint.strip()
    has_scheme = endpoint.startswith("http://") or endpoint.startswith("https://")
    use_ssl = secure
    if has_scheme:
        use_ssl = endpoint.startswith("https://")
        endpoint_url = endpoint
    else:
        endpoint_url = f"https://{endpoint}" if secure else f"http://{endpoint}"

    resource = boto3.resource(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        use_ssl=use_ssl,
        verify=False,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    return resource.meta.client


class MinioStorage(BaseStorage):
    """MinIO-backed storage for Kubernetes deployments."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        session_prefix: str = "sessions/",
        region: Optional[str] = None,
        presign_expiry: int = 600,
        public_endpoint: Optional[str] = None,
    ):
        self.client = _create_s3_client(endpoint, access_key, secret_key, region, secure)
        self.minio_endpoint = endpoint  # Store original endpoint for URL generation
        self.bucket = bucket
        self.session_prefix = session_prefix.strip("/")
        if self.session_prefix:
            self.session_prefix += "/"
        self.presign_expiry = presign_expiry
        self.public_endpoint = public_endpoint
        self.output_prefix = "outputs/"

    def _object_name(self, session_id: str, filename: str) -> str:
        return f"{self.session_prefix}{session_id}/{filename}"

    def _normalize_datetime(self, value: Union[datetime, int, float, None]) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        return datetime.now(timezone.utc)

    def list_files(self, session_id: str) -> List[FileRecord]:
        prefix = self._object_name(session_id, "")
        paginator = self.client.get_paginator("list_objects_v2")
        try:
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
        except ClientError as exc:
            raise StorageError(f"Failed to list objects with prefix {prefix}: {exc}") from exc

        records: List[FileRecord] = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj.get("Key")
                if not key or key.endswith("/"):
                    continue
                name = key.split("/")[-1]
                records.append(
                    FileRecord(
                        name=name,
                        size=int(obj.get("Size", 0)),
                        last_modified=self._normalize_datetime(obj.get("LastModified")),
                    )
                )
        return records

    def read_file(self, session_id: str, filename: str) -> BinaryIO:
        object_name = self._object_name(session_id, filename)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=object_name)
            body = response.get("Body")
            if body is None:
                raise StorageError(f"Object body missing for {object_name}")
            data = io.BytesIO(body.read())
            body.close()
            data.seek(0)
            return data
        except ClientError as exc:
            raise StorageError(f"Failed to read object {object_name}: {exc}") from exc

    def generate_presigned_url(self, session_id: str, filename: str, expiry: int) -> Optional[str]:
        object_name = self._object_name(session_id, filename)
        try:
            expires = expiry
            if isinstance(expires, (int, float)):
                expires = timedelta(seconds=expires)
            elif expires is None:
                expires = timedelta(seconds=self.presign_expiry)
            expires_in = int(expires.total_seconds())
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": object_name},
                ExpiresIn=expires_in,
            )

            if self.public_endpoint:
                from urllib.parse import urlparse, urlunparse

                parsed = urlparse(url)
                pub = urlparse(self.public_endpoint)

                return urlunparse(
                    (
                        pub.scheme or parsed.scheme,
                        pub.netloc or parsed.netloc,
                        parsed.path,
                        parsed.params,
                        parsed.query,
                        parsed.fragment,
                    )
                )
            return url
        except ClientError:
            return None

    def upload_artifact(self, session_id: str, local_path: Path, mime_type: str) -> Optional[str]:
        object_name = f"{self.output_prefix}{session_id}/{local_path.name}"
        try:
            extra_args = {"ContentType": mime_type} if mime_type else None
            self.client.upload_file(
                str(local_path),
                self.bucket,
                object_name,
                ExtraArgs=extra_args or {},
            )
            expires = timedelta(seconds=self.presign_expiry)
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": object_name},
                ExpiresIn=int(expires.total_seconds()),
            )
            if self.public_endpoint:
                from urllib.parse import urlparse, urlunparse

                parsed = urlparse(url)
                pub = urlparse(self.public_endpoint)
                return urlunparse(
                    (
                        pub.scheme or parsed.scheme,
                        pub.netloc or parsed.netloc,
                        parsed.path,
                        parsed.params,
                        parsed.query,
                        parsed.fragment,
                    )
                )
            return url
        except ClientError as exc:
            raise StorageError(f"Failed to upload artifact: {exc}") from exc

    def materialize_file(self, session_id: str, filename: str) -> tuple[Path, bool]:
        data = self.read_file(session_id, filename)
        suffix = Path(filename).suffix or ""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data.read())
        tmp.flush()
        tmp.close()
        return Path(tmp.name), True

    def store_file(self, session_id: str, filename: str, data: BinaryIO, size: int) -> None:
        object_name = self._object_name(session_id, filename)
        data.seek(0)
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=object_name,
                Body=data,
                ContentLength=size,
            )
        except ClientError as exc:
            raise StorageError(f"Failed to store object {object_name}: {exc}") from exc

    def delete_session(self, session_id: str) -> None:
        prefix = self._object_name(session_id, "")
        paginator = self.client.get_paginator("list_objects_v2")
        try:
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
        except ClientError:
            return

        keys_to_delete: List[Dict[str, str]] = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj.get("Key")
                if key:
                    keys_to_delete.append({"Key": key})

        if not keys_to_delete:
            return

        # Batch deletions in chunks of 1000 per S3 API limits
        for i in range(0, len(keys_to_delete), 1000):
            chunk = keys_to_delete[i : i + 1000]
            try:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": chunk})
            except ClientError:
                continue

    def list_sessions(self) -> List[SessionRecord]:
        prefix = self.session_prefix
        paginator = self.client.get_paginator("list_objects_v2")
        try:
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
        except ClientError as exc:
            raise StorageError(f"Failed to list sessions: {exc}") from exc

        latest_map: Dict[str, datetime] = {}
        for page in pages:
            for obj in page.get("Contents", []):
                name = obj.get("Key")
                if not name or not name.startswith(prefix):
                    continue
                remainder = name[len(prefix):]
                if "/" not in remainder:
                    continue
                session_id = remainder.split("/", 1)[0]
                last_modified = self._normalize_datetime(obj.get("LastModified"))
                current = latest_map.get(session_id)
                if current is None or last_modified > current:
                    latest_map[session_id] = last_modified
        return [
            SessionRecord(session_id=sid, last_modified=self._normalize_datetime(latest))
            for sid, latest in latest_map.items()
        ]


def build_storage(config: Dict[str, any]) -> BaseStorage:
    storage_cfg = config.get("storage", {})
    backend = storage_cfg.get("backend", "filesystem").lower()

    if backend == "filesystem":
        fs_root = Path(storage_cfg.get("filesystem", {}).get("root", "/tmp/uploads"))
        presign_expiry = storage_cfg.get("filesystem", {}).get("presign_expiry_seconds", 600)
        return FilesystemStorage(fs_root, presign_expiry=presign_expiry)

    if backend == "minio":
        minio_cfg = storage_cfg.get("minio", {})
        required_keys = ["endpoint", "access_key", "secret_key", "bucket"]
        missing = [key for key in required_keys if not minio_cfg.get(key)]
        if missing:
            raise StorageError(f"Missing MinIO configuration keys: {', '.join(missing)}")

        return MinioStorage(
            endpoint=minio_cfg["endpoint"],
            access_key=minio_cfg["access_key"],
            secret_key=minio_cfg["secret_key"],
            bucket=minio_cfg["bucket"],
            secure=bool(minio_cfg.get("secure", False)),
            session_prefix=minio_cfg.get("session_prefix", "sessions/"),
            region=minio_cfg.get("region"),
            presign_expiry=int(minio_cfg.get("presign_expiry_seconds", 600)),
            public_endpoint=minio_cfg.get("public_endpoint"),
        )

    raise StorageError(f"Unsupported storage backend: {backend}")
