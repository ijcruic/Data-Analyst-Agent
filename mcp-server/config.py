import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


DEFAULT_CONFIG_PATH = Path(
    os.getenv("MCP_CONFIG_PATH", Path(__file__).with_name("config.yaml"))
)

ENV_OVERRIDE_MAP: Dict[Tuple[str, ...], Tuple[str, ...]] = {
    ("server", "host"): ("MCP_HOST",),
    ("server", "port"): ("MCP_PORT",),
    ("storage", "backend"): ("MCP_STORAGE_BACKEND", "STORAGE_BACKEND"),
    ("storage", "minio", "endpoint"): ("MCP_MINIO_ENDPOINT", "MINIO_ENDPOINT"),
    ("storage", "minio", "region"): ("MCP_MINIO_REGION", "MINIO_REGION"),
    ("storage", "minio", "access_key"): ("MCP_MINIO_ACCESS_KEY", "MINIO_ACCESS_KEY"),
    ("storage", "minio", "secret_key"): ("MCP_MINIO_SECRET_KEY", "MINIO_SECRET_KEY"),
    ("storage", "minio", "bucket"): ("MCP_MINIO_BUCKET", "MINIO_BUCKET"),
    ("storage", "minio", "secure"): ("MCP_MINIO_SECURE", "MINIO_SECURE"),
    ("storage", "minio", "session_prefix"): (
        "MCP_MINIO_SESSION_PREFIX",
        "MINIO_SESSION_PREFIX",
    ),
    ("storage", "minio", "presign_expiry_seconds"): (
        "MCP_MINIO_PRESIGN_EXPIRY_SECONDS",
        "MINIO_PRESIGN_EXPIRY_SECONDS",
    ),
    ("storage", "minio", "public_endpoint"): (
        "MCP_MINIO_PUBLIC_ENDPOINT",
        "MINIO_PUBLIC_ENDPOINT",
    ),
    ("cleanup", "ttl_seconds"): ("MCP_CLEANUP_TTL_SECONDS",),
    ("cleanup", "interval_seconds"): ("MCP_CLEANUP_INTERVAL_SECONDS",),
    ("execution", "timeout_seconds"): ("MCP_EXECUTION_TIMEOUT_SECONDS",),
    ("execution", "memory_limit_mb"): ("MCP_EXECUTION_MEMORY_LIMIT_MB",),
    ("execution", "allowed_modules"): ("MCP_ALLOWED_MODULES",),
    ("telemetry", "matomo", "site_id"): ("MCP_MATOMO_SITE_ID", "MATOMO_SITE_ID"),
    ("telemetry", "matomo", "url"): ("MCP_MATOMO_URL", "MATOMO_URL"),
    ("telemetry", "matomo", "ssl_verify"): (
        "MCP_MATOMO_SSL_VERIFY",
        "MATOMO_SSL_VERIFY",
    ),
    ("results", "cache_enabled"): ("MCP_RESULTS_CACHE_ENABLED",),
    ("results", "max_entries"): ("MCP_RESULTS_MAX_ENTRIES",),
}


class ConfigError(Exception):
    """Raised when configuration loading fails."""


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
            if not isinstance(data, dict):
                raise ConfigError(f"Configuration root must be a mapping: {path}")
            return data
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config {path}: {exc}") from exc


def _set_nested(config: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    cursor = config
    path_list = list(path)
    for key in path_list[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[path_list[-1]] = value


def _get_nested(config: Dict[str, Any], path: Iterable[str]) -> Any:
    cursor: Any = config
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _convert_value(raw_value: str, current_value: Any) -> Any:
    if current_value is None:
        return raw_value

    if isinstance(current_value, bool):
        return raw_value.lower() in {"1", "true", "yes", "on"}

    if isinstance(current_value, int):
        try:
            return int(raw_value)
        except ValueError as exc:
            raise ConfigError(f"Expected integer value, got '{raw_value}'") from exc

    if isinstance(current_value, float):
        try:
            return float(raw_value)
        except ValueError as exc:
            raise ConfigError(f"Expected float value, got '{raw_value}'") from exc

    if isinstance(current_value, list):
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    return raw_value


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    for path, env_names in ENV_OVERRIDE_MAP.items():
        for env_name in env_names:
            env_value = os.getenv(env_name)
            if env_value is not None:
                current = _get_nested(config, path)
                converted = _convert_value(env_value, current)
                _set_nested(config, path, converted)
                break


@lru_cache
def load_config() -> Dict[str, Any]:
    """Load configuration from YAML and apply environment overrides."""
    config_path = DEFAULT_CONFIG_PATH
    override_path = os.getenv("CONFIG_YAML_PATH")
    if override_path:
        config_path = Path(override_path)

    config = _read_yaml(config_path)
    _apply_env_overrides(config)
    return config


def reload_config() -> Dict[str, Any]:
    """Clear the cache and reload configuration."""
    load_config.cache_clear()
    return load_config()
