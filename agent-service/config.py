import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


DEFAULT_CONFIG_PATH = Path(
    os.getenv("AGENT_CONFIG_PATH", Path(__file__).with_name("config.yaml"))
)

ENV_OVERRIDE_MAP: Dict[Tuple[str, ...], Tuple[str, ...]] = {
    ("server", "host"): ("AGENT_HOST",),
    ("server", "port"): ("AGENT_PORT",),
    ("llm", "provider"): ("AGENT_LLM_PROVIDER", "LLM_PROVIDER"),
    ("llm", "model"): ("OPENAI_MODEL", "AGENT_LLM_MODEL"),
    ("llm", "temperature"): ("AGENT_TEMPERATURE",),
    ("llm", "openai", "api_base"): (
        "OPENAI_API_BASE",
        "LLM_API_BASE",
        "AGENT_LLM_API_BASE",
    ),
    ("llm", "openai", "api_key"): ("OPENAI_API_KEY", "AGENT_LLM_API_KEY"),
    ("llm", "bedrock", "region"): ("AGENT_LLM_AWS_REGION", "AWS_REGION"),
    ("llm", "bedrock", "profile"): ("AGENT_LLM_AWS_PROFILE", "AWS_PROFILE"),
    ("llm", "bedrock", "access_key_id"): (
        "AGENT_LLM_AWS_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY_ID",
    ),
    ("llm", "bedrock", "secret_access_key"): (
        "AGENT_LLM_AWS_SECRET_ACCESS_KEY",
        "AWS_SECRET_ACCESS_KEY",
    ),
    ("llm", "bedrock", "session_token"): (
        "AGENT_LLM_AWS_SESSION_TOKEN",
        "AWS_SESSION_TOKEN",
    ),
    ("llm", "bedrock", "endpoint_url"): ("AGENT_LLM_AWS_ENDPOINT_URL",),
    ("mcp", "endpoints"): ("AGENT_MCP_ENDPOINTS", "MCP_URL"),
    ("mcp", "connect_timeout_seconds"): ("MCP_CONNECT_TIMEOUT_SECONDS",),
    ("mcp", "max_retries"): ("MCP_MAX_RETRIES",),
    ("storage", "minio", "endpoint"): ("AGENT_MINIO_ENDPOINT", "MINIO_ENDPOINT"),
    ("storage", "minio", "region"): ("AGENT_MINIO_REGION", "MINIO_REGION"),
    ("storage", "minio", "access_key"): ("AGENT_MINIO_ACCESS_KEY", "MINIO_ACCESS_KEY"),
    ("storage", "minio", "secret_key"): ("AGENT_MINIO_SECRET_KEY", "MINIO_SECRET_KEY"),
    ("storage", "minio", "bucket"): ("AGENT_MINIO_BUCKET", "MINIO_BUCKET"),
    ("storage", "minio", "secure"): ("AGENT_MINIO_SECURE", "MINIO_SECURE"),
    ("telemetry", "matomo", "site_id"): ("AGENT_MATOMO_SITE_ID", "MATOMO_SITE_ID"),
    ("telemetry", "matomo", "url"): ("AGENT_MATOMO_URL", "MATOMO_URL"),
    ("telemetry", "matomo", "ssl_verify"): (
        "AGENT_MATOMO_SSL_VERIFY",
        "MATOMO_SSL_VERIFY",
    ),
    ("logging", "level"): ("AGENT_LOG_LEVEL", "LOG_LEVEL"),
    ("logging", "json"): ("AGENT_LOG_JSON", "LOG_JSON"),
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
