"""阶段二配置测试。

测试只验证 Settings 的本地解析能力。这里不会读取真实 OpenAI key，也不会访问网络。
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import yaml
from pydantic import SecretStr

from redis_sre_agent.core.config import DEFAULT_CONFIG_PATHS, MCPServerConfig, Settings


def test_settings_can_be_created_without_real_external_secrets() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "Redis SRE Agent"
    assert settings.debug is False
    assert isinstance(settings.redis_url, SecretStr)
    assert settings.openai_api_key is None


def test_secretstr_does_not_leak_in_repr() -> None:
    settings = Settings(
        _env_file=None,
        redis_url=SecretStr("redis://user:fake-secret@example.invalid:6379/0"),
    )

    rendered = repr(settings)
    assert "fake-secret" not in rendered
    assert "SecretStr" in rendered


def test_environment_overrides_defaults() -> None:
    env = {
        "APP_NAME": "local-test-agent",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "HOST": "127.0.0.1",
        "PORT": "9001",
        "REDIS_URL": "redis://localhost:6379/2",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = Settings(_env_file=None)

    assert settings.app_name == "local-test-agent"
    assert settings.debug is True
    assert settings.log_level == "DEBUG"
    assert settings.host == "127.0.0.1"
    assert settings.port == 9001
    assert settings.redis_url.get_secret_value() == "redis://localhost:6379/2"


def test_yaml_config_file_loads_and_env_wins(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"app_name": "from-yaml", "debug": False, "mcp_servers": {"demo": {"command": "echo"}}}),
        encoding="utf-8",
    )

    with patch.dict(
        os.environ,
        {"SRE_AGENT_CONFIG": str(config_path), "DEBUG": "true"},
        clear=True,
    ):
        settings = Settings(_env_file=None)

    assert settings.app_name == "from-yaml"
    assert settings.debug is True
    assert "demo" in settings.mcp_servers
    server = settings.mcp_servers["demo"]
    if isinstance(server, MCPServerConfig):
        assert server.command == "echo"
    else:
        assert server["command"] == "echo"


def test_default_config_paths_are_stable() -> None:
    assert DEFAULT_CONFIG_PATHS == [
        "config.yaml",
        "config.yml",
        "config.toml",
        "config.json",
        "sre_agent_config.yaml",
        "sre_agent_config.yml",
        "sre_agent_config.toml",
        "sre_agent_config.json",
    ]
