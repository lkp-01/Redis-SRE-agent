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
    assert settings.openai_base_url == "https://api.deepseek.com"
    assert settings.openai_model == "deepseek-v4-pro"
    assert settings.openai_model_mini == "deepseek-v4-flash"
    assert settings.openai_model_nano == "deepseek-v4-flash"
    assert settings.llm_timeout == 180.0
    assert settings.llm_failover_enabled is True
    assert settings.deepseek_thinking_mode == "disabled"


def test_secretstr_does_not_leak_in_repr() -> None:
    settings = Settings(
        _env_file=None,
        redis_url=SecretStr("FAKE_TEST_REDIS_CONNECTION_REF"),
    )

    rendered = repr(settings)
    assert "FAKE_TEST_REDIS_CONNECTION_REF" not in rendered
    assert "SecretStr" in rendered


def test_environment_overrides_defaults() -> None:
    env = {
        "APP_NAME": "local-test-agent",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "HOST": "127.0.0.1",
        "PORT": "9001",
        "REDIS_URL": "LOCAL_TEST_REDIS_REFERENCE",
        "OPENAI_MODEL": "primary-test-model",
        "OPENAI_MODEL_MINI": "fallback-test-model",
        "LLM_FAILOVER_ENABLED": "false",
        "DEEPSEEK_THINKING_MODE": "enabled",
    }
    with patch.dict(os.environ, env, clear=True):
        settings = Settings(_env_file=None)

    assert settings.app_name == "local-test-agent"
    assert settings.debug is True
    assert settings.log_level == "DEBUG"
    assert settings.host == "127.0.0.1"
    assert settings.port == 9001
    assert settings.redis_url.get_secret_value() == "LOCAL_TEST_REDIS_REFERENCE"
    assert settings.openai_model == "primary-test-model"
    assert settings.openai_model_mini == "fallback-test-model"
    assert settings.llm_failover_enabled is False
    assert settings.deepseek_thinking_mode == "enabled"


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
