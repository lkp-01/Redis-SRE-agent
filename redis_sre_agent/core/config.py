"""配置读取。

资源层需要先知道几个基础事实：应用叫什么、日志开到什么级别、Redis 在哪里、未来
OpenAI/向量/工具/MCP/target/skill 这些扩展点的配置字段叫什么。

这里沿用原项目的 Pydantic Settings 思路：默认值写在模型里，环境变量可以覆盖默认值，
`.env` 适合本地开发，YAML/TOML/JSON 配置文件适合把一组配置放在一个文件里。优先级是：
构造函数显式传入 > 环境变量 > `.env` > 配置文件 > 默认值。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    InitSettingsSource,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
    YamlConfigSettingsSource,
)

from redis_sre_agent.tools.models import ToolActionKind, ToolCapability

ENV_FILE_OPT: str | None = None
TWENTY_MINUTES_IN_SECONDS = 1200

_env_path = Path(".env")
try:
    if _env_path.is_file():
        load_dotenv(dotenv_path=_env_path)
        ENV_FILE_OPT = str(_env_path)
    else:
        load_dotenv()
except (FileNotFoundError, OSError):
    pass

DEFAULT_CONFIG_PATHS = [
    "config.yaml",
    "config.yml",
    "config.toml",
    "config.json",
    "sre_agent_config.yaml",
    "sre_agent_config.yml",
    "sre_agent_config.toml",
    "sre_agent_config.json",
]

#它先看系统的环境变量里有没有指定文件路径；如果没有，就照着上面的名单挨个找，找到哪个算哪个。
def _get_config_file_path() -> str | None:
    config_path = os.environ.get("SRE_AGENT_CONFIG")
    if config_path:
        return config_path
    for default_path in DEFAULT_CONFIG_PATHS:
        if Path(default_path).is_file():
            return default_path
    return None

# 因为找到的配置文件可能是 YAML、TOML 或者是 JSON 格式的。
# 这个函数会根据文件的后缀名，挑一个合适的解析器，把文件内容翻译成 Python 能看懂的数据。
def _build_config_file_source(settings_cls: Type[BaseSettings]) -> PydanticBaseSettingsSource:
    config_path = _get_config_file_path()
    if config_path is None:
        return InitSettingsSource(settings_cls, {})

    suffix = Path(config_path).suffix.lower()
    if suffix in {".yaml", ".yml", ""}:
        source_type = YamlConfigSettingsSource
        source_kwargs = {"yaml_file": config_path}
    elif suffix == ".toml":
        source_type = TomlConfigSettingsSource
        source_kwargs = {"toml_file": config_path}
    elif suffix == ".json":
        source_type = JsonConfigSettingsSource
        source_kwargs = {"json_file": config_path}
    else:
        source_type = YamlConfigSettingsSource
        source_kwargs = {"yaml_file": config_path}

    try:
        return source_type(settings_cls, **source_kwargs)
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return InitSettingsSource(settings_cls, {})

#下面两个专门用来装与MCP工具相关的配置参数

class MCPToolConfig(BaseModel):

    capability: Optional[ToolCapability] = Field(default=None, description="工具能力分类。")
    description: Optional[str] = Field(default=None, description="给上层 Agent 看的工具说明。")
    action_kind: Optional[ToolActionKind] = Field(default=None, description="读或写动作类型。")


class MCPServerConfig(BaseModel):

    command: Optional[str] = Field(default=None, description="stdio 方式启动服务的命令。")
    args: Optional[List[str]] = Field(default=None, description="命令参数。")
    env: Optional[Dict[str, str]] = Field(default=None, description="服务进程环境变量。")
    url: Optional[str] = Field(default=None, description="HTTP/SSE 方式的服务地址。")
    headers: Optional[Dict[str, str]] = Field(default=None, description="HTTP 请求头。")
    transport: Optional[str] = Field(default=None, description="传输类型插槽。")
    tools: Optional[Dict[str, MCPToolConfig]] = Field(default=None, description="工具约束配置。")

# 下面两个专门用来装系统和外部目标（Target）集成时的配置项
#假设几个月后，你们公司不用普通的 Redis 了，改用云厂商特供版的 Redis（或者你们想通过 Kubernetes 来管理数据库）。
# 如果代码写死了，你就得去核心逻辑里改代码、加 if-else。
# 但有了这两个配置类，你一行核心代码都不用改。你只需要：
# 自己新建一个 Python 文件，写好云厂商特供版的连接逻辑。
# 在配置文件里，加一张新“名片”（TargetIntegrationComponentConfig），把 class_path 指向你刚写的那个文件。
# 系统一启动，就会像插上新 U 盘一样，自动读取这个路径，顺着路径找到你的新代码并运行。这就叫高扩展性和解耦。

#它定义了一个具体的插件长什么样。
# 当主系统想要调用某个外部功能时，请去这个代码路径下把那个 Python 类拽出来执行。
class TargetIntegrationComponentConfig(BaseModel):

    class_path: str
    config: Dict[str, Any] = Field(default_factory=dict)

# 把上面的一本“联络簿”都给整理好
class TargetIntegrationsConfig(BaseModel):
    """target 集成配置插槽。"""

    default_discovery_backend: str = "redis_catalog"
    default_binding_strategy: str = "redis_default"
    discovery_backends: Dict[str, TargetIntegrationComponentConfig] = Field( #负责找目标在哪
        default_factory=lambda: {
            "redis_catalog": TargetIntegrationComponentConfig(
                class_path="redis_sre_agent.targets.redis_catalog.RedisCatalogDiscoveryBackend"
            )
        }
    )
    binding_strategies: Dict[str, TargetIntegrationComponentConfig] = Field(#存放怎么连上目标的dict
        default_factory=lambda: {
            "redis_default": TargetIntegrationComponentConfig(
                class_path="redis_sre_agent.targets.redis_binding.RedisTargetBindingStrategy"
            )
        }
    )
    client_factories: Dict[str, TargetIntegrationComponentConfig] = Field(default_factory=dict)

#系统启动所需要的所有参数，并赋予默认值
class Settings(BaseSettings):
    """应用配置。

    SecretStr 用在 Redis URL、API key、token 这类敏感字段上。它的意义是：对象被打印或
    转成字符串时不会直接露出明文。真正需要连接 Redis 时，资源层会在很小的范围内调用
    `get_secret_value()` 取出真实值。
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_OPT,
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    app_name: str = Field(default="Redis SRE Agent", description="应用名称。")
    debug: bool = Field(default=False, description="是否开启调试模式。")
    log_level: str = Field(default="INFO", description="日志级别。")
    host: str = Field(default="0.0.0.0", description="服务监听地址。")
    port: int = Field(default=8000, description="服务端口。")
    redis_url: SecretStr = Field(
        default=SecretStr("redis://localhost:6379/0"),
        description="资源层 Redis 地址。默认值不包含密码。",
    )

    openai_api_key: Optional[SecretStr] = Field(default=None, description="OpenAI API key 插槽。")
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI 兼容服务地址插槽。")
    openai_model: str = Field(default="gpt-5", description="后续 Agent 推理模型插槽。")
    openai_model_mini: str = Field(default="gpt-5-mini", description="后续轻量模型插槽。")
    openai_model_nano: str = Field(default="gpt-5-nano", description="后续分类模型插槽。")

    embedding_provider: str = Field(default="openai", description="向量化 provider 插槽。")
    embedding_model: str = Field(default="text-embedding-3-small", description="向量模型插槽。")
    vector_dim: int = Field(default=1536, description="向量维度插槽。")
    embeddings_cache_ttl: Optional[int] = Field(default=86400 * 7, description="向量缓存 TTL。")
    vectorizer_factory: Optional[str] = Field(default=None, description="自定义向量器工厂插槽。")

    task_queue_name: str = Field(default="sre_agent_tasks", description="后续任务队列名称插槽。")
    max_task_retries: int = Field(default=3, description="后续任务重试次数插槽。")
    task_timeout: int = Field(default=TWENTY_MINUTES_IN_SECONDS, description="后续任务超时插槽。")
    max_iterations: int = Field(default=50, description="后续 Agent 最大循环次数插槽。")
    knowledge_max_iterations: int = Field(default=8, description="后续知识问答循环次数插槽。")
    tool_timeout: int = Field(default=60, description="后续工具超时插槽。")
    agent_permission_mode: Literal["read_only", "read_write"] = Field(default="read_only")

    prometheus_url: Optional[str] = Field(default=None, description="Prometheus 地址插槽。")
    grafana_url: Optional[str] = Field(default=None, description="Grafana 地址插槽。")
    grafana_api_key: Optional[SecretStr] = Field(default=None, description="Grafana key 插槽。")
    api_key: Optional[SecretStr] = Field(default=None, description="API 认证 key 插槽。")
    allowed_hosts: list[str] = Field(default=["*"], description="允许的 host。")

    tool_providers: List[str] = Field(default_factory=list, description="工具 provider 插槽。")
    mcp_servers: Dict[str, Union[MCPServerConfig, Dict[str, Any]]] = Field(default_factory=dict)
    skill_roots: List[str] = Field(default_factory=list, description="技能目录插槽。")
    skill_backend_kind: Literal["redis", "custom"] = Field(default="redis")
    skill_backend_class: Optional[str] = Field(default=None)
    skills_api_base_url: Optional[str] = Field(default=None)
    skills_api_tenant_id: Optional[str] = Field(default=None)
    skills_api_project_id: Optional[str] = Field(default=None)
    skills_api_agent_id: Optional[str] = Field(default=None)
    skills_api_token: Optional[SecretStr] = Field(default=None)
    skills_api_timeout_seconds: float = Field(default=15.0)
    target_integrations: TargetIntegrationsConfig = Field(default_factory=TargetIntegrationsConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """加入配置文件来源，同时保留原项目的优先级。"""

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _build_config_file_source(settings_cls),
            file_secret_settings,
        )


settings = Settings()
