"""工具枚举的最小兼容定义。

原项目的配置层会引用工具能力枚举，用来描述某个工具属于日志、指标、诊断还是通用工具。
阶段二只需要让配置模型能导入这些名字，不实现 ToolManager，也不注册任何真实工具。
"""

from __future__ import annotations

from enum import Enum


class ToolCapability(str, Enum):
    """工具能力分类插槽。"""

    ADMIN = "admin"
    DIAGNOSTICS = "diagnostics"
    KNOWLEDGE = "knowledge"
    LOGS = "logs"
    METRICS = "metrics"
    UTILITIES = "utilities"


class ToolActionKind(str, Enum):
    """工具动作类型插槽。"""

    READ = "read"
    WRITE = "write"
