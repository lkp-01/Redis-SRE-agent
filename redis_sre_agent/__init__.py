"""Redis SRE Agent 诊断切片的阶段一包入口。

这个文件会在执行 `import redis_sre_agent` 时最先运行。

第一阶段只需要确认两件事：
1. Python 能找到 `redis_sre_agent` 这个包。
2. 代码能拿到一个版本号，方便 CLI 或测试显示当前安装的项目。

`importlib.metadata.version()` 会从已安装的包元数据里读取版本。
如果项目还没有按打包方式安装，它可能找不到元数据，所以这里保留一个本地兜底版本号。
"""

from importlib.metadata import PackageNotFoundError, version

_DIST_NAME = "redis-sre-agent"

try:
    __version__ = version(_DIST_NAME)
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
