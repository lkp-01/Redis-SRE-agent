"""
若解析出query：会先调用invoke解析出query、version等命令，再调用getcommand，若query会直接掉query.py
因此本文件，总共规定了三大类命令行的方法，lazygroup用来和用户终端输入进行对齐，就这点东西。关键在于继承了click
"""

# 根命令只负责 Click 入口、命令注册和 lazy loading。具体 query 生命周期在
# redis_sre_agent.cli.query 中，避免把 Agent 主逻辑堆进 main.py。

from __future__ import annotations

import importlib

import click

from redis_sre_agent import __version__
from redis_sre_agent.cli.logging_utils import (
    configure_cli_logging,
    log_cli_exception,
    was_cli_exception_logged,
)

# 【模块加载时第一次调用】在 CLI 顶层提早配置日志系统，确保在进入 Click 路由前的一切潜在初始化错误能被捕获
configure_cli_logging()

# 定义动态加载（懒加载）的命令映射表，Key 为命令行命令，Value 为实际代码文件和函数的路径字符串
_COMMANDS = {
    "query": "redis_sre_agent.cli.query:query",
}

# 定义内置的、无需动态加载的命令集合
_BUILTIN_COMMANDS = {"version", "status"}


# 使用 click 装饰器将普通函数转换为一个 CLI 命令
@click.command()
def version() -> None:
    """Show the Redis SRE Agent version."""
    # 在控制台安全打印当前 Agent 的版本号
    click.echo(f"redis-sre-agent {__version__}")


# 使用 click 装饰器将普通函数转换为另一个 CLI 命令
@click.command()
def status() -> None:
    """Show the current diagnostic slice status."""
    # 在控制台打印当前诊断系统的就绪状态
    click.echo("redis-sre-agent 诊断切片：Stage 5 LangGraph Agent 主链路可以正常启动")


# 继承 click.MultiCommand 自定义一个命令组类，用于支持“按需延迟加载（Lazy Loading）”
class LazyGroup(click.MultiCommand):
    """Lazy loading of CLI commands to avoid hard dependencies at top level."""

    # 必须要重写的方法，用于告诉 Click 当前 CLI 都有哪些命令可用（比如在用户输入 --help 时展示）
    def list_commands(self, ctx):
        # 将动态加载命令表的 Key 列表与内置命令集合合并，并转换为普通 list 返回
        return list(_COMMANDS.keys()) + list(_BUILTIN_COMMANDS)

    # 核心方法：当用户在终端敲下某个命令名时（如 redis-sre-agent query），Click 会调用此方法动态加载该命令
    def get_command(self, ctx, name):
        # 【此处为第二次调用】在加载子命令前强行刷新日志配置，防止被第三方依赖或动态加载的子模块篡改或覆盖
        configure_cli_logging()

        # 如果用户输入的是内置的 version 命令，直接返回顶层已定义好的 version 函数对象
        if name == "version":
            return version

        # 如果用户输入的是内置的 status 命令，直接返回顶层已定义好的 status 函数对象
        if name == "status":
            return status

        # 尝试从动态命令映射表中获取目标子命令的路径字符串
        target = _COMMANDS.get(name)
        # 如果映射表中也没有，说明用户输入了一个不存在的命令，返回 None 让 Click 框架去报错（Command not found）
        if not target:
            return None

        # 将路径字符串（如 "redis_sre_agent.cli.query:query"）以第一个冒号为界拆分为：模块路径 和 函数名
        module_path, attr = target.split(":", 1)
        # 核心魔法：利用 Python 内置库动态将这个模块导入进内存（此时才会真正去读取和解析 query.py）
        mod = importlib.import_module(module_path)
        # 从刚刚导入的模块对象中，根据函数名获取具体的函数引用并返回给 Click 框架去执行
        return getattr(mod, attr)

    # 核心方法：承载了 CLI 执行的最终生命周期，并在这架设了全局最外层的“天网”来防御未预期异常
    def invoke(self, ctx):
        try:
            # 调用父类的 invoke 方法，这会顺着 click 框架真正去执行对应的子命令逻辑
            return super().invoke(ctx)
        # 捕获所有 Click 框架自身正常退出的信号异常（如参数输错、主动中止、SystemExit 等）
        except (click.exceptions.Exit, click.ClickException, click.Abort, SystemExit):
            # 这类是正常的退出信号，直接原封不动向上抛出，不记日志，保持控制台干净
            raise
        # 捕获所有其他由于程序 Bug 导致的、未被底层捕获的非预期重大崩溃
        except Exception as exc:
            # 配合状态跟踪：如果这个异常对象在底层（如 query.py 里）没有被记录过日志
            if not was_cli_exception_logged(exc):
                # 调用自定义工具，以 EXCEPTION（带堆栈）级别记录一条顶层全局兜底日志，确保留下错误证据
                log_cli_exception(__name__, "CLI command failed", exc)
            # 日志写完后，再次将异常向上抛出，维持程序原有的崩溃退出状态
            raise

# 解释器在加载模块时，实际上执行了这行代码
# main = click.command(cls=LazyGroup)(main)
@click.command(cls=LazyGroup)
def main() -> None:
    """Redis SRE Agent CLI."""

    pass


if __name__ == "__main__":
    main()
