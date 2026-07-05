"""阶段一的最小命令行入口。

这个文件只负责一件事：证明 `redis-sre-agent` 这个命令能启动到本项目代码。

它不会读取配置，不会连接 Redis，不会调用 OpenAI，也不会执行诊断流程。
这些能力都属于后续阶段。现在保留一个很小的 `status` 命令，是为了让安装后的人能快速确认入口点可用。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from redis_sre_agent import __version__


class ChineseArgumentParser(argparse.ArgumentParser):
    """把 argparse 默认错误出口改成中文。

    `argparse` 是 Python 标准库里的命令行参数解析工具。
    它负责把用户在终端里输入的字符串拆成参数对象。
    标准库默认错误文案是英文；这里不改变解析规则，只把最后展示给人的错误提示换成中文。
    """

    def error(self, message: str) -> None: 
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: 参数错误：{message}\n")
        # 强行终止程序。2 是一种约定俗成的状态码，专门告诉操作系统“这人命令行参数填错了”。
        # 后面的 f"..." 是把程序名字（self.prog）和具体的错误原因（message）拼接成一句中文提示并打印出来。


def build_parser() -> argparse.ArgumentParser:
    """创建阶段一 CLI 参数解析器。

    参数解析器只认识 `--version` 和 `status`。
    这里故意不导入 Agent、Redis 工具或配置模块，因为第一阶段要验证的是项目骨架，不是业务链路。
    """
    parser = ChineseArgumentParser(
        prog="redis-sre-agent",
        usage="redis-sre-agent [--version] [status]",
        description="Redis SRE Agent 诊断切片的阶段一命令行入口。",
        epilog="当前阶段只做本地骨架检查，不会访问 Redis、OpenAI 或外部服务。",
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        help="显示这段帮助信息后退出。",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"redis-sre-agent {__version__}",
        help="显示当前安装的包版本后退出。",
    )
    parser.add_argument(
        "command",
        nargs="?",
        metavar="命令",
        help="可选命令：status。它只检查本地项目骨架是否能启动。",
    )
    parser._positionals.title = "位置参数"
    parser._optionals.title = "可选参数"
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """运行最小 CLI。

    `argv` 是参数列表。测试时可以传入一个列表，真实命令行运行时则让 argparse 自己读取终端参数。
    函数返回整数退出码：`0` 表示正常结束，非零值表示参数错误或运行失败。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "status":
        print("redis-sre-agent 诊断切片：阶段一项目骨架可以正常启动")
        return 0

    if args.command is not None:
        parser.error(f"未知命令：{args.command}。当前阶段只支持 status。")

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
