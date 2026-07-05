# Redis SRE Agent 诊断切片

这是一个裁剪复刻项目，目标是逐步做出一个能诊断 Redis 问题的 SRE Agent。
当前仓库不是原项目的完整复制品，而是先从最小、能运行、能测试的 Python 项目骨架开始。

## 当前边界

- 原项目只读参考目录：`D:\developer\redis_sre_prac\original-redis-sre-agent-main`
- 当前可写项目目录：`D:\developer\redis_sre_prac\my_sre_agent`
- 当前只完成第一阶段：运行契约与裁剪项目骨架。
- 这一阶段只建立 Python 项目元数据、依赖声明、包目录、README 和最小导入测试。

包名继续使用 `redis_sre_agent`。这样做的原因很直接：后续如果要从原项目迁移少量代码，原来的 import 路径可以尽量少改。

命令行入口继续叫 `redis-sre-agent`。但现在它只做本地骨架状态检查，不会连接 Redis，不会访问 OpenAI，也不会访问任何外部服务。

## 现在已经有什么

- `pyproject.toml`：告诉 Python 打包工具这个项目叫什么、需要什么 Python 版本、怎么安装、测试怎么找。
- `redis_sre_agent` 包：这是 Python import 的入口。只要 `import redis_sre_agent` 能成功，说明包路径和安装方式是通的。
- `core`、`cli`、`agent`、`tools`、`targets`：这些目录目前只是插槽。它们存在，是为了让后续阶段能按原项目的大致形状继续往里填代码。
- `redis-sre-agent status`：一个最小命令，用来确认命令行入口能启动。
- `tests/test_imports.py`：只测试导入，不碰网络、不碰 Redis、不碰真实密钥。

## 为什么第一阶段这么小

Python 项目最底层的运行契约其实很简单：

1. 目录里要有一个能被打包工具理解的项目描述文件，也就是 `pyproject.toml`。
2. 项目里要有一个 Python 包目录，也就是这里的 `redis_sre_agent`。
3. 包目录里要有 `__init__.py`，这样 Python 才能把它当成一个可导入的包。
4. 如果要提供命令行命令，就要在 `pyproject.toml` 里声明脚本入口。
5. 测试必须能在没有外部服务的情况下运行，这样骨架问题和外部系统问题不会混在一起。

所以本阶段只验证这些基础契约：能安装、能导入、能编译、能跑测试、命令行入口能启动。

## 当前可运行命令

请在当前目录执行：

```powershell
python -m pip install -e .
python -m compileall redis_sre_agent tests
python -m pytest -q
python -c "import redis_sre_agent; print(redis_sre_agent.__name__)"
redis-sre-agent status
```

这些命令分别做什么：

- `python -m pip install -e .`：用可编辑模式安装当前项目。以后改源码后，不需要每次重新复制安装包。
- `python -m compileall redis_sre_agent tests`：让 Python 尝试编译源码，提前发现语法错误。
- `python -m pytest -q`：运行测试。当前测试只检查包能不能导入。
- `python -c "import redis_sre_agent; print(redis_sre_agent.__name__)"`：用一行 Python 直接验证包名和导入路径。
- `redis-sre-agent status`：验证命令行入口能找到并执行本项目代码。

## 后续阶段才会做什么

后续会逐步补上配置、Redis 存储、ToolManager、Redis 诊断工具、Agent 主链路、调度、RAG、MCP 和 evaluation。

这些能力当前都没有实现。当前项目只是一个干净、可安装、可导入、可测试的第一阶段骨架。
