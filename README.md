# Redis SRE Agent 诊断切片

这是一个裁剪复刻项目，目标是逐步做出一个能诊断 Redis 问题的 SRE Agent。当前目录不是原
项目的完整复制品，而是按阶段把最小可运行能力迁移过来。

## 项目边界

- 原项目只读参考目录：`D:\developer\redis_sre_prac\original-redis-sre-agent-main`
- 当前可写项目目录：`D:\developer\redis_sre_prac\my_sre_agent`
- 包名保持为 `redis_sre_agent`，这样后续从原项目迁移代码时 import 路径可以尽量少改。
- 当前仍然没有 ToolManager、Redis 诊断工具、Agent 主链路、LangGraph、后台任务、调度、RAG、MCP 或 evaluation。

## 当前阶段已经具备什么

阶段一完成了 Python 项目骨架：可安装、可导入、可编译、可运行最小测试。

阶段二增加了资源层能力：

- `core/config.py`：读取基础配置，支持环境变量、`.env`、YAML/TOML/JSON 配置文件。
- `core/encryption.py`：用 `REDIS_SRE_MASTER_KEY` 和 AES-GCM 加密 Redis 密码、连接串等敏感字段。
- `core/keys.py`：集中构造 Redis key，避免各处手写字符串。
- `core/redisearch.py`：对 RediSearch TAG 查询里的用户输入做转义。
- `core/redis.py`：提供 Redis 客户端工厂和实例/集群索引 schema 插槽。
- `core/instances.py`：提供 Redis 实例模型和保存、查询、删除等资源层函数。
- `core/clusters.py`：提供 Redis 集群模型和保存、查询、删除等资源层函数。

资源层的第一性原理是：上层业务不应该直接处理存储细节。上层只关心“我要保存一个实例”
或“我要按 id 读取一个集群”，资源层负责 key 怎么拼、敏感字段怎么加密、索引字段怎么放、
读取后怎么还原为模型。

## 本地 master key

加密函数需要 `REDIS_SRE_MASTER_KEY`。本地测试可以用下面的方式生成一个临时 key：

```powershell
python -c "import base64, os; print(base64.b64encode(os.urandom(32)).decode())"
```

把命令输出写进本地环境变量即可。不要把真实 key、Redis 密码或带密码的连接串写进文档、
日志、测试输出或提交记录。

## 当前可运行命令

请在当前目录执行：

```powershell
python -m pip install -e .
python -m compileall redis_sre_agent tests
python -m pytest -q
python -c "from redis_sre_agent.core.encryption import encrypt_secret; print(encrypt_secret.__name__)"
python -c "from redis_sre_agent.core.instances import RedisInstance; print(RedisInstance.__name__)"
python -c "from redis_sre_agent.core.clusters import RedisCluster; print(RedisCluster.__name__)"
python -c "from redis_sre_agent.core.redis import get_redis_client; print(get_redis_client.__name__)"
redis-sre-agent status
```

测试全部使用 mock/fake，不访问真实 Redis、OpenAI API 或外部网络。

## 后续阶段才会做什么

后续阶段会逐步补 ToolManager、Redis 诊断工具、Agent 主链路、target discovery、target
binding、调度、RAG、MCP 和 evaluation。当前阶段只完成配置、密钥、Redis 存储和实例/集群
模型这一层。
