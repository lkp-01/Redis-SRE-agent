"""Redis key 构造工具。

Redis 本质上是一个 key-value 存储。只要 key 拼错，数据就会散落到另一个位置，排查起来
很困难。资源层把 key 构造集中到一个类里，后续代码只调用函数，不手写字符串。
"""


class RedisKeys:
    """统一构造 Redis key。"""

    PREFIX_SRE = "sre"
    PREFIX_KNOWLEDGE = "sre_knowledge"
                    
    @staticmethod #静态方法，有了它就不需要实例化这个类了，比如用写 rk = RedisKeys()，直接用 RedisKeys.方法名() 就能用它。
    def instance_doc(instance_id: str) -> str:
        """实例 hash 文档 key。"""

        return f"sre_instances:{instance_id}"

    @staticmethod
    def cluster_doc(cluster_id: str) -> str:
        """集群 hash 文档 key。"""

        return f"sre_clusters:{cluster_id}"

    @staticmethod
    def user_instances(user_id: str) -> str:
        return f"sre:user:{user_id}:instances"

    @staticmethod
    def user_clusters(user_id: str) -> str:
        return f"sre:user:{user_id}:clusters"

    @staticmethod
    def thread_status(thread_id: str) -> str:
        return f"sre:thread:{thread_id}:status"

    @staticmethod
    def thread_messages(thread_id: str) -> str:
        return f"sre:thread:{thread_id}:messages"

    @staticmethod
    def thread_context(thread_id: str) -> str:
        return f"sre:thread:{thread_id}:context"

    @staticmethod
    def thread_metadata(thread_id: str) -> str:
        return f"sre:thread:{thread_id}:metadata"

    @staticmethod
    def threads_index() -> str:
        return "sre:threads:index"

    @staticmethod
    def thread_instances(thread_id: str) -> str:
        return f"sre:thread:{thread_id}:instances"

    @staticmethod
    def message_decision_trace(message_id: str) -> str:
        """一条 assistant 消息对应的工具决策轨迹。"""

        return f"sre:message:{message_id}:decision_trace"

    @staticmethod
    def task_status(task_id: str) -> str:
        return f"sre:task:{task_id}:status"

    @staticmethod
    def task_updates(task_id: str) -> str:
        return f"sre:task:{task_id}:updates"

    @staticmethod
    def task_result(task_id: str) -> str:
        return f"sre:task:{task_id}:result"

    @staticmethod
    def task_error(task_id: str) -> str:
        return f"sre:task:{task_id}:error"

    @staticmethod
    def knowledge_document(doc_id: str) -> str:
        return f"sre_knowledge:{doc_id}"

    @staticmethod
    def knowledge_chunk(document_hash: str, chunk_index: int) -> str:
        return f"sre_knowledge:{document_hash}:chunk:{chunk_index}"

    @staticmethod
    def knowledge_documents() -> str:
        return "sre_knowledge:documents"

    @staticmethod
    def schedule_key(schedule_id: str) -> str:
        return f"sre_schedules:{schedule_id}"

    @staticmethod
    def all_thread_keys(thread_id: str) -> dict[str, str]:
        """返回线程相关 key，给后续线程管理模块预留稳定形状。"""

        return {
            "status": RedisKeys.thread_status(thread_id),
            "messages": RedisKeys.thread_messages(thread_id),
            "context": RedisKeys.thread_context(thread_id),
            "metadata": RedisKeys.thread_metadata(thread_id),
        }
