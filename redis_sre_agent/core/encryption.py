"""敏感字段加密工具。

资源层会把 Redis 连接串、Redis Enterprise 管理密码、扩展 secrets 等字段存到 Redis。
这些字段不能明文落库。这里沿用原项目的 envelope encryption 思路：

1. 每次加密先生成一个随机数据密钥 DEK。
2. 用 DEK 和 AES-GCM 加密真实 secret。
3. 再用环境变量 `REDIS_SRE_MASTER_KEY` 里的主密钥加密 DEK。
4. 最终保存密文、nonce、被包裹的 DEK 和版本号。

这样即使 Redis 里的数据被单独拿走，没有主密钥也不能还原 secret。
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

CURRENT_VERSION = "v1"


class EncryptionError(Exception):
    """加密或解密失败。异常消息只说明失败原因，不包含明文 secret。"""


def _get_master_key() -> bytes:
    """从环境变量读取 32 字节主密钥。"""

    master_key_b64 = os.getenv("REDIS_SRE_MASTER_KEY")
    if not master_key_b64:
        raise EncryptionError(
            "缺少 REDIS_SRE_MASTER_KEY。请在本地测试环境生成 32 字节随机 key 后再运行加密。"
        )

    try:
        master_key = base64.b64decode(master_key_b64, validate=True)
    except binascii.Error as exc:
        raise EncryptionError("REDIS_SRE_MASTER_KEY 不是合法的 base64 字符串。") from exc

    if len(master_key) != 32:
        raise EncryptionError(
            f"REDIS_SRE_MASTER_KEY 解码后必须是 32 字节，当前是 {len(master_key)} 字节。"
        )
    return master_key


def encrypt_secret(plaintext: str) -> str:
    """加密一个 secret，返回 base64 包裹的 JSON envelope。"""

    try:
        master_key = _get_master_key()
        dek = AESGCM.generate_key(bit_length=256)

        dek_aes = AESGCM(dek)
        nonce = os.urandom(12)
        ciphertext = dek_aes.encrypt(nonce, plaintext.encode("utf-8"), None)

        master_aes = AESGCM(master_key)
        dek_nonce = os.urandom(12)
        wrapped_dek = master_aes.encrypt(dek_nonce, dek, None)

        envelope = {
            "version": CURRENT_VERSION,
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "wrapped_dek": base64.b64encode(wrapped_dek).decode("ascii"),
            "dek_nonce": base64.b64encode(dek_nonce).decode("ascii"),
        }
        return base64.b64encode(json.dumps(envelope).encode("utf-8")).decode("ascii")
    except EncryptionError:
        raise
    except Exception as exc:
        logger.error("加密 secret 失败。")
        raise EncryptionError("加密失败。") from exc


def decrypt_secret(encrypted_data: str) -> str:
    """解密 `encrypt_secret()` 生成的 envelope。"""

    try:
        master_key = _get_master_key()
        envelope_json = base64.b64decode(encrypted_data, validate=True).decode("utf-8")
        envelope = json.loads(envelope_json)

        version = envelope.get("version")
        if version != CURRENT_VERSION:
            raise EncryptionError(f"不支持的密文版本：{version}。")

        ciphertext = base64.b64decode(envelope["ciphertext"], validate=True)
        nonce = base64.b64decode(envelope["nonce"], validate=True)
        wrapped_dek = base64.b64decode(envelope["wrapped_dek"], validate=True)
        dek_nonce = base64.b64decode(envelope["dek_nonce"], validate=True)

        master_aes = AESGCM(master_key)
        dek = master_aes.decrypt(dek_nonce, wrapped_dek, None)
        dek_aes = AESGCM(dek)
        plaintext = dek_aes.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")
    except EncryptionError:
        raise
    except InvalidTag as exc:
        logger.error("解密 secret 失败：认证标签校验失败。")
        raise EncryptionError("解密失败：master key 不匹配或密文已损坏。") from exc
    except Exception as exc:
        logger.error("解密 secret 失败。")
        raise EncryptionError("解密失败：密文格式不正确或内容已损坏。") from exc


def is_encrypted(data: str) -> bool:
    """判断字符串是否像本模块生成的密文 envelope。"""

    try:
        envelope_json = base64.b64decode(data, validate=True).decode("utf-8")
        envelope = json.loads(envelope_json)
        return (
            envelope.get("version") == CURRENT_VERSION
            and "ciphertext" in envelope
            and "wrapped_dek" in envelope
            and "nonce" in envelope
            and "dek_nonce" in envelope
        )
    except Exception:
        return False


def migrate_plaintext_to_encrypted(plaintext: str) -> str:
    """把历史明文迁移成密文。如果已经是本模块密文，就原样返回。"""

    if is_encrypted(plaintext):
        return plaintext
    return encrypt_secret(plaintext)


def get_secret_value(data: str) -> str:
    """读取 secret 明文。

    已加密的密文会被解密；历史明文会原样返回。这样资源层可以逐步迁移，不需要一次性重写
    所有旧数据。
    """

    if not data:
        return data
    if is_encrypted(data):
        return decrypt_secret(data)
    logger.warning("检测到明文 secret，建议迁移为加密格式。")
    return data
