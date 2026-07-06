"""阶段二加密测试。

测试使用临时随机 master key。key 只在当前进程环境变量里存在，不写入文件或日志。
"""

from __future__ import annotations

import base64
import os
from unittest.mock import patch

import pytest

from redis_sre_agent.core.encryption import (
    EncryptionError,
    decrypt_secret,
    encrypt_secret,
    get_secret_value,
    is_encrypted,
    migrate_plaintext_to_encrypted,
)


@pytest.fixture
def master_key_env():
    key = base64.b64encode(os.urandom(32)).decode("ascii")
    with patch.dict(os.environ, {"REDIS_SRE_MASTER_KEY": key}):
        yield


def test_encrypt_decrypt_roundtrip(master_key_env) -> None:
    plaintext = "FAKE_LOCAL_SECRET"

    encrypted = encrypt_secret(plaintext)

    assert encrypted != plaintext
    assert decrypt_secret(encrypted) == plaintext
    assert is_encrypted(encrypted) is True


def test_same_plaintext_encrypts_to_different_ciphertext(master_key_env) -> None:
    plaintext = "FAKE_LOCAL_SECRET"

    encrypted_1 = encrypt_secret(plaintext)
    encrypted_2 = encrypt_secret(plaintext)

    assert encrypted_1 != encrypted_2
    assert decrypt_secret(encrypted_1) == plaintext
    assert decrypt_secret(encrypted_2) == plaintext


def test_wrong_master_key_fails(master_key_env) -> None:
    encrypted = encrypt_secret("FAKE_LOCAL_SECRET")
    wrong_key = base64.b64encode(os.urandom(32)).decode("ascii")

    with patch.dict(os.environ, {"REDIS_SRE_MASTER_KEY": wrong_key}):
        with pytest.raises(EncryptionError):
            decrypt_secret(encrypted)


def test_missing_and_bad_master_key_errors() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EncryptionError, match="REDIS_SRE_MASTER_KEY"):
            encrypt_secret("FAKE_LOCAL_SECRET")

    short_key = base64.b64encode(b"short").decode("ascii")
    with patch.dict(os.environ, {"REDIS_SRE_MASTER_KEY": short_key}):
        with pytest.raises(EncryptionError, match="32"):
            encrypt_secret("FAKE_LOCAL_SECRET")


def test_get_secret_value_supports_encrypted_and_plaintext(master_key_env) -> None:
    encrypted = encrypt_secret("FAKE_LOCAL_SECRET")

    assert get_secret_value(encrypted) == "FAKE_LOCAL_SECRET"
    assert get_secret_value("LEGACY_FAKE_PLAINTEXT") == "LEGACY_FAKE_PLAINTEXT"
    assert is_encrypted(migrate_plaintext_to_encrypted("LEGACY_FAKE_PLAINTEXT")) is True
