# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector, KVConnector


def test_async_load_zeroing_capability_is_proxied() -> None:
    inactive = KVConnector()
    assert not inactive.requires_block_zeroing_before_async_load

    active = object.__new__(ActiveKVConnector)
    active._disabled = False
    active.kv_connector = SimpleNamespace(
        requires_block_zeroing_before_async_load=True
    )
    assert active.requires_block_zeroing_before_async_load

    active._disabled = True
    assert not active.requires_block_zeroing_before_async_load
