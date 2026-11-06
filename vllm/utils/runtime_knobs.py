# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experiment knobs that a running server re-reads from a JSON file.

``VLLM_RUNTIME_KNOBS=<path>`` lets every process poll ``<path>`` (at most every
``_POLL_INTERVAL_S``, by mtime) so an A/B can alternate settings on one warmed
server instead of restarting it per arm. Unset, ``get_runtime_knobs()`` returns
None and nothing is polled. Only knobs whose consumers handle a switch between
steps read from here.
"""

import json
import os
import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_POLL_INTERVAL_S = 0.2


class RuntimeKnobs:
    def __init__(self, path: str):
        self.path = path
        self.values: dict[str, Any] = {}
        self._mtime_ns: int | None = None
        self._next_poll = 0.0

    def poll(self) -> bool:
        """Re-read the file if it changed; True when the values changed."""
        now = time.monotonic()
        if now < self._next_poll:
            return False
        self._next_poll = now + _POLL_INTERVAL_S
        try:
            mtime_ns = os.stat(self.path).st_mtime_ns
        except OSError:
            return False
        if mtime_ns == self._mtime_ns:
            return False
        try:
            with open(self.path) as f:
                values = json.load(f)
        except (OSError, ValueError):
            # Half-written or malformed: keep the previous values, retry later.
            return False
        self._mtime_ns = mtime_ns
        if not isinstance(values, dict) or values == self.values:
            return False
        self.values = values
        logger.info("Runtime knobs from %s: %s", self.path, values)
        return True

    def get(self, key: str, default: Any) -> Any:
        return self.values.get(key, default)


_KNOBS: RuntimeKnobs | None = None
_KNOBS_RESOLVED = False


def get_runtime_knobs() -> RuntimeKnobs | None:
    """This process's knob reader, or None when VLLM_RUNTIME_KNOBS is unset."""
    global _KNOBS, _KNOBS_RESOLVED
    if not _KNOBS_RESOLVED:
        import vllm.envs as envs

        _KNOBS_RESOLVED = True
        if envs.VLLM_RUNTIME_KNOBS:
            _KNOBS = RuntimeKnobs(envs.VLLM_RUNTIME_KNOBS)
    return _KNOBS
