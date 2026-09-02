# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability contract for transporting DSpark prompt-context KV.

DSpark's draft MLA cache is produced from target-model auxiliary hidden states
during prefill.  A Decode worker resuming an externally computed prefix may
only speculate when every one of those draft context pages was transported as
part of the same logical cache entry.
"""

from typing import Any

DSPARK_CONTEXT_KV_TRANSPORT = "dspark_context_kv_v1"
DSPARK_CONTEXT_REGION_KIND = "dspark_context_kv_v1"


def dspark_context_kv_transport_enabled(extra_config: dict[str, Any]) -> bool:
    """Return whether the connector explicitly enables DSpark context KV."""
    return (
        extra_config.get("dspark_context_transport_policy")
        == DSPARK_CONTEXT_KV_TRANSPORT
    )
