# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils


@dataclass
class _TPGroup:
    rank_in_group: int


def test_dspark_draft_uses_tp_local_rank(monkeypatch):
    parallel_config = object()
    captured: dict[str, object] = {}

    def replace(config, **kwargs):
        captured.update(config=config, **kwargs)
        return "draft_parallel_config"

    monkeypatch.setattr(dspark_utils, "replace", replace)
    monkeypatch.setattr(
        dspark_utils,
        "get_tp_group",
        lambda: _TPGroup(rank_in_group=7),
    )

    result = dspark_utils._make_draft_parallel_config(parallel_config)

    assert result == "draft_parallel_config"
    assert captured == {
        "config": parallel_config,
        "pipeline_parallel_size": 1,
        "rank": 7,
    }
