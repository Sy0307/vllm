# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report completed KV pulls between model outputs and the next schedule()."""

import threading

import torch
import zmq

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.utils import sched_yield
from vllm.utils.network_utils import get_open_zmq_ipc_path, make_zmq_socket


class EarlyKVRecvReceiver:
    """Engine core end: the request ids whose KV loads finished."""

    def __init__(self) -> None:
        self.path = get_open_zmq_ipc_path()
        self._socket = make_zmq_socket(zmq.Context.instance(), self.path, zmq.PULL)

    def drain(self) -> list[set[str]]:
        notifications: list[set[str]] = []
        while True:
            try:
                notifications.append(self._socket.recv_pyobj(zmq.NOBLOCK))
            except zmq.Again:
                return notifications

    def close(self) -> None:
        self._socket.close(linger=0)


class EarlyKVRecvSender:
    """Worker end: polls the KV connector once per step while waiting for an RPC."""

    def __init__(self, path: str) -> None:
        self._socket = make_zmq_socket(zmq.Context.instance(), path, zmq.PUSH)
        # Set by the async output thread once a step's output is handed back.
        self.step_done = threading.Event()

    def wait_for_rpc(self, rpc_mq: MessageQueue) -> None:
        """Poll at most once per completed output, on the worker thread."""
        if not has_kv_transfer_group():
            return
        connector = get_kv_transfer_group()
        while connector.has_pending_recving():
            if not self.step_done.is_set():
                if rpc_mq.readable():
                    return
                sched_yield()
                continue
            self.step_done.clear()
            with torch.inference_mode():
                req_ids = connector.poll_finished_recving()
            if req_ids:
                self._socket.send_pyobj(req_ids)
            return

    def close(self) -> None:
        self._socket.close(linger=0)
