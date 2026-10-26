# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Early reporting of finished remote KV loads (``VLLM_KV_EARLY_RECV``).

A worker reports a finished KV load in the KVConnectorOutput of the step that
polled it. Under async scheduling that output reaches the scheduler only after
the step's GPU work, when the next step is already scheduled, so a request whose
KV has arrived waits two to three steps before it runs. Here, each time a
step's output has been handed back to the engine core, the worker (waiting for
its next RPC) polls the connector once and pushes the finished loads over a ZMQ
socket; the engine core drains the socket right before its next schedule().
"""

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

    def drain(self) -> set[str]:
        req_ids: set[str] = set()
        while True:
            try:
                req_ids.update(self._socket.recv_pyobj(zmq.NOBLOCK))
            except zmq.Again:
                return req_ids

    def close(self) -> None:
        self._socket.close(linger=0)


class EarlyKVRecvSender:
    """Worker end: polls the KV connector once per step while waiting for an RPC."""

    def __init__(self, path: str) -> None:
        self._socket = make_zmq_socket(zmq.Context.instance(), path, zmq.PUSH)
        # Set by the async output thread once a step's output is handed back.
        self.step_done = threading.Event()

    def wait_for_rpc(self, rpc_mq: MessageQueue) -> None:
        """Poll once after an output completes, unless the next RPC is ready."""
        if not has_kv_transfer_group():
            return
        connector = get_kv_transfer_group()
        while connector.has_pending_recving() and not rpc_mq.readable():
            if not self.step_done.is_set():
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
