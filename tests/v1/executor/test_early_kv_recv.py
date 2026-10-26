# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import zmq

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.executor.early_kv_recv import EarlyKVRecvReceiver


def test_receiver_consumes_notifications_once():
    receiver = EarlyKVRecvReceiver()
    sender = make_zmq_socket(zmq.Context.instance(), receiver.path, zmq.PUSH)
    try:
        assert receiver.drain() == set()
        sender.send_pyobj({"a", "b"})
        sender.send_pyobj({"b", "c"})
        received: set[str] = set()
        deadline = time.monotonic() + 5
        while received != {"a", "b", "c"} and time.monotonic() < deadline:
            received |= receiver.drain()
        assert received == {"a", "b", "c"}
        assert receiver.drain() == set()
    finally:
        sender.close(linger=0)
        receiver.close()


def test_message_queue_readable():
    writer = MessageQueue(n_reader=1, n_local_reader=1, max_chunks=2)
    reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    writer.wait_until_ready()
    reader.wait_until_ready()
    try:
        assert not reader.readable()
        writer.enqueue("rpc")
        assert reader.readable()
        assert reader.readable()
        assert reader.dequeue(timeout=5) == "rpc"
        assert not reader.readable()
    finally:
        reader.shutdown()
        writer.shutdown()
