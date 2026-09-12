# SPDX-License-Identifier: Apache-2.0
"""Cancellation enters the same broadcast round as other Omni TP inputs."""

import queue
from types import SimpleNamespace

from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def test_tp_abort_is_queued_then_applied_in_broadcast_order(monkeypatch):
    from sglang_omni.scheduling import omni_scheduler

    scheduler = object.__new__(OmniScheduler)
    scheduler.tp_size = 2
    scheduler.is_entry_rank = True
    scheduler.inbox = queue.Queue()
    scheduler._idle_wait_message = None
    scheduler._aborted_request_ids = {"already-aborted"}
    scheduler._completed_request_ids = {}
    scheduler.tp_group = SimpleNamespace(rank=0, ranks=[0, 1])
    scheduler.tp_cpu_group = object()
    aborted = []
    broadcasts = []
    scheduler.abort = aborted.append

    def broadcast(messages, *args, **kwargs):
        broadcasts.append(messages)
        return messages

    monkeypatch.setattr(omni_scheduler, "broadcast_pyobj", broadcast)
    scheduler.inbox.put(
        IncomingMessage(request_id="new", type="new_request", data="payload")
    )
    scheduler.propagate_abort("already-aborted")
    assert aborted == []
    assert scheduler.recv_requests() == ["payload"]
    assert aborted == ["already-aborted"]
    assert [m.type for m in broadcasts[0]] == ["new_request", "abort"]


def test_tp_follower_does_not_enqueue_duplicate_abort():
    scheduler = object.__new__(OmniScheduler)
    scheduler.tp_size = 2
    scheduler.is_entry_rank = False
    scheduler.inbox = queue.Queue()
    scheduler.abort = lambda _: (_ for _ in ()).throw(AssertionError("local abort"))
    scheduler.propagate_abort("r")
    assert scheduler.inbox.empty()
