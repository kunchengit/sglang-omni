# SPDX-License-Identifier: Apache-2.0
"""Lifecycle/TLS checks; real native execution is opt-in in the GPU suite."""

from __future__ import annotations

import ast
import inspect
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.llada2_uni.components import decoder_runtime as runtime


@pytest.fixture
def owned():
    events = []

    tls = threading.local()

    def set_policy(**kwargs):
        tls.state = SimpleNamespace(**kwargs)

    def model_cleanup():
        events.append("model")

    def world_cleanup():
        events.append("world")

    ps = SimpleNamespace(
        destroy_model_parallel=model_cleanup,
        destroy_distributed_environment=world_cleanup,
    )
    args = SimpleNamespace(set_global_server_args=lambda value: events.append("args"))
    context = SimpleNamespace(reset_context=lambda: events.append("context"))
    precision = SimpleNamespace(
        _mixed_precision_state=tls, set_mixed_precision_policy=set_policy
    )
    handle = runtime.DecoderRuntimeHandle(
        torch.device("cpu"), torch.bfloat16, ps, args, context, precision
    )
    handle._world_started = handle._model_started = handle._published = True
    return handle, ps, tls, events


def test_cross_thread_compute_and_close_do_not_touch_factory_tls(owned):
    handle, _, tls, events = owned
    main_state = object()
    tls.state = main_state
    main_thread = threading.get_ident()

    def scheduler_work():
        assert threading.get_ident() != main_thread
        assert not hasattr(tls, "state")
        with handle.compute_context():
            assert tls.state.param_dtype == torch.bfloat16
        assert not hasattr(tls, "state")
        handle.close()
        assert not hasattr(tls, "state")

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(scheduler_work).result()
    assert tls.state is main_state
    handle.close()
    assert events == ["model", "world", "args", "context"]


def test_compute_precision_restored_after_failure(owned):
    handle, _, tls, _ = owned
    previous = object()
    tls.state = previous
    with pytest.raises(ValueError, match="compute failure"), handle.compute_context():
        raise ValueError("compute failure")
    assert tls.state is previous


def test_close_attempts_world_and_config_after_model_cleanup_error(owned):
    handle, ps, _, events = owned

    def fail():
        events.append("model")
        raise RuntimeError("model cleanup error")

    ps.destroy_model_parallel = fail
    with pytest.raises(RuntimeError, match="model cleanup error"):
        handle.close()
    assert events == ["model", "world", "args", "context"]


def test_context_manager_preserves_original_error_if_cleanup_also_fails(owned):
    handle, ps, _, events = owned

    def fail():
        raise RuntimeError("cleanup failure")

    ps.destroy_model_parallel = fail
    original = ValueError("original failure")
    with pytest.raises(ValueError) as error, handle:
        raise original
    assert error.value is original
    assert isinstance(error.value.__cause__, RuntimeError)
    assert events == ["world", "args", "context"]


def test_no_initialization_means_no_teardown(owned):
    handle, _, _, events = owned
    handle._world_started = handle._model_started = handle._published = False
    handle.close()
    assert events == []


def test_refuse_preexisting_default_world_before_import(monkeypatch):
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="dedicated process"):
        runtime.initialize_decoder_runtime("unused", sp_rank=0, sp_size=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nccl_port": 0},
        {"nccl_port": 65536},
        {"nccl_port": True},
        {"gpu_id": -1},
        {"dist_timeout": 0},
        {"dtype": torch.int32},
        {"sp_size": 2, "ulysses_degree": 2, "stage_role": "leader"},
    ],
)
def test_invalid_initialization_contract(kwargs):
    args = {"model_path": "unused", "sp_rank": 0, "sp_size": 1}
    args.update(kwargs)
    with pytest.raises(ValueError):
        runtime.initialize_decoder_runtime(**args)


def test_target_source_accepts_real_configuration_and_init_keywords():
    root = os.environ.get("SGLANG_SOURCE_DIR")
    if not root:
        pytest.skip("set SGLANG_SOURCE_DIR to audit the exact upstream signatures")
    root = Path(root) / "python/sglang"
    own = ast.parse(inspect.getsource(runtime.initialize_decoder_runtime))
    calls = {
        node.func.attr: node
        for node in ast.walk(own)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    args_tree = ast.parse(
        (root / "multimodal_gen/runtime/server_args/server_args.py").read_text()
    )
    args_cls = next(
        node
        for node in args_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ServerArgs"
    )
    fields = {
        node.target.id for node in args_cls.body if isinstance(node, ast.AnnAssign)
    }
    assert {kw.arg for kw in calls["ServerArgs"].keywords} <= fields
    ps_tree = ast.parse(
        (root / "multimodal_gen/runtime/distributed/parallel_state.py").read_text()
    )
    for name in ("init_distributed_environment", "initialize_model_parallel"):
        fn = next(
            node
            for node in ps_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        accepted = {arg.arg for arg in fn.args.args + fn.args.kwonlyargs}
        assert {kw.arg for kw in calls[name].keywords} <= accepted
