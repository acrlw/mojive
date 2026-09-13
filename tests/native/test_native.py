"""Compatibility and ownership at the private native runtime boundary."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mojive import math3d


def test_camera_matrices_match_existing_python_conventions(native):
    rng = np.random.default_rng(42)
    for _ in range(100):
        eye, target = rng.normal(size=(2, 3)).astype(np.float32)
        actual = native.look_at(eye.tolist(), target.tolist(), [0, 0, 1])
        assert actual.shape == (4, 4) and actual.dtype == np.float32
        assert actual.flags.c_contiguous
        np.testing.assert_allclose(actual, math3d.look_at(eye, target, [0, 0, 1]), atol=2e-6)
        actual[0, 0] = 100
        assert native.look_at(eye.tolist(), target.tolist(), [0, 0, 1])[0, 0] != 100
    np.testing.assert_allclose(
        native.perspective(0.9, 1.6, 0.1, 100), math3d.perspective(0.9, 1.6, 0.1, 100), atol=2e-6
    )
    np.testing.assert_allclose(
        native.orthographic(4, 2, 0.1, 100), math3d.orthographic(2, 2, 0.1, 100), atol=2e-6
    )
    for invalid in [float("nan"), float("inf"), 0, -1]:
        with pytest.raises(ValueError):
            native.perspective(invalid, 1, 0.1, 100)


def test_logs_own_unicode_metadata_and_subscription_results(native):
    options = native.LogOptions()
    options.capacity = 3
    log = native.Log(options)
    for index in range(7):
        log.publish(native.LogLevel.WARNING, "资源加载", f"加载 {index}", timestamp_ns=123000)
    first = log.read(limit=2)
    assert first.missed == 4 and first.next == 6
    assert [record.message for record in first.records] == ["加载 4", "加载 5"]
    assert log.read(limit=2).next == first.next
    assert log.read(first.next).records[0].message == "加载 6"
    record = first.records[0]
    assert record.origin == native.LogOrigin.PYTHON
    assert record.timestamp_ns == 123000
    assert record.component == "资源加载" and record.thread_id
    log.close()
    assert not log.publish(native.LogLevel.ERROR, "late", "ignored")
    del log, first
    gc.collect()
    assert record.message == "加载 4"


def test_native_logging_threads_close_and_runtime_isolation(native, tmp_path):
    path = tmp_path / "native.log"
    options = native.LogOptions()
    options.file = str(path)
    options.output_queue = 4096
    with native.Log(options) as log, native.Log() as peer:
        assert log.runtime_id != peer.runtime_id
        with ThreadPoolExecutor(4) as pool:
            results = list(
                pool.map(lambda i: log.publish(native.LogLevel.INFO, "worker", str(i)), range(400))
            )
        assert all(results) and log.stats().published == 400
        assert peer.stats().published == 0
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda _: log.close(), range(2)))
        assert log.stats().closed and not peer.stats().closed
    # The file completes without Python consuming any records.
    assert len(path.read_text().splitlines()) == 400
    assert log.stats().output_errors == 0 and log.stats().output_dropped == 0
    with pytest.raises(RuntimeError, match="closed"):
        log.__enter__()


def test_invalid_log_configuration_and_utf8_truncation(native):
    options = native.LogOptions()
    options.capacity = 0
    with pytest.raises(ValueError):
        native.Log(options)
    options.capacity = 4
    options.message_bytes = 4
    with native.Log(options) as log:
        log.publish(native.LogLevel.INFO, "text", "中英文")
        assert log.read().records[0].message == "中"
        assert log.stats().truncated == 1
        with pytest.raises(ValueError):
            log.read(after=100)
        with pytest.raises(ValueError):
            log.read(limit=0)


@pytest.mark.parametrize("native_first", [False, True])
def test_native_import_coexists_with_mujoco_and_loguru(native, native_first, tmp_path):
    module_path = os.environ["MOJIVE_NATIVE_TEST_MODULE"]
    script = f"""
import importlib.util
from loguru import logger
messages = []
sink = logger.add(messages.append)
def load_native():
    spec = importlib.util.spec_from_file_location('mojive._native', {module_path!r})
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native
if {native_first!r}:
    native = load_native()
import mujoco
if not {native_first!r}:
    native = load_native()
model = mujoco.MjModel.from_xml_string('<mujoco/>')
options = native.LogOptions()
options.file = {str(tmp_path / "shutdown.log")!r}
log = native.Log(options)
log.publish(native.LogLevel.INFO, 'native', 'before shutdown')
logger.info('host logger preserved')
assert len(messages) == 1
logger.remove(sink)
# Deliberately let interpreter teardown release the native logger.
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
    assert "before shutdown" in (tmp_path / "shutdown.log").read_text()
