"""Optional GPU timing must invalidate failed samples and recover its readback slots."""

from array import array
from queue import SimpleQueue
from types import SimpleNamespace

import pytest

pytest.importorskip("wgpu")

from mojive.render.webgpu.timing import WgpuTiming


def test_failed_readback_clears_stale_samples_and_recovers(monkeypatch):
    from mojive.render.webgpu import timing

    warnings = []
    monkeypatch.setattr(timing, "log", SimpleNamespace(warning=lambda *args: warnings.append(args)))
    timer = object.__new__(WgpuTiming)
    timer.gpu_ms = {"opaque": 8.0}
    timer.last_error = ""
    timer._released = False
    timer._jobs = SimpleQueue()
    timer._busy = [True]
    buffer = SimpleNamespace(
        map_state="unmapped",
        read_mapped=lambda **kwargs: memoryview(array("Q", (1, 4_000_001))).cast("B"),
    )
    timer._readbacks = [buffer]

    def fail():
        raise RuntimeError("readback unavailable")

    for _ in range(2):
        timer._busy[0] = True
        timer._jobs.put((0, (("opaque", 0, 1),), 2, SimpleNamespace(sync_wait=fail)))
        timer._jobs.put(None)
        timer._readback_loop()
        assert timer.gpu_ms == {}
        assert timer.last_error == "readback unavailable"
        assert not timer._busy[0]
    assert len(warnings) == 1
    timer._jobs.put((0, (("opaque", 0, 1),), 2, SimpleNamespace(sync_wait=lambda: None)))
    timer._jobs.put(None)
    timer._readback_loop()
    assert timer.gpu_ms == {"opaque": 4.0}
    assert timer.last_error == ""
    assert not timer._busy[0]
