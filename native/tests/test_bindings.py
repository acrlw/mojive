"""Behavioral checks for the optional native binding evaluation modules."""

import gc
import importlib
import itertools
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import pytest

MODULES = ["_mojive_nanobind_probe", "_mojive_pybind_probe"]


@pytest.fixture(params=MODULES)
def binding(request):
    return importlib.import_module(request.param)


def test_mujoco_buffers_and_owned_lifetime(binding):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    assert binding.address(data.qpos) == data.qpos.ctypes.data
    state = binding.OwnedState(data.qpos)
    view = state.view()
    expected = data.qpos.copy()
    mujoco.mj_step(model, data, nstep=10)
    assert not np.array_equal(expected, data.qpos)
    assert np.array_equal(view, expected)
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view.setflags(write=True)
    del state, model, data
    gc.collect()
    assert np.array_equal(view, expected)
    assert np.isfinite(view.sum())


def test_strict_arrays_and_batch(binding):
    with pytest.raises(ValueError):
        binding.FrameBatch(2**63)
    with pytest.raises(OverflowError):
        binding.ping(2**31 - 1)
    matrices = np.tile(np.eye(4, dtype=np.float32), (5101, 1, 1))
    matrices[:, :3, 3] = [1, 2, 3]
    batch = binding.FrameBatch(len(matrices))
    assert batch.pack(matrices) == 5101 * 6
    assert batch.bytes() == 816160
    for invalid in [matrices.astype(np.float64), matrices[:, :, ::-1], matrices.tolist()]:
        with pytest.raises(TypeError):
            batch.pack(invalid)
    with pytest.raises(ValueError):
        batch.pack(matrices.reshape(-1))
    with pytest.raises(ValueError):
        batch.pack(matrices[:1])
    with ThreadPoolExecutor(1) as pool, pytest.raises(RuntimeError):
        pool.submit(batch.pack, matrices).result()
    with pytest.raises(TypeError):
        binding.OwnedState(np.arange(6, dtype=np.float64)[::2])


def test_gil_release_on_owned_state(binding):
    state = binding.OwnedState(np.linspace(0, 1, 10000))
    started = threading.Event()

    def work():
        started.set()
        return state.work(2000)

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(work)
        assert started.wait(2)
        time.sleep(0.01)
        # This Python bytecode must run while the native computation is active.
        assert not future.done()
        ticks = 0
        while not future.done():
            ticks += 1
            time.sleep(0)
        assert ticks > 0
        assert np.isfinite(future.result())


def test_cross_library_cpp_objects_require_explicit_boundary():
    nb, pb = [importlib.import_module(name) for name in MODULES]
    for source, target in [(nb, pb), (pb, nb)]:
        state = source.OwnedState(np.arange(3, dtype=np.float64))
        with pytest.raises(TypeError):
            target.state_size(state)
        assert target.state_size(target.OwnedState(state.view())) == 3


@pytest.mark.parametrize("order", list(itertools.permutations(["mujoco", *MODULES])))
def test_import_orders(order):
    code = "\n".join(f"import {name}" for name in order)
    code += "\nimport numpy as np\n"
    code += "a=np.ones(8)\n"
    for name in MODULES:
        code += f"assert {name}.address(a)==a.ctypes.data\n"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30, env=os.environ)
