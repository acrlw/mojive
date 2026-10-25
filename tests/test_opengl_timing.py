"""GPU profiling remains bounded when timestamp results are delayed."""

import pytest

from mojive.render.opengl import timing


class DelayedQueries:
    has_query = True

    def __init__(self):
        self.created = 0
        self.deleted = []
        self.ready = False
        self.submitted = []

    def gen_query(self):
        self.created += 1
        return self.created

    def begin_time_query(self, query):
        self.submitted.append(query)

    def end_time_query(self):
        pass

    def query_ready(self, query, destination):
        return self.ready

    def query_result_ns(self, query, destination):
        return 1_000_000

    def delete_query(self, query):
        self.deleted.append(query)

    def push_debug_group(self, name):
        pass

    def pop_debug_group(self):
        pass


def test_delayed_gpu_samples_remain_bounded_and_recover_without_stalling(monkeypatch):
    driver = DelayedQueries()
    monkeypatch.setattr(timing.G, "native", lambda: driver)
    monkeypatch.setattr(timing.FrameTiming, "_probe", lambda self: True)
    timer = timing.FrameTiming(None)
    try:
        for _ in range(100):
            timer.begin_frame()
            with timer.scope("scene"):
                pass
            timer.collect()
            assert timer.cpu_table()["scene"] >= 0
        assert driver.created == timing._POOL_LIMIT
        assert len(timer._timers["scene"].pending) == timing._POOL_LIMIT
        driver.ready = True
        for _ in range(20):
            timer.begin_frame()
            timer.collect()
            with timer.scope("scene"):
                pass
        assert timer.gpu_table() == {"scene": 1.0}
        assert driver.created == timing._POOL_LIMIT
    finally:
        timer.release()
    assert sorted(driver.deleted) == list(range(1, driver.created + 1))


@pytest.mark.parametrize("interval, samples", [(1, 20), (4, 5)])
def test_gpu_sampling_preserves_cpu_frames_and_reuses_completed_queries(
    monkeypatch, interval, samples
):
    driver = DelayedQueries()
    driver.ready = True
    monkeypatch.setattr(timing.G, "native", lambda: driver)
    monkeypatch.setattr(timing.FrameTiming, "_probe", lambda self: True)
    timer = timing.FrameTiming(None, gpu_sample_interval=interval)
    try:
        for frame in range(20):
            timer.begin_frame()
            with timer.scope("scene"):
                pass
            timer.collect()
            assert "scene" in timer.cpu_table()
            if frame >= timing.LATENCY:
                # Result age uses rendered frames, not the number of GPU samples.
                assert timer.gpu_table()["scene"] == 1.0
        assert len(driver.submitted) == samples
        assert driver.created <= timing.LATENCY + 1
    finally:
        timer.release()
    assert sorted(driver.deleted) == list(range(1, driver.created + 1))


def test_new_pass_is_sampled_before_next_periodic_frame(monkeypatch):
    driver = DelayedQueries()
    driver.ready = True
    monkeypatch.setattr(timing.G, "native", lambda: driver)
    monkeypatch.setattr(timing.FrameTiming, "_probe", lambda self: True)
    timer = timing.FrameTiming(None, gpu_sample_interval=8)
    try:
        for frame in range(4):
            timer.begin_frame()
            with timer.scope("scene"):
                pass
            if frame > 0:
                with timer.scope("selection"):
                    pass
            timer.collect()
        assert len(driver.submitted) == 2
        assert timer.gpu_table() == {"scene": 1.0, "selection": 1.0}
    finally:
        timer.release()


def test_gpu_sampling_rejects_zero_interval():
    with pytest.raises(ValueError, match="positive"):
        timing.FrameTiming(None, gpu_sample_interval=0)
