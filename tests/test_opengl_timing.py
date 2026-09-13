"""GPU profiling remains bounded when timestamp results are delayed."""

from mojive.render.opengl import timing


class DelayedQueries:
    has_query = True

    def __init__(self):
        self.created = 0
        self.deleted = []
        self.ready = False

    def gen_query(self):
        self.created += 1
        return self.created

    def begin_time_query(self, query):
        pass

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
