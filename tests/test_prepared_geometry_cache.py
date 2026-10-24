"""Scene-row churn preserves hot layouts without changing paint or audit evidence."""

from copy import deepcopy

import pytest

from mojive.design.native.button_layout import button_metrics
from mojive.design.native.geometry import NativePartStyle
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore


@pytest.mark.parametrize("scale", (1.0, 1.25, 2.0))
def test_changing_rows_keep_hot_geometry_and_evict_cold_entries(tmp_path, scale):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    style = NativePartStyle(runtime, ("button", "tree_label"), authored_defaults=True).freeze()
    cold = style.box(-100, 0, 180, 28, scale=scale)
    hot = style.box(20, 30, 180, 28, scale=scale)
    for row in range(600):
        assert style.box(20, 30, 180, 28, scale=scale) is hot
        style.box(20, 100 + row * 28, 180 + row % 13, 28, scale=scale)
        assert len(style._geometry_cache) <= 256
    assert style.box(20, 30, 180, 28, scale=scale) is hot
    dependencies = {}
    for _value, reads, grouped in style._geometry_cache.values():
        previous = dependencies.setdefault(frozenset(reads), (reads, grouped))
        assert previous[0] is reads and previous[1] is grouped
    assert len(dependencies) < 10
    assert len(style._geometry_dependencies) <= 64
    rebuilt = style.box(-100, 0, 180, 28, scale=scale)
    assert rebuilt is not cold and rebuilt == cold

    runtime.track_field_reads = True
    runtime.begin_frame()
    cached = style.box(20, 30, 180, 28, scale=scale)
    cached_reads = deepcopy(runtime.used_fields)
    assert cached_reads
    style._geometry_cache.clear()
    runtime.begin_frame()
    fresh = style.box(20, 30, 180, 28, scale=scale)
    assert fresh == cached
    assert runtime.used_fields == cached_reads


def test_dependency_reuse_preserves_argument_dependent_reads(tmp_path):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    runtime.track_field_reads = True
    style = NativePartStyle(runtime, ("button", "tree_label"), authored_defaults=True).freeze()
    for intrinsic in (None, (180, 22), (20, 40), None):
        runtime.begin_frame()
        expected = NativePartStyle(runtime, ("button", "tree_label"), authored_defaults=True).box(
            5, 12, 100, 28, intrinsic=intrinsic
        )
        expected_reads = deepcopy(runtime.used_fields)
        runtime.begin_frame()
        assert style.box(5, 12, 100, 28, intrinsic=intrinsic) == expected
        assert runtime.used_fields == expected_reads


@pytest.mark.parametrize(
    "icon,label,authored_size", ((True, True, None), (True, False, 22), (False, True, None))
)
def test_prepared_button_metrics_keep_exact_field_evidence(tmp_path, icon, label, authored_size):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "design.json"))
    runtime.track_field_reads = True
    for radius in (None, 3):
        for fields in ({}, {"fontSize": 18, "iconSize": 26, "radius": 9, "columnGap": 12}):
            spec = deepcopy(runtime.spec)
            spec["parts"]["button"] = fields
            runtime.update_spec(spec)
            style = NativePartStyle(runtime, ("button",), authored_defaults=True).freeze()
            args = icon, label, 20, authored_size, radius, "center"
            runtime.begin_frame()
            expected = button_metrics.__wrapped__(style, *args)
            expected_reads = deepcopy(runtime.used_fields)
            for _ in range(2):
                runtime.begin_frame()
                assert button_metrics(style, *args) == expected
                assert runtime.used_fields == expected_reads
