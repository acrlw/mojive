"""CPU contracts for the parallel MuJoCo model corpus runner."""

from pathlib import Path

from forge_viewer.tools.mujoco_model_suite import (
    ModelAuditResult,
    ModelAuditStatus,
    build_report,
    classify_load_failure,
    discover_models,
    inspect_plugin_references,
)


def test_discovers_sorted_unique_xml_files(tmp_path: Path):
    """Model discovery accepts mixed file and directory roots without duplicates."""

    first = tmp_path / "a.xml"
    second = tmp_path / "nested" / "b.xml"
    second.parent.mkdir()
    first.write_text("<mujoco/>", encoding="utf-8")
    second.write_text("<mujoco/>", encoding="utf-8")

    assert discover_models([tmp_path, first]) == [first.resolve(), second.resolve()]


def test_plugin_references_do_not_hide_model_errors(tmp_path: Path):
    """A declared plugin only permits a skip when the runtime reports it unavailable."""

    model = tmp_path / "plugin.xml"
    model.write_text(
        '<mujoco><extension><plugin plugin="example.missing"/></extension></mujoco>',
        encoding="utf-8",
    )
    plugins = inspect_plugin_references(model)

    assert plugins == ("example.missing",)
    assert classify_load_failure(plugins, "unrecognized plugin example.missing") is (
        ModelAuditStatus.SKIPPED_DEPENDENCY
    )
    assert classify_load_failure(plugins, "invalid geom size") is ModelAuditStatus.LOAD_FAILED


def test_report_counts_every_outcome(tmp_path: Path):
    """JSON report summaries retain every status even when its count is zero."""

    report = build_report(
        [tmp_path],
        [
            ModelAuditResult(path="good.xml", status=ModelAuditStatus.PASSED),
            ModelAuditResult(path="bad.xml", status=ModelAuditStatus.RENDER_FAILED),
        ],
    )

    assert report["counts"]["passed"] == 1
    assert report["counts"]["render_failed"] == 1
    assert report["counts"]["skipped_dependency"] == 0
