"""CPU contracts for the parallel MuJoCo model corpus runner."""

from pathlib import Path

from mojive.tools.mujoco_model_suite import (
    ModelAuditResult,
    ModelAuditStatus,
    build_report,
    classify_load_failure,
    discover_models,
    inspect_fragment_files,
    inspect_plugin_references,
    inspect_unsupported_assets,
)


def test_discovers_sorted_unique_model_documents(tmp_path: Path):
    """Discovery accepts MJCF/URDF roots and rejects unrelated XML metadata."""

    first = tmp_path / "a.xml"
    second = tmp_path / "nested" / "b.urdf"
    second.parent.mkdir()
    first.write_text("<mujoco/>", encoding="utf-8")
    second.write_text("<robot name='b'/>", encoding="utf-8")
    third = tmp_path / "c.MJCF"
    third.write_text("<mujoco/>", encoding="utf-8")
    (tmp_path / "package.xml").write_text("<package/>", encoding="utf-8")

    assert discover_models([tmp_path, first]) == [
        first.resolve(),
        third.resolve(),
        second.resolve(),
    ]


def test_included_model_files_are_identified_as_fragments(tmp_path: Path):
    fragment = tmp_path / "keyframes.xml"
    fragment.write_text("<mujoco><keyframe><key qpos='1'/></keyframe></mujoco>")
    external_targets = tmp_path / "actuators.xml"
    external_targets.write_text(
        "<mujoco><worldbody><site name='anchor'/></worldbody>"
        "<actuator><motor joint='external'/></actuator></mujoco>"
    )
    scene = tmp_path / "scene.xml"
    scene.write_text('<mujoco><include file="keyframes.xml"/></mujoco>')
    models = discover_models([tmp_path])

    assert inspect_fragment_files(models) == {fragment.resolve(), external_targets.resolve()}
    assert classify_load_failure((), "invalid qpos", model_fragment=True) is (
        ModelAuditStatus.SKIPPED_FRAGMENT
    )


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


def test_collada_meshes_are_reported_as_unsupported_assets(tmp_path: Path):
    model = tmp_path / "collada.urdf"
    model.write_text(
        '<robot name="dae"><link name="x"><visual><geometry>'
        '<mesh filename="meshes/x.dae"/></geometry></visual></link></robot>'
    )

    assert inspect_unsupported_assets(model) == (".dae",)


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
    assert report["counts"]["workspace_failed"] == 0
    assert report["counts"]["composition_failed"] == 0
    assert report["counts"]["skipped_dependency"] == 0
    assert report["counts"]["skipped_fragment"] == 0
    assert report["counts"]["skipped_unsupported_asset"] == 0
