"""Joint fields share the heading's leading content column at every reflow size."""

from copy import deepcopy

import pytest

from mojive.design.groups import group_scope
from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from tests.test_ui_design_pose import editor as editor
from tests.test_ui_design_pose import session as session

pytestmark = pytest.mark.physics

_CASES = (
    ("hinge", True),
    ("slide", True),
    ("ball", False),
    ("ball", True),
    ("free", False),
    ("free", True),
)


def _draw(editor, tmp_path, kind, expanded, *, parts=None):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "alignment.json"))
    runtime.observe_components = True
    if parts:
        specification = deepcopy(runtime.spec)
        specification["parts"].update(parts)
        runtime.update_spec(specification)
    editor.ui.design = DesignPresentation(editor.ui, runtime)
    editor.kind = kind
    editor.show_names = editor.collapsible_poses = True
    with group_scope("Pose"):
        editor.frame(2)
        for owner in editor.panel._expanded:
            editor.panel._expanded[owner] = expanded
        runtime.observed.clear()
        editor.ui.design.primitives.observed.clear()
        editor.frame(2)
    records = {
        record["key"]: record
        for record in (*runtime.observed, *editor.ui.design.primitives.observed)
    }
    joint_id = editor.joint.joint_id
    return records[f"joint-card-{joint_id}"], records[f"pose-joint-{joint_id}"]


@pytest.mark.parametrize("editor", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("width", (160, 640))
@pytest.mark.parametrize("kind,expanded", _CASES)
def test_painted_heading_and_all_body_paths_share_the_icon_content_column(
    editor, tmp_path, monkeypatch, width, kind, expanded
):
    editor.width = width
    painted = []
    vector, rail, text = editor.ui.vector, editor.ui.rail, editor.ui.component_text

    def record_vector(key, x, y, width, *args, **kwargs):
        painted.append((x, width))
        return vector(key, x, y, width, *args, **kwargs)

    def record_rail(key, x, y, width, *args, **kwargs):
        painted.append((x, width))
        return rail(key, x, y, width, *args, **kwargs)

    def record_text(part, rect, label, *args, **kwargs):
        if label.startswith(("Position (", "Rotation (")):
            painted.append((rect[0], rect[2]))
        return text(part, rect, label, *args, **kwargs)

    monkeypatch.setattr(editor.ui, "vector", record_vector)
    monkeypatch.setattr(editor.ui, "rail", record_rail)
    monkeypatch.setattr(editor.ui, "component_text", record_text)
    group, heading = _draw(editor, tmp_path, kind, expanded)
    x, _y, width, _height = group["bodyColumn"]
    assert group["alignmentAxis"] == "leading-content"
    assert x == pytest.approx(heading["content"][0])
    assert x == pytest.approx(heading["iconPosition"][0])
    assert x < heading["textPosition"][0]
    assert x - group["content"][0] == pytest.approx(8 * editor.ui.s)
    assert painted
    for painted_x, painted_width in painted:
        assert painted_x == pytest.approx(x)
        assert 0 < painted_width <= width + 0.001
    joint_id = editor.joint.joint_id
    for key, (field_x, _field_y, field_width, _field_height) in editor.ui.hits.items():
        if key.startswith((f"##pose-joint-{joint_id}", f"pose-summary-{joint_id}-")):
            assert field_x >= x - 0.001, key
            assert field_x + field_width <= x + width + 0.001, key


@pytest.mark.parametrize("editor", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("kind,expanded", _CASES)
def test_body_column_tracks_authored_heading_edges_without_repeating_joint_padding(
    editor, tmp_path, kind, expanded
):
    editor.width = 640
    group, heading = _draw(
        editor,
        tmp_path,
        kind,
        expanded,
        parts={
            "button": {"paddingLeft": 13, "paddingRight": 17, "marginLeft": 3, "marginRight": 5},
            "pose_joint": {"paddingLeft": 11, "paddingRight": 15},
        },
    )
    x, _y, width, _height = group["bodyColumn"]
    gx, _gy, gw, _gh = group["content"]
    assert x == pytest.approx(heading["content"][0])
    assert x - gx == pytest.approx(16 * editor.ui.s)
    assert gx + gw - x - width == pytest.approx(22 * editor.ui.s)


@pytest.mark.parametrize("editor", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("kind", ("ball", "free"))
def test_unheaded_joint_fields_keep_the_entire_parent_content_width(editor, tmp_path, kind):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "unheaded.json"))
    runtime.observe_components = True
    editor.ui.design = DesignPresentation(editor.ui, runtime)
    editor.kind = kind
    editor.show_names = False
    with group_scope("Pose"):
        editor.frame(2)
    group = next(
        record for record in reversed(runtime.observed) if record["key"].startswith("joint-card-")
    )
    assert group["alignmentAxis"] == "unheaded"
    assert group["headingContentColumn"] is None
    assert group["bodyColumn"][0] == pytest.approx(group["content"][0])
    assert group["bodyColumn"][2] == pytest.approx(group["content"][2])


@pytest.mark.parametrize("editor", (1, 1.2, 1.25, 2.25), indirect=True)
@pytest.mark.parametrize("kind,expanded", _CASES)
def test_authored_joint_width_measures_the_same_body_budget_used_for_paint(
    editor, tmp_path, kind, expanded
):
    editor.width = 640
    group, heading = _draw(
        editor,
        tmp_path,
        kind,
        expanded,
        parts={"pose_joint": {"widthMode": "fixed", "width": 180}},
    )
    x, y, width, height = group["bodyColumn"]
    assert group["hit"][2] == pytest.approx(180 * editor.ui.s)
    assert x == pytest.approx(heading["content"][0])
    assert x + width == pytest.approx(group["content"][0] + group["content"][2] - 8 * editor.ui.s)
    joint_id = editor.joint.joint_id
    for key, (fx, fy, fw, fh) in editor.ui.hits.items():
        if key.startswith((f"##pose-joint-{joint_id}", f"pose-summary-{joint_id}-")):
            assert x - 0.001 <= fx <= fx + fw <= x + width + 0.001, key
            if kind in ("ball", "free"):
                assert y - 0.001 <= fy <= fy + fh <= y + height + 0.001, key
