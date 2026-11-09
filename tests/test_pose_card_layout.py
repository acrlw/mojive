"""Production pose fields preserve complete XYZ rows and stable numeric geometry."""

from copy import deepcopy

import pytest

from mojive.design.native.presentation import DesignPresentation
from mojive.design.runtime import DesignRuntime
from mojive.design.store import SpecificationStore
from mojive.ui.imgui_draw import ImguiDraw2D
from tests.test_ui_design_pose import editor as editor
from tests.test_ui_design_pose import session as session

pytestmark = pytest.mark.physics


@pytest.mark.parametrize(
    "designed,width,columns",
    (
        (False, 160, 1),
        (False, 360, 3),
        (False, 640, 3),
        (True, 160, 1),
        (True, 320, 3),
        (True, 640, 3),
    ),
)
@pytest.mark.parametrize("kind", ("ball", "free"))
def test_expanded_pose_uses_a_complete_xyz_row_when_its_fixed_precision_fits(
    editor, tmp_path, designed, width, columns, kind
):
    if designed:
        editor.ui.design = DesignPresentation(
            editor.ui, DesignRuntime(SpecificationStore(tmp_path / "spec.json"))
        )
    editor.width, editor.kind = width, kind
    editor.frame(3)
    names = ("position", "rotation") if kind == "free" else ("rotation",)
    for name in names:
        fields = [
            editor.ui.hits[f"##pose-joint-{editor.joint.joint_id}-{name}{axis}-frame"]
            for axis in "XYZ"
        ]
        assert len({round(x, 3) for x, _y, _w, _h in fields}) == columns
        assert len({round(y, 3) for _x, y, _w, _h in fields}) == 3 // columns
        x, y, width, height = editor.ui.hits[f"pose-fields-{editor.joint.joint_id}"]
        for fx, fy, fw, fh in fields:
            assert x <= fx < fx + fw <= x + width + 1
            assert y <= fy < fy + fh <= y + height + 1
            if columns == 3:
                assert fw >= editor.ui.numeric_minimum(format="%.3f") - 1


@pytest.mark.parametrize("columns,rows", ((1, 3), (2, 3), (3, 1)))
def test_expanded_pose_respects_explicit_xyz_columns(editor, tmp_path, columns, rows):
    runtime = DesignRuntime(SpecificationStore(tmp_path / "spec.json"))
    editor.ui.design = DesignPresentation(editor.ui, runtime)
    spec = deepcopy(runtime.spec)
    spec["parts"]["vector_group"] = {"columns": columns}
    runtime.update_spec(spec)
    editor.width, editor.kind = 640, "free"

    editor.frame(2)
    fields = [
        editor.ui.hits[f"##pose-joint-{editor.joint.joint_id}-position{axis}-frame"]
        for axis in "XYZ"
    ]
    assert len({round(field[1], 3) for field in fields}) == rows


def test_expanded_xyz_keeps_signed_decimal_domain_and_geometry_stable(
    editor, tmp_path, monkeypatch
):
    editor.ui.design = DesignPresentation(
        editor.ui, DesignRuntime(SpecificationStore(tmp_path / "spec.json"))
    )
    editor.width, editor.kind = 320, "free"
    readings = []
    draw_text = ImguiDraw2D.text

    def text(draw, position, color, caption, **kwargs):
        readings.append(caption)
        return draw_text(draw, position, color, caption, **kwargs)

    monkeypatch.setattr(ImguiDraw2D, "text", text)
    editor.frame(2)
    start = editor.joint.qpos_adr
    values = (-360.0, 0, 0.123456789)
    editor.session.adapter.primary.data.qpos[start : start + 3] = values
    readings.clear()
    editor.frame(1)
    assert all(shown in readings for shown in ("-360.000", "0.000", "0.123"))
    fields = [
        editor.ui.hits[f"##pose-joint-{editor.joint.joint_id}-position{axis}-frame"]
        for axis in "XYZ"
    ]
    assert len({round(field[1], 3) for field in fields}) == 1
    assert editor.session.adapter.primary.data.qpos[start : start + 3] == pytest.approx(
        values, abs=1e-12
    )
    editor.session.adapter.primary.data.qpos[start : start + 3] = (1e9, -1e-9, 123.456789)
    editor.frame(1)
    assert [
        editor.ui.hits[f"##pose-joint-{editor.joint.joint_id}-position{axis}-frame"]
        for axis in "XYZ"
    ] == fields
