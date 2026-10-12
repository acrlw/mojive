"""Native design input and document lifecycle at the interactive desktop size."""

import time
from pathlib import Path

import numpy as np
import pytest
from examples.ui_design.native.docking_study import Study, frames
from examples.ui_design.native.reference_menus import SAVED_SCENE
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.app.ui.window import create_window
from mojive.scene.assets import resolve
from mojive.scene.queries import node_world_pose
from mojive.session.model_edits import model_edit_scope
from mojive.types import MeshShape
from mojive.ui.window import WindowConfig

pytestmark = pytest.mark.gpu
OUTPUT = Path(__file__).resolve().parents[2] / "output/ui_design/native"


@pytest.mark.parametrize("runtime", ((2880, 1600),), indirect=True)
@pytest.mark.parametrize("language", ("en", "zh_CN"))
def test_saved_file_receipt_stays_compact_and_actions_stable(
    runtime, monkeypatch, tmp_path, language
):
    from mojive.ui.app import status as status_module

    study, app = runtime, runtime.preview
    app.localizer.set_language(language, persist=False)
    observed = {}
    button = imgui.button

    def observe(label, *args, **kwargs):
        result = button(label, *args, **kwargs)
        if "##status-" in label:
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            observed[label.split("##")[1]] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        return result

    monkeypatch.setattr(imgui, "button", observe)
    path = tmp_path / "viewport-ui-recording.mp4"
    path.touch()
    app.output.publish(
        f"{app.localizer.text('Saved video to')} {path}", duration=None, copy_text=str(path)
    )
    study.frame(4)
    receipt = app._status_notice_bounds
    assert receipt is not None
    x, y, right, bottom = receipt
    open_button, copy_button = observed["status-open-folder"], observed["status-copy-path"]
    assert bottom - y == pytest.approx(open_button[3] + 16 * study.s, abs=1)
    assert right - x <= 720 * study.s + 1
    assert right - x < app._viewport_rect[2] * 0.85
    for bx, by, bw, bh in (open_button, copy_button):
        assert x + 8 * study.s <= bx < bx + bw <= right - 8 * study.s + 1
        assert y + 8 * study.s <= by < by + bh <= bottom - 8 * study.s + 1
    revealed = []
    monkeypatch.setattr(status_module, "reveal_path", revealed.append)
    study.event(
        "add_mouse_pos_event",
        open_button[0] + open_button[2] / 2,
        open_button[1] + open_button[3] / 2,
    )
    study.event("add_mouse_button_event", 0, True)
    app.output.publish(
        f"{app.localizer.text('Saved video to')} {path} · {app.localizer.text('File copied to clipboard')}",
        duration=None,
        copy_text=str(path),
    )
    study.frame(3)
    assert app._status_notice_bounds == pytest.approx(receipt)
    assert observed["status-open-folder"] == pytest.approx(open_button)
    study.event("add_mouse_button_event", 0, False)
    assert revealed == [path]
    study.capture("saved-file-receipt-" + language)


@pytest.fixture
def runtime(backend_name, monkeypatch, tmp_path, request):
    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("MOJIVE_UI_SCALE", "2")
    width, height = getattr(request, "param", (1440, 900))
    window = create_window(
        WindowConfig(
            width=width,
            height=height,
            ui_scale=1,
            docking=True,
            ini_path="",
            show_on_start=False,
        ),
        backend_name,
    )
    study = Study(window, backend_name, output_directory=tmp_path)
    events, modal_buttons = [], {}
    process_inputs = window._input.process_inputs
    native_button = imgui.button

    def inputs():
        process_inputs()
        io = imgui.get_io()
        for method, args in events:
            getattr(io, method)(*args)
        events.clear()
        io.delta_time = 1 / 60

    def button(label, *args, **kwargs):
        pressed = native_button(label, *args, **kwargs)
        if label in ("Cancel", "Discard", "Save"):
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            modal_buttons[label] = (lo.x, lo.y, hi.x - lo.x, hi.y - lo.y)
        return pressed

    monkeypatch.setattr(window._input, "process_inputs", inputs)
    monkeypatch.setattr(imgui, "button", button)

    def frame(count=1):
        for _ in range(count):
            study.draw()

    def event(method, *args):
        events.append((method, args))
        frame()

    def click(point):
        event("add_mouse_pos_event", *point)
        event("add_mouse_button_event", 0, True)
        event("add_mouse_button_event", 0, False)

    def click_rect(rect):
        x, y, width, height = rect
        click((x + width / 2, y + height / 2))

    def click_hit(key):
        assert key in study.ui.hits, (key, study.ui.hits)
        click_rect(study.ui.hits[key])

    def keypress(key):
        x, y, width, height = study.rects["viewport_content"]
        point = None
        for ry in (0.6, 0.7, 0.4, 0.25):
            for rx in (0.5, 0.65, 0.35, 0.85, 0.15):
                candidate = (x + width * rx, y + height * ry)
                if not any(
                    a <= candidate[0] <= a + w and b <= candidate[1] <= b + h
                    for a, b, w, h in study.rects["overlays"]
                ):
                    point = candidate
                    break
            if point is not None:
                break
        assert point is not None, study.rects
        event("add_mouse_pos_event", *point)
        event("add_key_event", key, True)
        event("add_key_event", key, False)

    def capture(name):
        path = OUTPUT / backend_name / "runtime" / (name + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        return frames(window, study, path, count=8)

    def file_action(action):
        application = imgui.internal.find_window_by_name("Application")
        assert application is not None
        click((application.pos.x + 58 * study.s, application.pos.y + 18 * study.s))
        frame(2)
        observed = study.menus.observed.get("File")
        assert observed is not None, study.menus.observed
        label = {
            "save": "Save locally",
            "export": "Export scene JSON",
            "load": "Open saved scene",
        }[action]
        row = next(row for row in observed["rows"] if row["label"] == label)
        assert row["enabled"] and row["visible"], row
        click_rect(row["rect"])

    def modal_action(action):
        frame(3)
        assert action in modal_buttons, modal_buttons
        click_rect(modal_buttons[action])

    def workspace_action(name):
        key = "workspace-" + name
        if key in study.ui.hits:
            click_hit(key)
            return
        if "menu-Window" not in study.ui.hits:
            click_hit("menu-More")
            frame(2)
        click_hit("menu-Window")
        frame(2)
        workspace = next(
            row for row in study.menus.observed["Window"]["rows"] if row["label"] == "Workspace"
        )
        click_rect(workspace["rect"])
        frame(3)
        option = next(
            row for row in study.menus.observed["Window/Workspace"]["rows"] if row["label"] == name
        )
        assert option["enabled"] and option["visible"]
        click_rect(option["rect"])

    study.frame = frame
    study.event = event
    study.click_hit = click_hit
    study.workspace_action = workspace_action
    study.keypress = keypress
    study.capture = capture
    study.file_action = file_action
    study.modal_action = modal_action
    frame(8)
    try:
        assert study.s == pytest.approx(2)
        yield study
    finally:
        study.close()
        window.close()


def wait_loading(study):
    app = study.preview
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        study.frame()
        if (
            app._model_load_future is None
            and not app._model_load_queue
            and app._model_load_completion is None
        ):
            study.frame(4)
            return
        time.sleep(0.005)
    raise AssertionError("Queued native design document loading did not finish")


def test_color_popup_formats_clipboard_cancel_and_single_undo(runtime):
    study, session = runtime, runtime.preview.session
    entity = next(item for item in study.preview.entities if item["name"] == "free_body")
    session.submit(cmd.SelectNode(entity["node_id"]))
    study.frame(3)
    study.click_hit("tab-Material")
    study.frame(2)
    original = session.source.geom_rgba.copy()
    study.click_hit("material-color")
    study.frame(2)
    assert "color-editor-copy" in study.ui.hits
    study.click_hit("color-editor-format-2")
    imgui.set_clipboard_text("rgba(0.125, 0.25, 0.5, 0.75)")
    study.click_hit("color-editor-paste")
    study.frame(2)
    edited = session.source.geom_rgba.copy()
    assert not np.array_equal(edited, original)
    study.click_hit("color-editor-copy")
    assert imgui.get_clipboard_text() == "rgba1(0.125, 0.25, 0.5, 0.75)"
    imgui.set_clipboard_text("rgba(500, 0, 0, 1)")
    study.click_hit("color-editor-paste")
    study.frame(2)
    assert np.array_equal(session.source.geom_rgba, edited)
    assert study.menus.color_editor.error
    study.click_hit("done-color")
    study.frame(2)
    assert "color-editor-copy" in study.ui.hits
    imgui.set_clipboard_text("rgba1(0.125, 0.25, 0.5, 0.75)")
    study.click_hit("color-editor-paste")
    study.frame(2)
    assert not study.menus.color_editor.error
    study.capture("color-rgba-normalized")
    study.click_hit("done-color")
    study.frame(2)
    session.submit(cmd.Undo())
    study.frame(2)
    assert np.array_equal(session.source.geom_rgba, original)

    study.click_hit("material-color")
    study.click_hit("color-editor-format-0")
    study.click_hit("color-editor-hex")
    study.event("add_key_event", imgui.Key.mod_ctrl, True)
    study.event("add_key_event", imgui.Key.a, True)
    study.event("add_key_event", imgui.Key.a, False)
    study.event("add_key_event", imgui.Key.mod_ctrl, False)
    study.event("add_input_characters_utf8", "#803319CC")
    study.event("add_key_event", imgui.Key.enter, True)
    study.event("add_key_event", imgui.Key.enter, False)
    assert not np.array_equal(session.source.geom_rgba, original)
    study.capture("color-hex-alpha")
    study.event("add_key_event", imgui.Key.escape, True)
    study.event("add_key_event", imgui.Key.escape, False)
    study.frame(2)
    assert np.array_equal(session.source.geom_rgba, original)


def test_timeline_click_workspace_and_shortcut_round_trips(runtime):
    study = runtime
    assert not study.open["Timeline"]
    study.capture("timeline-initial")
    expanded_height = None
    for cycle in range(3):
        study.click_hit("expand-timeline" if cycle % 2 == 0 else "timeline-tab")
        snapshot = study.capture(f"timeline-expanded-{cycle}")
        assert study.open["Timeline"]
        height = snapshot["Timeline"]["rect"][3]
        if expanded_height is None:
            expanded_height = height
        else:
            assert height == pytest.approx(expanded_height, abs=2)
        study.click_hit("collapse-timeline")
        study.capture(f"timeline-collapsed-{cycle}")
        assert not study.open["Timeline"]
    study.workspace_action("Review")
    study.capture("workspace-review")
    assert study.workspace == "Review" and study.open["Timeline"]
    study.workspace_action("Edit")
    study.capture("workspace-edit")
    assert study.workspace == "Edit" and not study.open["Timeline"]
    for opened in (True, False):
        study.keypress(imgui.Key.t)
        study.capture("timeline-shortcut-" + ("open" if opened else "closed"))
        assert study.open["Timeline"] is opened


def test_file_inputs_keep_cancelled_edits_and_load_through_worker(runtime):
    study, app = runtime, runtime.preview
    app._add_scene_object(MeshShape.BOX, "runtime_box")
    study.frame(4)
    node = app.session.selected_node
    assert node is not None and app.entity["scalable"]
    position, rotation = node_world_pose(app.session, node)
    position, rotation = position.copy(), rotation.copy()
    study.file_action("save")
    study.capture("document-saved")
    saved = study.output_directory / SAVED_SCENE
    assert saved.is_file() and not app.session.dirty
    assert app.session.asset_path == saved.resolve()
    moved = position + np.array([0.4, 0.0, 0.0])
    assert app.session.submit(cmd.SetPose(node.node_id, moved, rotation)).ok
    study.capture("document-dirty")
    document_id = app.session.document_id
    study.file_action("load")
    study.capture("document-unsaved-prompt")
    assert app._pending_document_action == ("open_scene", saved)
    study.modal_action("Cancel")
    study.capture("document-cancelled")
    assert app._pending_document_action is None
    assert app.session.document_id == document_id and app.session.dirty
    np.testing.assert_allclose(node_world_pose(app.session, node)[0], moved)
    study.file_action("load")
    study.modal_action("Discard")
    wait_loading(study)
    study.capture("document-loaded")
    assert app.session.document_id != document_id and not app.session.dirty
    node = next(node for node in app.session.nodes if node.name == "runtime_box")
    np.testing.assert_allclose(node_world_pose(app.session, node)[0], position)
    study.click_hit("expand-timeline")
    study.capture("loaded-timeline-expanded")
    assert study.open["Timeline"]
    study.click_hit("collapse-timeline")
    study.capture("loaded-timeline-collapsed")
    app.session.submit(cmd.SelectNode(node.node_id))
    with model_edit_scope(app.session, app._intercept_model_edit):
        assert app.session.submit(cmd.SetScale(node.node_id, np.array([2.0, 1.0, 1.0]))).ok
    study.capture("document-pending-dimensions")
    assert app.model_edits.active
    study.click_hit("expand-timeline")
    study.capture("document-pending-timeline")
    study.click_hit("collapse-timeline")
    study.capture("document-pending-timeline-collapsed")
    stored = saved.read_text()
    study.file_action("save")
    study.capture("document-save-blocked")
    assert app.model_edits.active and saved.read_text() == stored
    assert app.session.last_message == "Apply or discard pending model edits first"
    study.click_hit("discard-model-edits")
    study.capture("document-dimensions-discarded")
    assert not app.model_edits.active


def test_physics_tab_uses_node_identity_in_composed_model(runtime, monkeypatch):
    study, app = runtime, runtime.preview
    assert app.session.submit(
        cmd.AddSceneModel(resolve("joint_types"), np.array([0.0, 3.0, 0.0]), np.eye(3))
    ).ok
    app.selected = "hinge_body"
    study.activate("Inspector")
    study.frame(4)
    node = app.session.selected_node
    assert node.node_id != node.body_index
    reads = []
    body_properties = app.session.body_properties

    def properties(node_id):
        reads.append(node_id)
        return body_properties(node_id)

    monkeypatch.setattr(app.session, "body_properties", properties)
    study.click_hit("tab-Physics")
    study.capture("physics-composed-model")
    assert study.inspector_tab == "Physics"
    assert reads and set(reads) == {node.node_id}
