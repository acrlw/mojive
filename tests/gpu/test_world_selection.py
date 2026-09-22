"""Production world controls commit bounded subsets only on explicit Apply."""

from pathlib import Path

import pytest
from imgui_bundle import imgui
from PIL import Image

from mojive.adapters.joint_replay import JointReplayAdapter
from mojive.app.ui.window import create_window
from mojive.render.backend import NullBackend
from mojive.session import Session
from mojive.tools.g1_replay import generate
from mojive.ui.localization import Localizer
from mojive.ui.panels import PanelContext
from mojive.ui.panels.hierarchy import HierarchyPanel
from mojive.ui.window import WindowConfig

pytestmark = [pytest.mark.gpu, pytest.mark.physics]


@pytest.mark.parametrize("language,scale", [("en", 1.0), ("zh_CN", 1.5)])
def test_world_controls_apply_counts_ids_and_reject_invalid_drafts(
    tmp_path, monkeypatch, backend_name, language, scale
):
    monkeypatch.setattr("mojive.tools.g1_worlds.JOINT_NAMES", ())
    model = tmp_path / "robot.xml"
    model.write_text(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint name="floating_base_joint"/>'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    archive = tmp_path / "archive"
    generate(model, archive, 4, 6, 30)
    session = Session(JointReplayAdapter(archive, worlds=2, max_worlds=4, realtime=False))
    panel = HierarchyPanel()
    localizer = Localizer(path=tmp_path / "settings.json")
    localizer.set_language(language, persist=False)
    ctx = PanelContext(session, NullBackend(), style_scale=scale, translate=localizer.text)
    rectangles = {}

    def tracked(native):
        def draw(label, *args, **kwargs):
            result = native(label, *args, **kwargs)
            lo, hi = imgui.get_item_rect_min(), imgui.get_item_rect_max()
            rectangles[label] = (lo.x, lo.y, hi.x, hi.y)
            return result

        return draw

    for name in ("button", "checkbox", "input_int", "input_text_with_hint"):
        monkeypatch.setattr(imgui, name, tracked(getattr(imgui, name)))
    config = WindowConfig(
        width=round(340 * scale),
        height=round(680 * scale),
        ui_scale=scale,
        docking=False,
        ini_path="",
        vsync=False,
        show_on_start=False,
    )
    with create_window(config, backend_name) as window:

        def frame():
            window.begin_frame()
            imgui.set_next_window_pos((0, 0))
            imgui.set_next_window_size(imgui.get_io().display_size)
            imgui.begin(localizer.text("Hierarchy"))
            panel.draw(ctx)
            imgui.end()
            return window.end_frame(readback=True)[::-1].copy()

        def click(label):
            lo, top, hi, bottom = rectangles[label]
            io = imgui.get_io()
            io.add_mouse_pos_event(lo + (hi - lo) * 0.3, (top + bottom) / 2)
            frame()
            io.add_mouse_button_event(0, True)
            frame()
            io.add_mouse_button_event(0, False)
            frame()

        def edit(label, value):
            click(label)
            io = imgui.get_io()
            modifier = imgui.Key.mod_super if io.config_mac_osx_behaviors else imgui.Key.mod_ctrl
            io.add_key_event(modifier, True)
            io.add_key_event(imgui.Key.a, True)
            frame()
            io.add_key_event(imgui.Key.a, False)
            io.add_key_event(modifier, False)
            io.add_input_characters_utf8(value)
            frame()

        try:
            for _ in range(3):
                frame()
            generation = session.structure_generation
            edit("##world_count", "1")
            assert session.structure_generation == generation
            click(localizer.text("Apply worlds"))
            assert session.world_selection.world_ids == (0,)
            click(localizer.text("Choose world IDs"))
            edit("##world_ids", "3, 1")
            assert session.world_selection.world_ids == (0,)
            click(localizer.text("Apply worlds"))
            assert session.world_selection.world_ids == (3, 1)
            for ids in ("3, 3", "9", "0, 1, 2, 3, 4"):
                edit("##world_ids", ids)
                click(localizer.text("Apply worlds"))
                assert session.world_selection.world_ids == (3, 1)
            # An RPC/native change refreshes the draft through the same Session state.
            from mojive import commands as cmd

            assert session.submit(cmd.SetWorldSelection((2, 0))).ok
            frame()
            click(localizer.text("Apply worlds"))
            assert session.world_selection.world_ids == (2, 0)
            output = Path("output/world-selection")
            output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frame()).save(output / f"controls-{backend_name}-{language}.png")
        finally:
            session.release()


@pytest.mark.parametrize("language", ["en", "zh_CN"])
def test_local_rollout_playback_buttons_scrubbing_shortcuts_and_manual_sync(
    tmp_path, monkeypatch, backend_name, language
):
    import time

    import numpy as np

    from mojive import commands as cmd
    from mojive.adapters.rollout_replay import RolloutReplayAdapter
    from mojive.app.composition import build_from_adapter
    from mojive.remote.rollout import RolloutServer, RolloutStore
    from mojive.tools.ui_runtime import _activate_panel, _click, _item_center, _item_rect

    monkeypatch.setenv("MOJIVE_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setattr("mojive.tools.g1_worlds.JOINT_NAMES", ())
    model = tmp_path / "robot.xml"
    model.write_text(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint name="floating_base_joint"/>'
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    archive = tmp_path / "archive"
    generate(model, archive, 4, 60, 30)
    store = RolloutStore.from_archive(archive, max_worlds=4)
    now = [100.0]
    monkeypatch.setattr("mojive.adapters.joint_replay.time.perf_counter", lambda: now[0])
    with RolloutServer(store, port=0) as server:
        adapter = RolloutReplayAdapter(
            f"http://127.0.0.1:{server.address[1]}",
            world_ids=(3, 0),
            max_worlds=4,
            window_frames=32,
        )
        with build_from_adapter(
            adapter, width=960, height=720, vsync=False, show_window=False, renderer=backend_name
        ) as viewer:
            viewer.app.set_language(language)
            t = viewer.app.localizer.text

            def click_control(label):
                _click(viewer, _item_center(viewer, "button", label, reveal=True))

            for _ in range(6):
                viewer.sync()
            _activate_panel(viewer, "Keyframes")
            assert adapter.replay_info().paused
            requests = store.window_requests
            click_control(t("Play") + "##replay_play")
            assert not adapter.replay_info().paused
            now[0] += 4.1 / 30
            viewer.sync()
            assert adapter.replay_info().frame_index == 4
            click_control(t("Pause") + "##replay_play")
            assert adapter.replay_info().paused
            click_control(t("Next frame") + "##replay_next")
            assert adapter.replay_info().frame_index == 5
            click_control(t("Previous frame") + "##replay_previous")
            assert adapter.replay_info().frame_index == 4
            # A held scrub is a draft: apply once on release, rather than doing FK every UI frame.
            lo, hi = _item_rect(viewer, "slider_int", "##replay_frame", reveal=True)
            io = imgui.get_io()
            io.add_mouse_pos_event(lo[0] + (hi[0] - lo[0]) * 0.65, (lo[1] + hi[1]) / 2)
            viewer.sync()
            io.add_mouse_button_event(0, True)
            viewer.sync()
            for _ in range(3):
                viewer.sync()
                assert adapter.replay_info().frame_index == 4
            io.add_mouse_button_event(0, False)
            viewer.sync()
            assert 15 <= adapter.replay_info().frame_index <= 25
            assert adapter.replay_info().paused
            click_control(t("Restart") + "##replay_restart")
            assert adapter.replay_info().frame_index == 0
            x, y, width, height = viewer.app._viewport_rect
            _click(viewer, (x + width * 0.7, y + height * 0.7))
            io.add_key_event(imgui.Key.space, True)
            viewer.sync()
            io.add_key_event(imgui.Key.space, False)
            viewer.sync()
            assert not adapter.replay_info().paused
            assert viewer.session.submit(cmd.SetReplayPlayback(paused=True)).ok
            assert store.window_requests == requests
            # Publishing does not replace the clip until the user explicitly requests it.
            values = np.array(np.load(archive / "qpos.npy")[-32:])
            values[..., 0] += 1
            revision = store.publish(values, start_step=500)
            for _ in range(5):
                viewer.sync()
            assert adapter.rollout_sync_info().revision != revision
            _activate_panel(viewer, "Keyframes")
            source = viewer.session.source
            output = Path("output/rollout-preview")
            output.mkdir(parents=True, exist_ok=True)
            viewer.capture(output / f"before-sync-{backend_name}-{language}.png", surface="window")
            click_control(t("Sync latest rollout"))
            until = time.monotonic() + 3
            while adapter.rollout_sync_info().pending and time.monotonic() < until:
                viewer.sync()
                time.sleep(0.005)
            assert adapter.rollout_sync_info().revision == revision, adapter.rollout_sync_info()
            assert not adapter.rollout_sync_info().error
            assert viewer.session.source is source
            assert adapter.replay_info().paused
            assert store.window_requests == requests + 1
            output = Path("output/rollout-preview")
            output.mkdir(parents=True, exist_ok=True)
            viewer.capture(output / f"controls-{backend_name}-{language}.png", surface="window")
            # A failure must be visible at the world-selection entry point as well.
            store.max_window_bytes = 1
            _activate_panel(viewer, "Hierarchy")
            _click(viewer, _item_center(viewer, "button", t("Sync worlds")))
            until = time.monotonic() + 3
            while adapter.rollout_sync_info().pending and time.monotonic() < until:
                viewer.sync()
                time.sleep(0.005)
            failure = adapter.rollout_sync_info().error
            assert "byte budget" in failure
            assert adapter.rollout_sync_info().revision == revision
            drawn = []
            text_wrapped = imgui.text_wrapped

            def track_text(text):
                drawn.append(text)
                text_wrapped(text)

            monkeypatch.setattr(imgui, "text_wrapped", track_text)
            viewer.sync()
            assert failure in drawn
