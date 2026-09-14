# Passive viewing

`launch_passive(model, data)` displays an existing MuJoCo simulation while the calling code
owns all physics stepping. A spawned process owns the window and renderer. Its Python GIL,
event loop, and display rate are independent of the policy loop. `sync()` exchanges the latest
integration state and UI input at most `max_fps` times per second; it does not render, wait for
presentation, or call `mj_step`. Pending old frames do not accumulate.

```python
import mujoco
import mojive


def main():
    model = mujoco.MjModel.from_xml_path("assets/joint_types.xml")
    data = mujoco.MjData(model)
    with mojive.launch_passive(model, data, max_fps=60) as viewer:
        step = 0
        while viewer.is_running():
            # Replace with data.ctrl[:] = policy(observation).
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            step += 1
            viewer.sync(step=step)


if __name__ == "__main__":
    main()
```

Run from an importable script with the guarded entry point shown above. Python's spawn
context also keeps GLFW on the display process's main thread on macOS. Interactive notebooks
should invoke a script, or use synchronous embedding below. The caller may run as fast as
possible or choose its own pacing. `max_fps` limits display work; it is not a physics frequency.
The default display limit is 60 FPS. Set a higher value for a high-refresh-rate display.
Python scheduling does not provide a hard real-time deadline guarantee.

`viewer.model` and `viewer.data` are the exact objects supplied by the caller. The display owns
private copies, and the model's structure and parameters are fixed at launch. State changes,
including resets, mocap targets, activation, and applied forces, appear on the next published
snapshot. Close and relaunch after replacing or editing a model. `step` is an optional
caller-supplied episode counter (default zero), while simulation time comes from `data.time`.

The status bar labels **Physics … Hz** and **Render … FPS** separately. Physics throughput
uses actual counter increments over wall time, never the nominal `1 / timestep`. Supply
`sync(step=step)` for a passive viewer to measure it; without an authoritative counter the
physics rate is shown as unknown. Pauses and episode resets restart or settle the measurement
without negative rates. Render FPS includes slow frames and remains meaningful below 10 FPS.
Static scenes show only render throughput. Rates are averaged to keep the readout stable.
The left-hand status starts with **Passive** (**被动模式** in Chinese), separate from the
rendering backend. Call `set_status("", paused=...)` to show the localized **Running** or
**Paused** state beside it. Until the caller reports a state, only the mode is shown;
Mojive does not infer whether policy inference is running from publication rate.

Actuator slider changes apply once at a `sync()` boundary, so a stale display value cannot
keep overwriting policy output. Mouse perturbation writes the dragged body's `xfrc_applied`,
then clears it on release or close; other bodies' applied forces are preserved. Use
`with viewer.lock():` to serialize multiple caller threads touching the same model/data.
The window cannot pause, reset, or step caller-owned physics. Pose and topology editing are
disabled for the passive display. Camera navigation and visual inspection remain available.
`AdapterCaps.clock_control` distinguishes caller-only clock ownership from an externally
clocked adapter that can forward explicit pause/step commands, such as a remote publisher.
For policies that exclusively compute `data.ctrl`, pass `control_writeback=False` to
`launch_passive`. Actuator values stay visible but both UI and RPC writes are rejected at the
capability boundary. Change high-level velocity commands through caller-owned actions instead;
an actuator slider cannot override a policy that replaces controls on the next physics step.

## Live video without taking over physics

The playback toolbar's video button, its recording settings, the View → Record menu, and
Ctrl/Cmd+Shift+R work with passive viewing. Video capture does not require simulation snapshots.
Take recording and replay remain unavailable: live caller updates must not compete with a
second owner restoring recorded physics states. Camera and viewport capture remain independent.

```python
viewer.configure_recording(mojive.RecordingConfig(fps=30, crf=20, countdown=0))
viewer.start_recording("output/evaluation.mp4", surface="window")
# Continue the ordinary policy / mj_step / sync loop.
viewer.pause_recording()    # Stops writing frames, not policy inference.
viewer.resume_recording()
path = viewer.stop_recording()  # Returns after the encoder has finalized the file.
```

`viewer.recording` reports phase, frame count, duration and any asynchronous encoding error.
`stop_recording()` raises on encoding failure; a new recording clears the previous error.
Configuration changes apply to the next recording. The “Run simulation when recording starts”
setting is disabled for caller-owned clocks and has no effect even if previously saved as true.
Pause/resume controls only change video writing. `close()` also finalizes an active video and
waits for its completion; shutdown timeout/failure is reported instead of claiming a saved file.

Video samples the displayed scene using wall time. The latest-state mailbox may skip physics
steps and the recorder may repeat display frames; this is not an exact, deterministic policy
trace. Record observations, actions and integration states in the caller for step-exact evaluation,
then replay that separate trace for export. Encoding runs in the display process but still shares
machine resources; do not equate this separation with a hard real-time guarantee.

## Caller-owned controls and feedback

Use declarative actions instead of a desktop-global keyboard listener. Input is accepted only
while the viewport is focused, without a modal, active widget or modifier chord. Claimed keys
are not also consumed by Mojive's tools. Application callbacks are never run in the display process.

```python
viewer.configure_actions((
    mojive.PassiveAction("pause_policy", "space", "Pause / resume policy", "toggle"),
    mojive.PassiveAction("forward", "8", "Increase forward speed"),
))
paused = False
viewer.set_status("", paused=False)
while viewer.is_running():
    for event in viewer.poll_events():  # Nonblocking; process at a policy/step boundary.
        try:
            if event.action == "pause_policy":
                paused = not paused
            elif event.action == "forward":
                velocity[0] += 0.1
            viewer.set_status("", paused=paused)
        except Exception as exc:
            viewer.acknowledge_event(event, error=str(exc))
        else:
            viewer.acknowledge_event(event)
    if not paused:
        data.ctrl[:] = policy(observation)
        mujoco.mj_step(model, data)
    viewer.sync()
    # Apply your normal pacing here, including while paused.
```

The optional `control` binds an existing toolbar button (`toggle`, `reset`, `step`, `previous`).
Only explicitly bound buttons become available; the adapter's clock commands remain unsupported.
The caller must reset policy history and recurrent state along with physics when offering reset.
`set_status(text, paused=...)` updates presentation only. It does not execute or acknowledge an action.
Empty text uses the standard localized state; optional custom text and action labels are
supplied by the application in its preferred language.

At most 64 actions can be configured. Each action has at most one outstanding request until
acknowledgement, so an unattended UI cannot build an unbounded control backlog. Requests are
not retried automatically. Replace bindings only after acknowledging pending requests. Polling is
nonblocking; configuration, status updates, acknowledgement and capture/recording methods wait
for a display response. Call those at explicit control boundaries, not on every physics step.

Keys use `InputContext` names: letters (`a`), digits (`8`), `space`, `enter`, `escape`,
`up_arrow`, `down_arrow`, `left_arrow`, and `right_arrow`, for example. Display labels such as
`Up` are independent of binding identifiers. Modifier-only actions, including `left_shift`
and `mod_ctrl`, are rejected because passive actions do not accept modifier chords.

Action hints default to individual keys in the status bar. Group related keys and move them
to the existing viewport hint capsule without changing input behavior:

```python
from mojive.ui import ToolHint

viewer.configure_actions(
    (
        mojive.PassiveAction("increase", "up_arrow", "Increase speed"),
        mojive.PassiveAction("decrease", "down_arrow", "Decrease speed"),
    ),
    hints=(ToolHint("keys", label="Speed", keys=(("Up", "+"), ("Down", "−"))),),
    hint_surface="scene",
)
```

This displays `Speed [Up] + / [Down] −` with real keycaps. Scene hints wrap at whole-group
boundaries within 30% of the viewport height. They scale together down to 75% of the requested
size, then show an ellipsis for overflow; hover the capsule to read the complete hints.
`hints` replaces the automatic hints; `hints=()` hides
them while retaining all bindings. Labels remain caller-provided, and viewport UI visibility
controls whether the scene capsule appears.

Update presentation independently of actions, even while an action awaits acknowledgement:

```python
viewer.configure_tool_hints(
    (ToolHint("keys", label="Speed", keys=(("Up", "+"), ("Down", "−"))),),
    surface="scene",
)
viewer.configure_tool_hints(())  # Hide application hints; inputs still work.
```

Ordinary `Viewer` provides the same method. Moving this configured group to another surface
removes its previous placement; unrelated individually registered hints remain. Calling
`configure_actions` again still replaces bindings and resets their hints according to its
arguments. Hint updates are synchronous display requests, so update changed text at control
boundaries rather than on every physics step.

For alternative drawing, see [custom hint presentation](../how-to/ui-drawing.md#custom-hint-presentation).
PassiveViewer sends declarative data, not arbitrary Python draw callbacks, to its display worker.

`is_running()` becomes false when the window closes or its worker exits. `close()` and the
context manager stop and reap the worker. Startup and explicit capture/control failures raise
exceptions. `sync()` after close is harmless, allowing the physics loop to continue without
a viewer if desired.

## Capture without files

`viewer.capture_array()` publishes the current state and returns an owned, top-left RGB
`uint8` array. This explicit readback waits for one rendered frame. Use `surface="window"`
or `surface="viewport"` to include the corresponding UI. `capture(output)` writes only when
an output path is explicitly requested.

For repeated cross-process capture, reuse a `SharedImage` with `viewer.capture_into(buffer,
surface="window")`. A window buffer is RGB `uint8` with the window's current framebuffer shape;
a scene buffer matches the viewport render target. The returned metadata contains the captured
step and time. Read `buffer.array` after completion and consume it before the next capture into
that buffer. This avoids sending image bytes through the display process's control pipe.
See [shared image transport](../how-to/rpc-control.md#in-memory-capture) for ownership and timeout
semantics. Normal synchronous viewers accept a NumPy destination with `viewer.capture_into(out)`.

For camera RGB, metric depth, and segmentation in a visual policy, the existing
`Renderer(model)` / `update_scene(data)` / `render()` API returns NumPy arrays directly and
avoids transporting images through the display process. See [MuJoCo rendering](mujoco-rendering.md).
For external agents, `viewer.start_rpc(path)` attaches to the passive display; its state is the
last published snapshot. The display recomputes derived measurements from that snapshot.
Read caller-side `data.sensordata` and solver outputs when exact policy-step measurements are
required. [RPC control](../how-to/rpc-control.md) documents in-memory image transport and observations.

## Synchronous embedding

```python
with mojive.build(model=model, data=data, vsync=False) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

This binds the supplied objects directly and performs one UI frame per `sync()` on the calling
thread. Call less frequently when physics runs faster than display. Instantiated models default
to `external_clock=True`; UI pause/play cannot change that ownership. Set `external_clock=False`
explicitly when Mojive should own stepping. Asset-path builds retain their existing paused and
wall-clock scheduling behavior. Adapter integrations can use
`MuJoCoAdapter(external_clock=True)` with `load_model()` and `build_from_adapter()`.

## Run the rate and lifecycle example

```bash
make passive-viewer
make passive-viewer ARGS='--physics-hz 500 --hidden'
MOJIVE_LANGUAGE=zh_CN MOJIVE_UI_SCALE=2.5 make passive-viewer ARGS='--record --hidden --width 3200 --height 1800'
make passive-viewer ARGS='--renderer wgpu --output output/passive-viewer-wgpu --hidden'
```

The example reports physics/display rates, checks the exact physics clock and display progress
while the caller is idle, and verifies shutdown. It explicitly saves representative scene/window
captures and a JSON report under `output/passive-viewer/`.

```python
--8<-- "examples/passive_viewer.py"
```
