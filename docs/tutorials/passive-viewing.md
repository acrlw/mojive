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

Actuator slider changes apply once at a `sync()` boundary, so a stale display value cannot
keep overwriting policy output. Mouse perturbation writes the dragged body's `xfrc_applied`,
then clears it on release or close; other bodies' applied forces are preserved. Use
`with viewer.lock():` to serialize multiple caller threads touching the same model/data.
The window cannot pause, reset, or step caller-owned physics. Pose and topology editing are
disabled for the passive display. Camera navigation and visual inspection remain available.
`AdapterCaps.clock_control` distinguishes caller-only clock ownership from an externally
clocked adapter that can forward explicit pause/step commands, such as a remote publisher.

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
make passive-viewer ARGS='--renderer wgpu --output output/passive-viewer-wgpu --hidden'
```

The example reports physics/display rates, checks the exact physics clock and display progress
while the caller is idle, and verifies shutdown. It explicitly saves representative scene/window
captures and a JSON report under `output/passive-viewer/`.

```python
--8<-- "examples/passive_viewer.py"
```
