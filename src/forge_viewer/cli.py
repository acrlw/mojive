"""Command-line entry points for viewing, capture, and diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .log import configure, get_logger
from .render.backend import DebugView, RenderFlag

DEFAULT_BACKEND = "mujoco"
log = get_logger("cli")


def _setup_logging(json_mode: bool, verbose: bool) -> None:
    configure(verbose=verbose, warnings_only=json_mode)


def _resolve(name: str) -> Path:
    from .assets import resolve

    return resolve(name)


def cmd_backends(args: argparse.Namespace) -> int:
    from .backends import available_backends

    infos = available_backends()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": b.name,
                        "physics": b.physics,
                        "renderer": b.renderer,
                        "available": b.available,
                        "reason": b.reason,
                    }
                    for b in infos
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    width = max(len(b.name) for b in infos)
    for b in infos:
        mark = "✓" if b.available else "✗"
        line = f"{mark} {b.name:<{width}}  {b.physics} + {b.renderer}"
        print(line if b.available else f"{line}   ← {b.reason}")
    return 0


def cmd_assets(args: argparse.Namespace) -> int:
    from .assets import assets_dir, list_assets

    names = list_assets()
    free: dict[str, int] = {}
    if not args.quick:
        from .assets import resolve as resolve_asset
        from .backends import make_adapter

        for n in names:
            try:
                adapter = make_adapter(args.backend, resolve_asset(n))
                free[n] = sum(1 for node in adapter.nodes() if node.posable)
                adapter.release()
            except Exception:
                free[n] = -1

    if args.json:
        print(
            json.dumps(
                {"dir": str(assets_dir()), "assets": names, "free_bodies": free},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(f"{assets_dir()}  ({len(names)} assets)")
    width = max((len(n) for n in names), default=0)
    for n in names:
        count = free.get(n)
        if count is None:
            note = ""
        elif count < 0:
            note = "  load failed"
        elif count == 0:
            note = "  —"
        else:
            note = f"  {count} free bodies · gizmo and Ctrl+drag available"
        print(f"  {n:<{width}}{note}")
    if free and not any(v > 0 for v in free.values()):
        print("\n  No asset contains a free body; object manipulation is unavailable.")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from .backends import make_adapter

    path = _resolve(args.asset)
    adapter = make_adapter(args.backend, path)
    try:
        nodes = adapter.nodes()
        joints = adapter.joints()
        actuators = adapter.actuators()
        keyframes = adapter.keyframes() if adapter.caps.keyframes else []
        sensors = adapter.sensors() if adapter.caps.sensors else []
        source = adapter.scene_source()
        doc = {
            "asset": str(path),
            "backend": args.backend,
            "counts": {
                "nodes": len(nodes),
                "joints": len(joints),
                "actuators": len(actuators),
                "keyframes": len(keyframes),
                "sensors": len(sensors),
                "instances": source.instance_count,
                "meshes": len(source.meshes),
                "textures": len(source.textures),
                "materials": len(source.materials),
            },
            "nodes": [
                {
                    "id": n.node_id,
                    "name": n.name,
                    "kind": str(n.kind),
                    "parent": n.parent,
                    "object_id": int(n.object_id),
                    "posable": n.posable,
                }
                for n in nodes
            ],
            "joints": [
                {
                    "id": j.joint_id,
                    "name": j.name,
                    "kind": j.kind,
                    "limited": j.limited,
                    "range": list(j.range),
                    "dof": j.dof,
                }
                for j in joints
            ],
            "actuators": [
                {
                    "id": a.actuator_id,
                    "name": a.name,
                    "range": list(a.ctrl_range),
                    "ctrl_address": a.ctrl_address,
                    "ctrl_count": a.ctrl_count,
                }
                for a in actuators
            ],
            "keyframes": [{"id": k.keyframe_id, "name": k.name, "time": k.time} for k in keyframes],
            "sensors": [
                {
                    "id": sensor.sensor_id,
                    "name": sensor.name,
                    "kind": sensor.kind,
                    "adr": sensor.data_adr,
                    "dim": sensor.dim,
                }
                for sensor in sensors
            ],
        }
        if args.json:
            print(json.dumps(doc, ensure_ascii=False, indent=2))
            return 0

        print(f"{path.name}   backend {args.backend}")
        c = doc["counts"]
        print(
            f"  nodes {c['nodes']} · joints {c['joints']} · actuators {c['actuators']} · "
            f"keyframes {c['keyframes']} · sensors {c['sensors']} · instances {c['instances']} · "
            f"meshes {c['meshes']} · textures {c['textures']}"
        )
        print("\nScene tree:")
        _print_tree(nodes)
        if joints:
            print("\nJoints:")
            for j in joints:
                lim = f"[{j.range[0]:.3g}, {j.range[1]:.3g}]" if j.limited else "unlimited"
                print(f"  {j.joint_id:>3}  {j.name:<24} {j.kind:<6} dof={j.dof}  {lim}")
        if actuators:
            print("\nActuators:")
            for a in actuators:
                print(
                    f"  {a.actuator_id:>3}  {a.name:<24} ctrl[{a.ctrl_address}:"
                    f"{a.ctrl_address + a.ctrl_count}]  "
                    f"[{a.ctrl_range[0]:.3g}, {a.ctrl_range[1]:.3g}]"
                )
        return 0
    finally:
        adapter.release()


def cmd_audit(args: argparse.Namespace) -> int:
    """Report exactly what forge will render, hide, degrade, or skip in a MuJoCo model."""
    from .adapters.conformance import check_adapter
    from .adapters.mujoco_adapter import MuJoCoAdapter
    from .mujoco_audit import audit_model

    path = _resolve(args.asset)
    adapter = MuJoCoAdapter(path)
    try:
        report = audit_model(adapter.model)
        report["asset"] = str(path)
        report["adapter_caps"] = asdict(adapter.caps)
        try:
            runtime = check_adapter(adapter)
            report["runtime_validation"] = {
                "ok": runtime.ok,
                "checks": [asdict(check) for check in runtime.checks],
            }
        except Exception as exc:
            report["runtime_validation"] = {"ok": False, "error": str(exc)}
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            counts = report["counts"]
            print(
                f"{path.name}: {counts['geom']} geom, {counts['site']} site, "
                f"{counts['tendon']} tendon, {counts['camera']} camera"
            )
            for finding in report["findings"]:
                print(
                    f"  {finding['status'].upper():<11} {finding['feature']:<20} "
                    f"x{finding['count']:<4} {finding['detail']}"
                )
            if not report["findings"]:
                print("  SUPPORTED   No skipped or degraded visual features found")
            enabled = [
                name
                for name, value in report["adapter_caps"].items()
                if name not in ("name", "notes") and value is True
            ]
            disabled = [
                name
                for name, value in report["adapter_caps"].items()
                if name not in ("name", "notes") and value is False
            ]
            print(f"\nAdapter API: {', '.join(enabled)}")
            print(f"Not implemented: {', '.join(disabled) or 'none'}")
            runtime = report["runtime_validation"]
            print(
                f"Runtime frame: {'PASS' if runtime['ok'] else 'FAIL'}"
                + (f"  {runtime['error']}" if runtime.get("error") else "")
            )
            print("\nMuJoCo visualization flags:")
            for group, items in report["coverage"].items():
                print(f"  {group}")
                for item in items:
                    print(
                        f"    {item['status'].upper():<11} {item['feature']:<22} {item['detail']}"
                    )
        failed = bool(report["unsupported"] or not report["runtime_validation"]["ok"])
        return 1 if args.strict and failed else 0
    finally:
        adapter.release()


def _print_tree(nodes, parent: int = -1, depth: int = 0) -> None:
    for n in nodes:
        if n.parent != parent:
            continue
        tag = " ◆" if n.posable else ""
        print(f"  {'  ' * depth}{n.name}  ({n.kind}, id={n.object_id}){tag}")
        _print_tree(nodes, n.node_id, depth + 1)


def cmd_view(args: argparse.Namespace) -> int:
    from .composition import build

    viewer = build(
        _resolve(args.asset),
        args.backend,
        paused=args.paused,
        vsync=not args.no_vsync,
    )
    try:
        for name in args.enable_render:
            viewer.backend.set_flag(RenderFlag(name), True)
        viewer.run()
    finally:
        viewer.release()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Publish simulation snapshots from a headless physics process."""
    import time

    from . import commands as cmd
    from .adapters.base import FrameNeeds
    from .backends import make_adapter
    from .recording import SnapshotWriter
    from .remote import RemoteFrame, SnapshotPublisher, handle_session_command, snapshot_structure
    from .session import Session

    path = _resolve(args.asset)
    session = Session(make_adapter(args.backend, path), path)
    publisher = SnapshotPublisher(args.host, args.port)
    writer = SnapshotWriter(Path(args.record_snapshot)) if args.record_snapshot else None
    if args.paused and not session.paused:
        session.submit(cmd.Pause())
    needs = FrameNeeds(
        poses=True,
        qpos=True,
        qvel=True,
        contacts=True,
        tendons=True,
        actuator=True,
        sensors=True,
        deformables=True,
        diagnostics=True,
    )
    period = 1.0 / max(float(args.hz), 1.0)
    previous = time.perf_counter()
    deadline = previous
    published_generation = -1
    log.info("Publishing {} at {}:{}", path.name, args.host, args.port)
    try:
        while True:
            now = time.perf_counter()
            publisher.pump_commands(lambda message: handle_session_command(session, message))
            frame = session.tick(needs, wall_dt=max(0.0, now - previous))
            previous = now
            if session.structure_generation != published_generation:
                structure = snapshot_structure(session)
                publisher.publish_structure(structure)
                if writer is not None:
                    writer.write(structure)
                published_generation = session.structure_generation
            sequence = publisher.publish_frame(frame)
            if writer is not None:
                writer.write(RemoteFrame(sequence, frame, tuple(frame.debug_commands or ())))
            deadline += period
            delay = deadline - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
            else:
                deadline = time.perf_counter()
    finally:
        if writer is not None:
            writer.close()
        publisher.close()
        session.release()


def cmd_attach(args: argparse.Namespace) -> int:
    """Attach an independent forge window to a snapshot publisher."""
    from .composition import build_from_adapter
    from .remote import RemoteSceneAdapter

    viewer = build_from_adapter(
        RemoteSceneAdapter(args.host, args.port),
        vsync=not args.no_vsync,
        title=args.title,
    )
    try:
        viewer.backend.set_debug_view(DebugView(args.debug_view))
        viewer.run()
    finally:
        viewer.release()
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Republish a recorded snapshot stream through the normal attach path."""
    import time
    from dataclasses import replace

    from .commands import CommandResult
    from .recording import read_snapshots
    from .remote import RemoteFrame, RemoteStructure, SnapshotPublisher

    publisher = SnapshotPublisher(args.host, args.port)
    path = Path(args.snapshot)
    log.info("Replaying {} at {}:{}", path.name, args.host, args.port)

    def wait(seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0.0:
            publisher.pump_commands(lambda _message: CommandResult.bad("replay is read-only"))
            time.sleep(min(remaining, 0.01))

    try:
        while True:
            previous_time = None
            packets = 0
            for packet in read_snapshots(path):
                packets += 1
                if isinstance(packet, RemoteStructure):
                    caps = replace(
                        packet.caps,
                        name=f"replay:{packet.caps.name}",
                        simulation=False,
                        write_pose=False,
                        write_qpos=False,
                        perturb=False,
                        raycast=False,
                        equality_constraints=False,
                        model_cameras=bool(packet.cameras),
                        visual_groups=False,
                        reload=False,
                    )
                    publisher.publish_structure(replace(packet, caps=caps))
                elif isinstance(packet, RemoteFrame):
                    frame_time = float(packet.frame.time)
                    if previous_time is not None:
                        wait(max(0.0, frame_time - previous_time) / max(args.speed, 0.01))
                    publisher.publish_frame(
                        replace(packet.frame, paused=True), packet.debug_commands
                    )
                    previous_time = frame_time
            if packets == 0:
                raise ValueError("snapshot recording is empty")
            if not args.loop:
                return 0
    finally:
        publisher.close()


def cmd_canvas(args: argparse.Namespace) -> int:
    from .composition import build_scene
    from .demos import canvas_scene, lighting_scene

    scene = lighting_scene() if args.demo == "lighting" else canvas_scene()
    viewer = build_scene(scene, vsync=not args.no_vsync, title=f"forge {args.demo}")
    try:
        for name in args.enable_render:
            viewer.backend.set_flag(RenderFlag(name), True)
        if args.demo == "lighting":
            viewer.backend.set_flag(RenderFlag.FOG, True)
            viewer.backend.set_flag(RenderFlag.HAZE, True)
            if viewer.backend.debug is not None:
                from .render.debugdraw import Occlusion

                viewer.backend.debug.layer("demo.lighting.help", Occlusion.ALWAYS).text(
                    "atmosphere",
                    (0.0, -1.0, 3.2),
                    "F9: toggle fog / haze   |   near → far",
                    align=(0.5, 1.0),
                )
        if args.demo == "text" and viewer.backend.debug is not None:
            from .render.debugdraw import Occlusion

            depth = viewer.backend.debug.layer("demo.text.depth", Occlusion.DEPTH)
            depth.text("crate", (-0.7, 0.0, 1.0), "crate", offset_px=(0, -8), align=(0.5, 1))
            depth.text("ball", (0.45, -0.3, 0.92), "ball 0.42 m", offset_px=(0, -8), align=(0.5, 1))
            always = viewer.backend.debug.layer("demo.text.always", Occlusion.ALWAYS)
            always.text("origin", (0, 0, 0), "world origin", offset_px=(8, 8), align=(0, 0))
        viewer.run()
    finally:
        viewer.release()
    return 0


def cmd_toy(args: argparse.Namespace) -> int:
    """Open the dependency-free reference physics adapter through the production viewer."""
    from .backends import make_adapter
    from .composition import build_from_adapter

    viewer = build_from_adapter(
        make_adapter("toy"), vsync=not args.no_vsync, title="forge toy physics"
    )
    try:
        viewer.run()
    finally:
        viewer.release()
    return 0


def cmd_conformance(args: argparse.Namespace) -> int:
    """Run adapter contract checks in a headless process."""
    from .adapters.conformance import check_adapter
    from .backends import make_adapter

    asset = _resolve(args.asset) if args.asset else None
    adapter = make_adapter(args.backend, asset)
    try:
        report = check_adapter(adapter)
        if args.json:
            print(
                json.dumps(
                    {
                        "backend": report.backend,
                        "ok": report.ok,
                        "checks": [check.__dict__ for check in report.checks],
                    },
                    indent=2,
                )
            )
        else:
            for check in report.checks:
                print(f"{'PASS' if check.ok else 'FAIL':<4}  {check.name:<20} {check.detail}")
            print(f"\n{'PASS' if report.ok else 'FAIL'}  adapter={report.backend}")
        return 0 if report.ok else 1
    finally:
        adapter.release()


def cmd_doctor(args: argparse.Namespace) -> int:
    from .composition import doctor

    report = doctor(_resolve(args.asset), args.backend, frames=args.frames)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for check, ok, note in report["checks"]:
            print(f"{'✓' if ok else '✗'} {check:<28} {note}")
        print(f"\n{'PASS' if report['ok'] else 'FAIL'}  frames {report['frames']}")
    return 0 if report["ok"] else 1


def cmd_capture(args: argparse.Namespace) -> int:
    from .composition import capture

    size = (args.width, args.height) if args.width and args.height else None
    output = Path(args.output).expanduser().resolve()
    ok = capture(
        _resolve(args.asset),
        output,
        args.backend,
        include_ui=args.include_ui,
        size=size,
        render_flags=tuple(args.enable_render),
        camera_name=args.camera,
    )
    if not ok:
        print("Capture failed", file=sys.stderr)
        return 1
    print(output)
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    from .composition import build

    viewer = build(
        _resolve(args.asset),
        args.backend,
        vsync=False,
        width=args.width,
        height=args.height,
        title="forge recording",
    )
    try:
        for name in args.enable_render:
            viewer.backend.set_flag(RenderFlag(name), True)
        viewer.record(
            Path(args.output),
            frames=args.frames,
            fps=args.fps,
            size=(args.width, args.height),
        )
    finally:
        viewer.release()
    print(args.output)
    return 0


def cmd_keyframes(args: argparse.Namespace) -> int:
    from . import commands as cmd
    from .composition import build

    viewer = build(
        _resolve(args.asset),
        args.backend,
        paused=True,
        vsync=False,
        width=args.width,
        height=args.height,
        title="forge keyframes",
    )
    try:
        for name in args.enable_render:
            viewer.backend.set_flag(RenderFlag(name), True)
        count = len(viewer.session.keyframes)
        if not count:
            print("Model has no keyframes", file=sys.stderr)
            return 1
        viewer.session.submit(cmd.LoadKeyframe(0))
        viewer.app.set_fixed_render_size(args.width, args.height)
        if args.camera:
            camera = next(
                (item for item in viewer.session.cameras if item.name == args.camera), None
            )
            if camera is None:
                raise ValueError(f"model camera {args.camera!r} is unavailable")
            viewer.app.select_model_camera(camera.camera_id)
        viewer.sync()  # compile passes and frame the loaded pose; not part of the video
        if not args.camera:
            viewer.app.camera.distance *= args.camera_distance_scale

        def load(index, current):
            result = current.session.submit(cmd.LoadKeyframe(index))
            if not result.ok:
                raise RuntimeError(result.message)

        viewer.record(
            Path(args.output),
            frames=count,
            fps=args.fps,
            before_frame=load,
            size=(args.width, args.height),
        )
    finally:
        viewer.release()
    print(args.output)
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    import subprocess

    tool = Path(__file__).resolve().parents[2] / "tools" / "probe_gl.py"
    return subprocess.call([sys.executable, str(tool)])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forge-viewer", description="Interactive 3D simulation viewer")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def with_asset(sp):
        sp.add_argument("asset", help="Path or asset name; the extension is optional")
        sp.add_argument("-b", "--backend", default=DEFAULT_BACKEND)
        return sp

    def with_render_flags(sp):
        sp.add_argument(
            "--enable-render",
            action="append",
            default=[],
            choices=tuple(x.value for x in RenderFlag),
            metavar="FLAG",
            help="enable a supported render flag before the first frame (repeatable)",
        )
        return sp

    sp = with_render_flags(with_asset(sub.add_parser("view", help="Open the viewer")))
    sp.add_argument("--paused", action="store_true")
    sp.add_argument("--no-vsync", action="store_true")
    sp.set_defaults(func=cmd_view, json=False)

    sp = with_render_flags(sub.add_parser("canvas", help="Open a procedural 3D canvas"))
    sp.add_argument("--demo", choices=("canvas", "lighting", "text"), default="canvas")
    sp.add_argument("--no-vsync", action="store_true")
    sp.set_defaults(func=cmd_canvas, json=False)

    sp = sub.add_parser("toy", help="Open the toy physics backend")
    sp.add_argument("--no-vsync", action="store_true")
    sp.set_defaults(func=cmd_toy, json=False)

    sp = sub.add_parser("conformance", help="Validate a SceneAdapter without a window")
    sp.add_argument("backend", nargs="?", default="toy")
    sp.add_argument("--asset", help="Asset for adapters that load model files")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_conformance)

    sp = with_asset(sub.add_parser("serve", help="Run physics and publish live snapshots"))
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=47650)
    sp.add_argument("--hz", type=float, default=120.0, help="snapshot publish rate")
    sp.add_argument("--paused", action="store_true")
    sp.add_argument("--record-snapshot", metavar="FILE", help="append the published stream")
    sp.set_defaults(func=cmd_serve, json=False)

    sp = sub.add_parser("attach", help="Open a viewer connected to live snapshots")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=47650)
    sp.add_argument("--title", default="forge remote")
    sp.add_argument(
        "--debug-view",
        choices=tuple(view.value for view in DebugView),
        default=DebugView.SHADED.value,
    )
    sp.add_argument("--no-vsync", action="store_true")
    sp.set_defaults(func=cmd_attach, json=False)

    sp = sub.add_parser("replay", help="Replay recorded snapshots")
    sp.add_argument("snapshot")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=47650)
    sp.add_argument("--speed", type=float, default=1.0)
    sp.add_argument("--loop", action="store_true")
    sp.set_defaults(func=cmd_replay, json=False)

    sp = with_asset(sub.add_parser("doctor", help="Run a 90-frame smoke test"))
    sp.add_argument("--json", action="store_true")
    sp.add_argument("-n", "--frames", type=int, default=90)
    sp.set_defaults(func=cmd_doctor)

    sp = with_asset(sub.add_parser("inspect", help="Print the scene tree and joint table"))
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_inspect)

    sp = with_asset(sub.add_parser("audit", help="audit MuJoCo visual coverage without a window"))
    sp.add_argument("--json", action="store_true")
    sp.add_argument(
        "--strict", action="store_true", help="exit 1 when an unsupported feature is present"
    )
    sp.set_defaults(func=cmd_audit)

    sp = with_render_flags(with_asset(sub.add_parser("capture", help="Save a PNG image")))
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--include-ui", action="store_true", help="Include panels and gizmos")
    sp.add_argument("--width", type=int, default=0, help="Output width, such as 3840 for 4K")
    sp.add_argument("--height", type=int, default=0)
    sp.add_argument("--camera", default="", help="capture through a named model camera")
    sp.set_defaults(func=cmd_capture, json=False)

    sp = with_render_flags(with_asset(sub.add_parser("record", help="Record viewport video")))
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--frames", type=int, default=300)
    sp.add_argument("--fps", type=float, default=30.0)
    sp.add_argument("--width", type=int, default=1280)
    sp.add_argument("--height", type=int, default=720)
    sp.set_defaults(func=cmd_record, json=False)

    sp = with_render_flags(with_asset(sub.add_parser("keyframes", help="Record model keyframes")))
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--fps", type=float, default=60.0)
    sp.add_argument("--width", type=int, default=1920)
    sp.add_argument("--height", type=int, default=1080)
    sp.add_argument("--camera-distance-scale", type=float, default=1.0)
    sp.add_argument("--camera", default="", help="follow a named model camera")
    sp.set_defaults(func=cmd_keyframes, json=False)

    sp = sub.add_parser("backends", help="List backend availability")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_backends)

    sp = sub.add_parser("assets", help="List assets and free-body support")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--quick", action="store_true", help="List names without loading assets")
    sp.add_argument("-b", "--backend", default=DEFAULT_BACKEND)
    sp.set_defaults(func=cmd_assets)

    sp = sub.add_parser("probe", help="Probe OpenGL capabilities")
    sp.set_defaults(func=cmd_probe, json=False)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(getattr(args, "json", False), args.verbose)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
