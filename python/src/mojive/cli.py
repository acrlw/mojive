"""Command-line entry points for viewing, capture, and diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import ExitStack
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


def _positive_int(value: str) -> int:
    try:
        result = int(value)
        if result > 0:
            return result
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("value must be a positive integer")


def _positive_float(value: str) -> float:
    try:
        result = float(value)
        if math.isfinite(result) and result > 0.0:
            return result
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("value must be finite and positive")


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
        renderer = "OpenGL" if b.renderer == "opengl" else b.renderer
        line = f"{mark} {b.name:<{width}}  {b.physics} + {renderer}"
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
                try:
                    free[n] = sum(1 for node in adapter.nodes() if node.posable)
                finally:
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
                    "type": str(n.type),
                    "parent": n.parent,
                    "object_id": int(n.object_id),
                    "posable": n.posable,
                    "source_editable": n.source_editable,
                }
                for n in nodes
            ],
            "joints": [
                {
                    "id": j.joint_id,
                    "name": j.name,
                    "type": j.type,
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
                    "type": sensor.type,
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
                print(f"  {j.joint_id:>3}  {j.name:<24} {j.type:<6} dof={j.dof}  {lim}")
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
    """Report exactly what Mojive will render, hide, degrade, or skip in a MuJoCo model."""
    if args.backend != "mujoco":
        raise ValueError("audit supports the mujoco adapter only")
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
            schema = report["schema_coverage"]
            summary = ", ".join(f"{status}={count}" for status, count in schema["counts"].items())
            print(f"MuJoCo {schema['mujoco_version']} schema: {summary}")
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
    children = {}
    for n in nodes:
        children.setdefault(n.parent, []).append(n)
    pending = [(iter(children.get(parent, ())), depth)]
    visited = set()
    while pending:
        siblings, depth = pending[-1]
        n = next(siblings, None)
        if n is None:
            pending.pop()
            continue
        if n.node_id in visited:
            raise ValueError(f"Scene tree contains a cycle or duplicate node ID: {n.node_id}")
        visited.add(n.node_id)
        tag = " ◆" if n.posable else ""
        print(f"  {'  ' * depth}{n.name}  ({n.type}, id={n.object_id}){tag}")
        if n.node_id in children:
            pending.append((iter(children[n.node_id]), depth + 1))


def cmd_view(args: argparse.Namespace) -> int:
    from .composition import build, build_workspace

    compose = build_workspace if args.backend == "mujoco" else build
    viewer = compose(
        _resolve(args.asset),
        args.backend,
        paused=args.paused,
        vsync=not args.no_vsync,
    )
    try:
        if getattr(args, "rpc_socket", None):
            viewer.start_rpc(Path(args.rpc_socket))
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
    with ExitStack() as resources:
        session = Session(make_adapter(args.backend, path), path)
        resources.callback(session.release)
        publisher = SnapshotPublisher(args.host, args.port)
        resources.callback(publisher.close)
        writer = (
            resources.enter_context(SnapshotWriter(Path(args.record_snapshot)))
            if args.record_snapshot
            else None
        )
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
        period = 1.0 / args.hz
        previous = time.perf_counter()
        deadline = previous
        published_generation = -1
        log.info("Publishing {} at {}:{}", path.name, args.host, args.port)

        def handle_command(message):
            return handle_session_command(session, message)

        while True:
            now = time.perf_counter()
            publisher.pump_commands(handle_command)
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
                writer.write(
                    RemoteFrame(
                        sequence,
                        frame,
                        tuple(frame.debug_commands or ()),
                        structure_revision=published_generation,
                    )
                )
            deadline += period
            delay = deadline - time.perf_counter()
            if delay > 0.0:
                # Publication cadence must not delay remote command responses.
                while delay > 0.0:
                    time.sleep(min(delay, 0.01))
                    publisher.pump_commands(handle_command)
                    delay = deadline - time.perf_counter()
            else:
                deadline = time.perf_counter()


def cmd_attach(args: argparse.Namespace) -> int:
    """Attach an independent Mojive window to a snapshot publisher."""
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

    from .adapters.base import AdapterCaps
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
                    caps = AdapterCaps(
                        name=f"replay:{packet.caps.name}",
                        external_clock=True,
                        contacts=packet.caps.contacts,
                        model_cameras=bool(packet.cameras),
                        sensors=packet.caps.sensors,
                        notes=(*packet.caps.notes, "Recorded playback is read-only"),
                    )
                    publisher.publish_structure(replace(packet, caps=caps))
                elif isinstance(packet, RemoteFrame):
                    frame_time = float(packet.frame.time)
                    if previous_time is not None:
                        wait(max(0.0, frame_time - previous_time) / args.speed)
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
    from .scene import Scene

    if args.demo == "empty":
        scene = Scene()
    else:
        scene = lighting_scene() if args.demo == "lighting" else canvas_scene()
    viewer = build_scene(scene, vsync=not args.no_vsync, title=f"Mojive · {args.demo}")
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


def cmd_editor(args: argparse.Namespace) -> int:
    from .composition import build_editor, build_workspace

    viewer = (
        build_workspace(
            _resolve(args.asset),
            "mujoco",
            vsync=not args.no_vsync,
            title="Mojive",
        )
        if args.asset
        else build_editor(vsync=not args.no_vsync, title="Mojive")
    )
    try:
        if getattr(args, "rpc_socket", None):
            viewer.start_rpc(Path(args.rpc_socket))
        viewer.run()
    finally:
        viewer.release()
    return 0


def cmd_toy(args: argparse.Namespace) -> int:
    """Open the dependency-free reference physics adapter through the production viewer."""
    from .backends import make_adapter
    from .composition import build_from_adapter

    viewer = build_from_adapter(
        make_adapter("toy"), vsync=not args.no_vsync, title="Mojive toy physics"
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
    if bool(args.width) != bool(args.height):
        raise ValueError("Capture width and height must be provided together")
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
        title="Mojive recording",
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
        title="Mojive keyframes",
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

    return subprocess.call([sys.executable, "-m", "mojive.tools.probe_gl"])


def cmd_rpc_serve(args: argparse.Namespace) -> int:
    from .backends import make_adapter
    from .control_rpc import ControlServer, ControlService

    path = _resolve(args.asset)
    service = ControlService(make_adapter(args.backend, path), path)
    try:
        server = ControlServer(Path(args.socket), service)
        log.info("Control RPC listening on {}", server.socket_path)
        try:
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        service.close()
    return 0


def cmd_control(args: argparse.Namespace) -> int:
    from .control_rpc import RpcClient, RpcError

    try:
        try:
            source = "{}" if args.params is None else args.params
            if args.params_file is not None:
                source = (
                    sys.stdin.read()
                    if args.params_file == "-"
                    else Path(args.params_file).expanduser().read_text(encoding="utf-8")
                )
            params = json.loads(source)
        except ValueError as exc:
            raise RpcError("invalid_params", f"Invalid parameter JSON: {exc}") from exc
        if not isinstance(params, dict):
            raise RpcError("invalid_params", "Parameters must be a JSON object")
        with RpcClient(Path(args.socket), args.timeout) as client:
            result = client.call(args.method, params)
    except (RpcError, ValueError) as exc:
        if not args.json:
            raise
        if not isinstance(exc, RpcError):
            exc = RpcError("invalid_params", str(exc))
        print(json.dumps({"error": exc.payload()}, indent=2))
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
    elif isinstance(result, dict) and result.get("message"):
        print(result["message"])
    else:
        print(json.dumps(result, indent=2))
    return 0


def cmd_operations(args: argparse.Namespace) -> int:
    """Describe installed operation contracts without starting a viewer or service."""
    from .control_errors import ControlError
    from .operations import OPERATIONS

    if args.name is not None and args.name not in OPERATIONS:
        raise ControlError("unknown_method", f"Unknown control method: {args.name}")
    selected = [OPERATIONS[args.name]] if args.name else OPERATIONS.values()
    descriptions = [
        item.specification() for item in selected if args.scope is None or item.scope == args.scope
    ]
    if args.json:
        print(
            json.dumps(
                {
                    "schema_dialect": "https://json-schema.org/draft/2020-12/schema",
                    "operations": descriptions,
                },
                indent=2,
            )
        )
    elif args.name:
        print(json.dumps(descriptions, indent=2))
    else:
        for item in descriptions:
            action = "write" if item["mutates"] else "read"
            print(f"{item['name']:<28} {item['scope']:<8} {action:<5} {item['description']}")
        print("\nUse control describe_operations to check live availability and document identity.")
    return 0


def build_parser(*, parser_class=argparse.ArgumentParser) -> argparse.ArgumentParser:
    p = parser_class(prog="mojive", description="Interactive 3D simulation viewer")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def with_asset(sp):
        sp.add_argument("asset", help="Path or asset name; the extension is optional")
        sp.add_argument(
            "-b",
            "--adapter",
            "--backend",
            dest="backend",
            default=DEFAULT_BACKEND,
            help="Scene adapter name",
        )
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
    startup = sp.add_mutually_exclusive_group()
    startup.add_argument("--paused", dest="paused", action="store_true")
    startup.add_argument("--play", dest="paused", action="store_false")
    sp.add_argument("--no-vsync", action="store_true")
    sp.add_argument("--rpc-socket", help="Expose this viewer through a local control socket")
    sp.set_defaults(func=cmd_view, json=False, paused=True)

    sp = with_render_flags(sub.add_parser("canvas", help="Open a procedural 3D canvas"))
    sp.add_argument("--demo", choices=("empty", "canvas", "lighting", "text"), default="canvas")
    sp.add_argument("--no-vsync", action="store_true")
    sp.set_defaults(func=cmd_canvas, json=False)

    sp = sub.add_parser("editor", help="Open a model and scene workspace")
    sp.add_argument("asset", nargs="?", help="Optional MJCF or URDF path or asset name")
    sp.add_argument("--no-vsync", action="store_true")
    sp.add_argument("--rpc-socket", help="Expose this viewer through a local control socket")
    sp.set_defaults(func=cmd_editor, json=False)

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
    sp.add_argument("--hz", type=_positive_float, default=120.0, help="snapshot publish rate")
    sp.add_argument("--paused", action="store_true")
    sp.add_argument("--record-snapshot", metavar="FILE", help="append the published stream")
    sp.set_defaults(func=cmd_serve, json=False)

    sp = sub.add_parser("attach", help="Open a viewer connected to live snapshots")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=47650)
    sp.add_argument("--title", default="Mojive remote")
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
    sp.add_argument("--speed", type=_positive_float, default=1.0)
    sp.add_argument("--loop", action="store_true")
    sp.set_defaults(func=cmd_replay, json=False)

    sp = with_asset(sub.add_parser("doctor", help="Run a 90-frame smoke test"))
    sp.add_argument("--json", action="store_true")
    sp.add_argument("-n", "--frames", type=_positive_int, default=90)
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
    sp.add_argument(
        "--width", type=_positive_int, default=0, help="Output width, such as 3840 for 4K"
    )
    sp.add_argument("--height", type=_positive_int, default=0)
    sp.add_argument("--camera", default="", help="capture through a named model camera")
    sp.set_defaults(func=cmd_capture, json=False)

    sp = with_render_flags(with_asset(sub.add_parser("record", help="Record viewport video")))
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--frames", type=_positive_int, default=300)
    sp.add_argument("--fps", type=_positive_float, default=30.0)
    sp.add_argument("--width", type=_positive_int, default=1280)
    sp.add_argument("--height", type=_positive_int, default=720)
    sp.set_defaults(func=cmd_record, json=False)

    sp = with_render_flags(with_asset(sub.add_parser("keyframes", help="Record model keyframes")))
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--fps", type=_positive_float, default=60.0)
    sp.add_argument("--width", type=_positive_int, default=1920)
    sp.add_argument("--height", type=_positive_int, default=1080)
    sp.add_argument("--camera-distance-scale", type=_positive_float, default=1.0)
    sp.add_argument("--camera", default="", help="follow a named model camera")
    sp.set_defaults(func=cmd_keyframes, json=False)

    sp = sub.add_parser("backends", help="List backend availability")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_backends)

    sp = sub.add_parser("assets", help="List assets and free-body support")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--quick", action="store_true", help="List names without loading assets")
    sp.add_argument(
        "-b",
        "--adapter",
        "--backend",
        dest="backend",
        default=DEFAULT_BACKEND,
        help="Scene adapter name",
    )
    sp.set_defaults(func=cmd_assets)

    sp = sub.add_parser("probe", help="Probe OpenGL capabilities")
    sp.set_defaults(func=cmd_probe, json=False)

    sp = with_asset(sub.add_parser("rpc-serve", help="Run the local scene control service"))
    sp.add_argument("--socket", default="output/mojive.sock")
    sp.set_defaults(func=cmd_rpc_serve, json=False)

    sp = sub.add_parser("control", help="Send one typed command to a local control service")
    sp.add_argument("method")
    params = sp.add_mutually_exclusive_group()
    params.add_argument("--params", help="JSON object containing method parameters")
    params.add_argument(
        "--params-file", metavar="FILE", help="Read parameter JSON from a UTF-8 file; - reads stdin"
    )
    sp.add_argument("--socket", default="output/mojive.sock")
    sp.add_argument("--timeout", type=float, default=5.0)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_control)

    sp = sub.add_parser("operations", help="Describe installed control schemas without a service")
    sp.add_argument("name", nargs="?", help="Optional operation name")
    sp.add_argument("--scope", choices=("scene", "capture", "viewport", "service"))
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_operations)
    return p


class _UsageError(Exception):
    def __init__(self, parser, message):
        super().__init__(message)
        self.parser = parser


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise _UsageError(self, message)


def _report_error(exc: Exception, code: str, json_mode: bool) -> int:
    from .control_errors import ControlError

    if json_mode:
        error = (
            exc.payload() if isinstance(exc, ControlError) else {"code": code, "message": str(exc)}
        )
        print(json.dumps({"error": error}, indent=2))
    else:
        print(str(exc), file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser(parser_class=_ArgumentParser).parse_args(argv)
    except _UsageError as exc:
        json_mode = "--json" in argv[: argv.index("--")] if "--" in argv else "--json" in argv
        if not json_mode:
            exc.parser.print_usage(sys.stderr)
        return _report_error(exc, "invalid_arguments", json_mode)
    _setup_logging(getattr(args, "json", False), args.verbose)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        return _report_error(e, "not_found", args.json)
    except OSError as e:
        return _report_error(e, "io_error", args.json)
    except ValueError as e:
        return _report_error(e, "invalid_params", args.json)
    except RuntimeError as e:
        return _report_error(e, "operation_failed", args.json)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
