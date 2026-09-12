"""Cli: inspection."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .common import _resolve


def cmd_backends(args: argparse.Namespace) -> int:
    from mojive.application.backends import available_backends

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
    from mojive.scene.assets import assets_dir, list_assets

    names = list_assets()
    free: dict[str, int] = {}
    if not args.quick:
        from mojive.application.backends import make_adapter
        from mojive.scene.assets import resolve as resolve_asset

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
    from mojive.application.backends import make_adapter

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
    from mojive.adapters.conformance import check_adapter
    from mojive.adapters.mujoco import MuJoCoAdapter
    from mojive.adapters.mujoco.audit import audit_model

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


def cmd_conformance(args: argparse.Namespace) -> int:
    """Run adapter contract checks in a headless process."""
    from mojive.adapters.conformance import check_adapter
    from mojive.application.backends import make_adapter

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
    from mojive.application.composition import doctor

    report = doctor(_resolve(args.asset), args.backend, frames=args.frames)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for check, ok, note in report["checks"]:
            print(f"{'✓' if ok else '✗'} {check:<28} {note}")
        print(f"\n{'PASS' if report['ok'] else 'FAIL'}  frames {report['frames']}")
    return 0 if report["ok"] else 1
