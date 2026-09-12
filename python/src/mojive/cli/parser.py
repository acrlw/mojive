"""Cli: parser."""

from __future__ import annotations

import argparse
import json
import sys

from mojive.render.backend import DebugView, RenderFlag

from .capture import cmd_capture, cmd_keyframes, cmd_probe, cmd_record
from .common import DEFAULT_BACKEND, _positive_float, _positive_int, _setup_logging
from .control import cmd_control, cmd_operations, cmd_rpc_serve
from .inspection import (
    cmd_assets,
    cmd_audit,
    cmd_backends,
    cmd_conformance,
    cmd_doctor,
    cmd_inspect,
)
from .viewer import cmd_attach, cmd_canvas, cmd_editor, cmd_replay, cmd_serve, cmd_toy, cmd_view


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
    from mojive.control.errors import ControlError

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
