"""Headless CLI discovery, strict input boundaries, and linear scene traversal."""

import io
import json
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from mojive import cli
from mojive.control.rpc import ControlServer, ControlService, RpcClient


@pytest.mark.parametrize("argument", ["--params", "--params-file"])
def test_control_accepts_unicode_parameter_sources(argument, tmp_path, monkeypatch, capsys):
    params = {"name": "测试 'quoted' $literal", "position": [1, 2, 3]}
    content = json.dumps(params, ensure_ascii=False)
    path = tmp_path / "parameters.json"
    path.write_text(content, encoding="utf-8")
    received = []
    monkeypatch.setattr(
        RpcClient, "call", lambda self, name, values: received.append((name, values))
    )
    assert (
        cli.main(
            [
                "control",
                "add_scene_object",
                argument,
                content if argument == "--params" else str(path),
                "--json",
            ]
        )
        == 0
    )
    assert received == [("add_scene_object", params)]
    assert json.loads(capsys.readouterr().out) is None


@pytest.mark.parametrize(
    "content", ["[]", "{broken", '{"value": NaN}', '{"value": Infinity}', '{"value": 1e999}']
)
def test_invalid_stdin_never_connects(content, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(content))
    monkeypatch.setattr(RpcClient, "_connect", lambda self: pytest.fail("Invalid input connected"))
    assert cli.main(["control", "edit_scene", "--params-file", "-", "--json"]) == 2
    result = capsys.readouterr()
    assert json.loads(result.out)["error"]["code"] == "invalid_params"
    assert not result.err


@pytest.mark.parametrize(
    "args",
    [
        ["control", "hello", "--timeout", "no"],
        ["control", "hello", "--params", "{}", "--params-file", "-"],
        ["control", "hello", "--unknown"],
        ["operations", "--scope", "invalid"],
        ["missing-command"],
    ],
)
def test_json_usage_errors_are_one_document(args, capsys):
    assert cli.main([*args, "--json"]) == 2
    result = capsys.readouterr()
    assert json.loads(result.out)["error"]["code"] == "invalid_arguments"
    assert not result.err


def test_text_usage_error_preserves_subcommand_help(capsys):
    assert cli.main(["control", "hello", "--timeout", "invalid"]) == 2
    result = capsys.readouterr()
    assert not result.out
    assert "usage: mojive control" in result.err
    assert "invalid float value" in result.err


@pytest.mark.parametrize(
    "args",
    [
        ["control", "hello", "--params-file", "/nonexistent/parameters.json"],
        ["inspect", "/nonexistent/scene.xml"],
    ],
)
def test_json_missing_files_are_machine_readable(args, capsys):
    assert cli.main([*args, "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "not_found"


def test_offline_catalog_matches_live_contracts(capsys):
    from mojive.adapters.static import StaticSceneAdapter
    from mojive.control.operations import OPERATIONS
    from mojive.scene import Scene

    assert cli.main(["operations", "--json"]) == 0
    offline = json.loads(capsys.readouterr().out)
    assert len(offline["operations"]) == len(OPERATIONS)
    service = ControlService(StaticSceneAdapter(Scene()))
    try:
        live = service.dispatch("describe_operations", {})
        for item in live["operations"]:
            item.pop("available")
            item.pop("unavailable_reason")
        assert offline["operations"] == live["operations"]
        # Callers cannot mutate the central validator or future discovery output.
        offline["operations"][0]["input_schema"].clear()
        assert OPERATIONS[live["operations"][0]["name"]].specification() == live["operations"][0]
    finally:
        service.close()


def test_offline_catalog_filters_and_unknown_names(capsys):
    assert cli.main(["operations", "set_camera", "--json"]) == 0
    assert [item["name"] for item in json.loads(capsys.readouterr().out)["operations"]] == [
        "set_camera"
    ]
    assert cli.main(["operations", "--scope", "capture", "--json"]) == 0
    assert {item["scope"] for item in json.loads(capsys.readouterr().out)["operations"]} == {
        "capture"
    }
    assert cli.main(["operations", "not_a_method", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "unknown_method"


def test_offline_catalog_does_not_start_application_or_graphics():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from mojive.cli import main
assert main(['operations', 'set_camera', '--json']) == 0
for name in ('mojive.control.application', 'mojive.ui.app', 'glfw', 'mujoco', 'moderngl', 'wgpu'):
    assert name not in sys.modules, name
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


class _Node:
    parent_reads = 0

    def __init__(self, node_id, parent):
        self.node_id = self.object_id = node_id
        self._parent = parent
        self.name = str(node_id)
        self.type = "geom"
        self.posable = False

    @property
    def parent(self):
        type(self).parent_reads += 1
        return self._parent


def test_tree_printing_visits_edges_once_without_recursing(capsys):
    nodes = [_Node(i, i - 1) for i in range(1500)]
    _Node.parent_reads = 0
    cli._print_tree(nodes)
    assert _Node.parent_reads == len(nodes)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(nodes)
    assert lines[-1].endswith("1499  (geom, id=1499)")


def test_tree_printing_keeps_source_sibling_order_and_detects_cycles(capsys):
    cli._print_tree([_Node(2, -1), _Node(1, 2), _Node(3, -1)])
    assert [line.split()[0] for line in capsys.readouterr().out.splitlines()] == ["2", "1", "3"]
    with pytest.raises(ValueError, match="cycle or duplicate"):
        cli._print_tree([_Node(0, -1), _Node(1, 0), _Node(0, 1)])


def test_assets_release_adapter_when_node_query_fails(monkeypatch, capsys):
    released = []

    def fail():
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr("mojive.scene.assets.list_assets", lambda: ["test"])
    monkeypatch.setattr("mojive.scene.assets.resolve", lambda name: name)
    monkeypatch.setattr(
        "mojive.application.backends.make_adapter",
        lambda *args: SimpleNamespace(nodes=fail, release=lambda: released.append(True)),
    )
    assert cli.main(["assets", "--json"]) == 0
    assert released == [True]
    assert json.loads(capsys.readouterr().out)["free_bodies"] == {"test": -1}


@pytest.mark.integration
def test_cli_pipe_executes_and_reads_back_one_real_edit(tmp_path):
    from mojive.adapters.static import StaticSceneAdapter
    from mojive.scene import Scene

    service = ControlService(StaticSceneAdapter(Scene()))
    server = ControlServer(tmp_path / "cli.sock", service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        args = [sys.executable, "-m", "mojive.cli", "control"]
        common = ["--socket", str(server.socket_path), "--json"]
        result = subprocess.run(
            [*args, "add_scene_object", "--params-file", "-", *common],
            input=json.dumps({"shape": "box", "name": "CLI box", "position": [1, 2, 3]}),
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        created = json.loads(result.stdout)
        assert created["ok"] and not result.stderr
        result = subprocess.run(
            [
                *args,
                "inspect_object",
                "--params",
                json.dumps({"object_id": created["object_id"]}),
                *common,
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        inspected = json.loads(result.stdout)
        assert inspected["name"] == "CLI box"
        assert inspected["position"] == [1, 2, 3]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        service.close()


def test_probe_uses_an_installed_module(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "call", lambda args: calls.append(args) or 0)
    assert cli.main(["probe"]) == 0
    assert calls == [[sys.executable, "-m", "mojive.tools.probe_gl"]]


def test_rpc_serve_releases_service_when_socket_creation_fails(monkeypatch):
    closed = []
    monkeypatch.setattr("mojive.cli.viewer._resolve", lambda value: value)
    monkeypatch.setattr("mojive.cli.control._resolve", lambda value: value)
    monkeypatch.setattr("mojive.application.backends.make_adapter", lambda *args: None)
    monkeypatch.setattr(
        "mojive.control.rpc.ControlService",
        lambda *args: SimpleNamespace(close=lambda: closed.append(True)),
    )

    def fail(*args):
        raise OSError("Socket already active")

    monkeypatch.setattr("mojive.control.rpc.ControlServer", fail)
    assert cli.main(["rpc-serve", "unused"]) == 2
    assert closed == [True]


@pytest.mark.parametrize("stage", ["publisher", "writer"])
def test_snapshot_serve_releases_acquired_resources_when_startup_fails(stage, monkeypatch):
    closed = []
    monkeypatch.setattr("mojive.cli.viewer._resolve", lambda value: value)
    monkeypatch.setattr("mojive.cli.control._resolve", lambda value: value)
    monkeypatch.setattr("mojive.application.backends.make_adapter", lambda *args: None)
    monkeypatch.setattr(
        "mojive.session.Session",
        lambda *args: SimpleNamespace(release=lambda: closed.append("session")),
    )

    def fail(*args):
        raise OSError("Unable to initialize " + stage)

    monkeypatch.setattr(
        "mojive.remote.SnapshotPublisher",
        fail
        if stage == "publisher"
        else lambda *args: SimpleNamespace(close=lambda: closed.append("publisher")),
    )
    monkeypatch.setattr("mojive.capture.recording.SnapshotWriter", fail)
    assert cli.main(["serve", "unused", "--record-snapshot", "unused.fvs"]) == 2
    assert closed == (["session"] if stage == "publisher" else ["publisher", "session"])


@pytest.mark.parametrize(
    "args",
    [
        ["capture", "unused", "-o", "unused.png", "--width", "0"],
        ["capture", "unused", "-o", "unused.png", "--width", "-1"],
        ["record", "unused", "-o", "unused.mp4", "--frames", "0"],
        ["record", "unused", "-o", "unused.mp4", "--fps", "NaN"],
        ["keyframes", "unused", "-o", "unused.mp4", "--fps", "inf"],
        ["doctor", "unused", "--frames", "-1"],
        ["replay", "unused", "--speed", "0"],
        ["serve", "unused", "--hz", "-1"],
        ["serve", "unused", "--hz", "abc"],
        ["record", "unused", "-o", "unused.mp4", "--frames", "1.5"],
    ],
)
def test_invalid_numeric_options_fail_before_loading(args, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_resolve", lambda _: pytest.fail("Invalid input loaded an asset"))
    assert cli.main(args) == 2
    assert "value must be" in capsys.readouterr().err


def test_capture_requires_both_dimensions_before_loading(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_resolve", lambda _: pytest.fail("Incomplete size loaded an asset"))
    assert cli.main(["capture", "unused", "-o", "unused.png", "--width", "640"]) == 2
    assert "width and height must be provided together" in capsys.readouterr().err


def test_audit_rejects_unsupported_adapter_before_loading(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_resolve", lambda _: pytest.fail("Wrong adapter loaded an asset"))
    assert cli.main(["audit", "unused", "--adapter", "toy", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_params"
