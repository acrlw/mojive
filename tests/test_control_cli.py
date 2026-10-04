"""CLI automation preserves structured errors and validates parameters before connecting."""

import json

import pytest

from mojive import cli
from mojive.control.rpc import RpcError


@pytest.mark.parametrize("params", ["[]", "false", "{broken"])
def test_json_parameter_errors_are_machine_readable(params, capsys):
    assert cli.main(["control", "get_scene", "--json", "--params", params]) == 2
    result = capsys.readouterr()
    assert json.loads(result.out)["error"]["code"] == "invalid_params"
    assert not result.err


def test_cli_keeps_server_error_details(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RpcError("stale_document", "Refresh scene", details={"actual": {"id": "current"}})

    monkeypatch.setattr("mojive.control_rpc.RpcClient.call", fail)
    assert cli.main(["control", "get_scene", "--json"]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "stale_document"
    assert error["details"]["actual"]["id"] == "current"


def test_invalid_timeout_is_a_structured_cli_error(capsys):
    assert cli.main(["control", "hello", "--json", "--timeout", "0"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_params"


def test_mcp_entry_uses_existing_socket_and_keeps_stdout_for_protocol(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        "mojive.control.mcp.run_stdio", lambda path, **kwargs: calls.append((path, kwargs))
    )
    assert cli.main(["mcp", "--socket", "output/existing.sock", "--timeout", "12"]) == 0
    assert calls == [("output/existing.sock", {"timeout": 12.0})]
    assert not capsys.readouterr().out
