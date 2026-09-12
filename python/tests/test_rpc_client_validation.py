"""Malformed peer responses remain structured failures without implicit retries."""

import json

import pytest

from mojive.control.rpc import RpcClient, RpcError


class _Peer:
    def __init__(self, reply):
        self.reply = reply
        self.sent = []
        self.closed = False

    def sendall(self, data):
        self.sent.append(data)

    def settimeout(self, value):
        pass

    def recv(self, count):
        result, self.reply = self.reply, b""
        return result

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "reply",
    [
        b"[]\n",
        b"null\n",
        b"{broken\n",
        b"\xff\n",
        b'{"version":1,"id":1,"result":null}',
        b'{"version":true,"id":1,"result":null}\n',
        b'{"version":1,"id":true,"result":null}\n',
        b'{"version":1,"id":2,"result":null}\n',
        b'{"version":1,"id":1}\n',
        b'{"version":1,"id":1,"error":{}}\n',
        b'{"version":1,"id":1,"error":"broken"}\n',
        b'{"version":1,"id":1,"error":{"code":2,"message":"bad"}}\n',
        b'{"version":1,"id":1,"error":{"code":"bad","message":"bad","details":[]}}\n',
    ],
)
def test_invalid_envelope_closes_connection_and_next_call_recovers(reply, monkeypatch):
    peer = _Peer(reply)
    client = RpcClient()
    client._client = peer
    with pytest.raises(RpcError) as error:
        client.call("get_state")
    assert error.value.code == "invalid_response"
    assert peer.closed and client._client is None
    assert len(peer.sent) == 1
    recovered = _Peer(b'{"version":1,"id":2,"result":{"ok":true}}\n')
    client._client = recovered
    assert client.call("get_state") == {"ok": True}


@pytest.mark.parametrize(
    "params", [{"value": float("nan")}, {"value": float("inf")}, {"value": object()}]
)
def test_non_json_requests_fail_before_connect(params, monkeypatch):
    monkeypatch.setattr(
        RpcClient, "_connect", lambda self: pytest.fail("Invalid request connected")
    )
    with pytest.raises(RpcError) as error:
        RpcClient().call("set_ctrl", params)
    assert error.value.code == "invalid_params"


def test_valid_application_error_keeps_connection_and_details():
    reply = {
        "version": 1,
        "id": 1,
        "error": {"code": "stale_document", "message": "Refresh", "details": {"actual": "new"}},
    }
    peer = _Peer(json.dumps(reply).encode() + b"\n")
    client = RpcClient()
    client._client = peer
    with pytest.raises(RpcError) as error:
        client.call("edit_scene")
    assert error.value.code == "stale_document"
    assert error.value.details == {"actual": "new"}
    assert client._client is peer and not peer.closed
