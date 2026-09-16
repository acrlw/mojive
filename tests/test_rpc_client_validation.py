"""Malformed peer responses remain structured failures without implicit retries."""

import json
from concurrent.futures import ThreadPoolExecutor

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
        b'{"version":1,"id":1,"result":NaN}\n',
        b'{"version":1,"id":1,"result":Infinity}\n',
        b'{"version":1,"id":1,"result":1e999}\n',
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


def test_timeout_includes_waiting_for_shared_client_without_closing_active_connection():
    peer = _Peer(b'{"version":1,"id":1,"result":{"ok":true}}\n')
    client = RpcClient(timeout=0.02)
    client._client = peer
    client._lock.acquire()
    with ThreadPoolExecutor(max_workers=1) as worker:
        pending = worker.submit(client.hello)
        try:
            with pytest.raises(RpcError, match="not sent") as error:
                pending.result(timeout=0.5)
            assert error.value.code == "timeout"
            assert not peer.sent and not peer.closed
        finally:
            client._lock.release()
    assert client.hello() == {"ok": True}
    assert len(peer.sent) == 1


def test_connect_and_write_share_the_request_deadline(monkeypatch):
    import mojive.control.rpc.client as module

    clock = [10.0]
    timeouts = []
    peer = _Peer(b'{"version":1,"id":1,"result":{"ok":true}}\n')
    peer.connect = lambda _address: clock.__setitem__(0, clock[0] + 0.4)
    peer.settimeout = timeouts.append
    send = peer.sendall
    peer.sendall = lambda data: (timeouts.append(("send", timeouts[-1])), send(data))
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.socket, "socket", lambda *_args: peer)
    assert RpcClient(timeout=1).hello() == {"ok": True}
    assert next(item[1] for item in timeouts if isinstance(item, tuple)) == pytest.approx(0.6)
