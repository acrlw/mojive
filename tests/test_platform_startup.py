"""Desktop startup must not require Unix-only RPC facilities."""

import os
import subprocess
import sys

import pytest


def test_cli_import_without_unix_facilities():
    script = """
import os, socket, socketserver
for module, name in ((os, 'getuid'), (socket, 'AF_UNIX'),
                     (socketserver, 'UnixStreamServer')):
    if hasattr(module, name):
        delattr(module, name)
from mojive.cli import main
raise SystemExit(main(['--help']))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "doctor" in result.stdout


def test_windows_runtime_directory_is_per_user(tmp_path, monkeypatch):
    from mojive.control.rpc.protocol import _default_socket

    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delattr(os, "getuid", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert _default_socket() == tmp_path / "mojive" / "control.sock"


@pytest.mark.parametrize("name", ["local.sock", "\u6d4b.sock"])
def test_local_socket_echo_and_timeout(tmp_path, name):
    from mojive._local_socket import is_socket, local_socket

    path = tmp_path / name
    with local_socket() as listener:
        listener.bind(str(path))
        listener.listen(1)
        listener.settimeout(0.05)
        assert is_socket(path.lstat())
        with pytest.raises(TimeoutError):
            listener.accept()
        with local_socket() as sender:
            sender.settimeout(1)
            sender.connect(str(path))
            peer, _ = listener.accept()
            with peer:
                peer.settimeout(1)
                sender.sendall(b"hello")
                assert peer.recv(5) == b"hello"
                peer.sendall(b"reply")
                assert sender.recv(5) == b"reply"
    path.unlink()


def test_process_discovery_does_not_terminate_the_current_process():
    from mojive.remote.bridge import _alive

    assert _alive(os.getpid())
