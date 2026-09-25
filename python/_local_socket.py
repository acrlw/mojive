"""Local stream sockets, including Winsock AF_UNIX on Windows CPython.

Windows supports filesystem Unix sockets, but CPython does not expose their
address encoding. Keep Python's socket I/O and bridge only those Winsock calls.
"""

from __future__ import annotations

import os
import socket
import socketserver
import stat


def is_socket(info: os.stat_result) -> bool:
    return stat.S_ISSOCK(info.st_mode) or getattr(info, "st_reparse_tag", 0) == 0x80000023


def local_socket() -> socket.socket:
    if os.name == "nt":
        return _WindowsSocket()
    return socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)


if os.name == "nt":
    import ctypes
    import select
    import time

    class _Address(ctypes.Structure):
        _fields_ = [("family", ctypes.c_ushort), ("path", ctypes.c_char * 108)]

    _winsock = ctypes.WinDLL("Ws2_32.dll")
    for _name in ("bind", "connect", "getsockname", "accept"):
        _function = getattr(_winsock, _name)
        _function.argtypes = [
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p if _name in ("getsockname", "accept") else ctypes.c_int,
        ]
        _function.restype = ctypes.c_size_t if _name == "accept" else ctypes.c_int
    _winsock.WSAGetLastError.restype = ctypes.c_int

    def _address(path):
        encoded = os.fsencode(path)
        if b"\0" in encoded or len(encoded) >= 108:
            raise ValueError(
                "Local socket path must be shorter than 108 UTF-8 bytes and contain no NUL"
            )
        return _Address(1, encoded)

    def _error(code=None):
        return ctypes.WinError(_winsock.WSAGetLastError() if code is None else code)

    def _restrict_owner(path):
        # chmod only changes writability on Windows. Protect the socket's DACL
        # from inheritance and grant access solely to its owner before listen().
        api = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
        convert = api.ConvertStringSecurityDescriptorToSecurityDescriptorW
        convert.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        convert.restype = ctypes.c_int
        apply = api.SetFileSecurityW
        apply.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_void_p]
        apply.restype = ctypes.c_int
        free = ctypes.WinDLL("Kernel32.dll").LocalFree
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_void_p
        descriptor = ctypes.c_void_p()
        if not convert("D:P(A;;GA;;;OW)", 1, ctypes.byref(descriptor), None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not apply(os.fsdecode(path), 0x80000004, descriptor):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            free(descriptor)

    class _WindowsSocket(socket.socket):
        def __init__(self, *, fileno=None):
            super().__init__(1, socket.SOCK_STREAM, fileno=fileno)

        def bind(self, address):
            value = _address(address)
            if _winsock.bind(self.fileno(), ctypes.byref(value), ctypes.sizeof(value)):
                raise _error()
            _restrict_owner(address)

        def connect(self, address):
            value = _address(address)
            if not _winsock.connect(self.fileno(), ctypes.byref(value), ctypes.sizeof(value)):
                return
            code = _winsock.WSAGetLastError()
            if code != 10035 or self.gettimeout() == 0:
                raise _error(code)
            _, ready, failed = select.select([], [self], [self], self.gettimeout())
            if not ready and not failed:
                raise TimeoutError("Local socket connection timed out")
            code = self.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
            if code:
                raise _error(code)

        def accept(self):
            if self.gettimeout() is not None and self.gettimeout() > 0:
                ready, _, _ = select.select([self], [], [], self.gettimeout())
                if not ready:
                    raise TimeoutError("Local socket accept timed out")
            handle = _winsock.accept(self.fileno(), None, None)
            if handle == ctypes.c_size_t(-1).value:
                raise _error()
            peer = _WindowsSocket(fileno=handle)
            peer.setblocking(True)
            return peer, ""

        def getsockname(self):
            value = _Address()
            size = ctypes.c_int(ctypes.sizeof(value))
            if _winsock.getsockname(self.fileno(), ctypes.byref(value), ctypes.byref(size)):
                raise _error()
            return os.fsdecode(value.path)

    class LocalStreamServer(socketserver.TCPServer):
        address_family = 1

        def server_bind(self):
            self.socket.close()
            self.socket = local_socket()
            self.socket.bind(self.server_address)
            self.server_address = self.socket.getsockname()

        def shutdown_request(self, request):
            # Winsock can reset a socket with unread input and discard a queued
            # rejection response. Half-close first, then drain within a fixed
            # budget so a stalled peer cannot hold the accept loop indefinitely.
            try:
                request.shutdown(socket.SHUT_WR)
                deadline = time.monotonic() + 0.1
                while (remaining := deadline - time.monotonic()) > 0:
                    request.settimeout(remaining)
                    if not request.recv(65536):
                        break
            except OSError:
                pass
            finally:
                self.close_request(request)

else:
    LocalStreamServer = getattr(socketserver, "UnixStreamServer", socketserver.TCPServer)
