"""Local protocol, budgets and lazily imported client/server implementations."""

from importlib import import_module
from typing import TYPE_CHECKING

from mojive.control.contracts import RpcLimits
from mojive.control.errors import ControlError as RpcError

from .protocol import DEFAULT_SOCKET, PROTOCOL_VERSION

if TYPE_CHECKING:
    from .client import RpcClient as RpcClient
    from .server import ControlServer as ControlServer
    from .service import ControlService as ControlService
    from .viewer import ViewerControlService as ViewerControlService
    from .viewer import ViewerRpcServer as ViewerRpcServer

_EXPORTS = {
    "RpcClient": "client",
    "ControlServer": "server",
    "ControlService": "service",
    "ViewerControlService": "viewer",
    "ViewerRpcServer": "viewer",
}
__all__ = ["DEFAULT_SOCKET", "PROTOCOL_VERSION", "RpcError", "RpcLimits", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
