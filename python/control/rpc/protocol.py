from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from mojive.control.errors import ControlError as RpcError

PROTOCOL_VERSION = 1


def _default_socket() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        directory = Path(runtime) / "mojive"
    elif hasattr(os, "getuid"):
        directory = Path(tempfile.gettempdir()) / f"mojive-{os.getuid()}"
    else:
        directory = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "mojive"
    return directory / "control.sock"


DEFAULT_SOCKET = _default_socket()


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("RPC JSON numbers must be finite")
    return number


def _decode_json(payload):
    """Reject nonfinite constants and overflow before dispatch or correlation."""
    return json.loads(payload, parse_float=_finite_float, parse_constant=_finite_float)


def _validate_response(response, request_id: int) -> None:
    """Reject malformed envelopes before exposing their contents to callers."""
    if not isinstance(response, dict):
        raise RpcError("invalid_response", "RPC response must be a JSON object")
    if type(response.get("version")) is not int or response["version"] != PROTOCOL_VERSION:
        raise RpcError("invalid_response", "RPC response version is incompatible")
    transport_error = (
        response.get("id") is None
        and isinstance(response.get("error"), dict)
        and response["error"].get("code") in {"busy", "request_too_large"}
    )
    if not transport_error and (
        type(response.get("id")) is not int or response["id"] != request_id
    ):
        raise RpcError("invalid_response", "RPC response ID does not match the request")
    error = response.get("error")
    if error is not None:
        if (
            not isinstance(error, dict)
            or not isinstance(error.get("code"), str)
            or not isinstance(error.get("message"), str)
            or ("details" in error and not isinstance(error["details"], dict))
        ):
            raise RpcError("invalid_response", "RPC error must contain a code and message")
    elif "result" not in response:
        raise RpcError("invalid_response", "RPC response is missing its result")


def _request_deadline(request: dict[str, Any]) -> float | None:
    """Read an optional deadline on the host's shared monotonic clock."""
    value = request.get("deadline")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RpcError("invalid_request", "deadline must be finite monotonic seconds")
    return float(value)


def _response(request_id, *, result=None, error=None) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "id": request_id,
        "result": result if error is None else None,
        "error": error,
    }
