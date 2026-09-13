from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from mojive.control.errors import ControlError as RpcError

PROTOCOL_VERSION = 1
DEFAULT_SOCKET = Path("output/mojive.sock")


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
