"""Structured application failures shared by local and remote control transports."""

from __future__ import annotations

from typing import Any


class ControlError(RuntimeError):
    """An operation failure with a stable code and optional machine-readable details."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def payload(self) -> dict[str, Any]:
        """Return the serializable failure shared by Python, CLI, and RPC clients."""
        result = {"code": self.code, "message": str(self)}
        if self.details:
            result["details"] = self.details
        return result
