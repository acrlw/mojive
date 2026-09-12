"""Cli: control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import _resolve, log


def cmd_rpc_serve(args: argparse.Namespace) -> int:
    from mojive.application.backends import make_adapter
    from mojive.control.rpc import ControlServer, ControlService

    path = _resolve(args.asset)
    service = ControlService(make_adapter(args.backend, path), path)
    try:
        server = ControlServer(Path(args.socket), service)
        log.info("Control RPC listening on {}", server.socket_path)
        try:
            server.serve_forever()
        finally:
            server.server_close()
    finally:
        service.close()
    return 0


def cmd_control(args: argparse.Namespace) -> int:
    from mojive.control.rpc import RpcClient, RpcError

    try:
        try:
            source = "{}" if args.params is None else args.params
            if args.params_file is not None:
                source = (
                    sys.stdin.read()
                    if args.params_file == "-"
                    else Path(args.params_file).expanduser().read_text(encoding="utf-8")
                )
            params = json.loads(source)
        except ValueError as exc:
            raise RpcError("invalid_params", f"Invalid parameter JSON: {exc}") from exc
        if not isinstance(params, dict):
            raise RpcError("invalid_params", "Parameters must be a JSON object")
        with RpcClient(Path(args.socket), args.timeout) as client:
            result = client.call(args.method, params)
    except (RpcError, ValueError) as exc:
        if not args.json:
            raise
        if not isinstance(exc, RpcError):
            exc = RpcError("invalid_params", str(exc))
        print(json.dumps({"error": exc.payload()}, indent=2))
        return 2
    if args.json:
        print(json.dumps(result, indent=2))
    elif isinstance(result, dict) and result.get("message"):
        print(result["message"])
    else:
        print(json.dumps(result, indent=2))
    return 0


def cmd_operations(args: argparse.Namespace) -> int:
    """Describe installed operation contracts without starting a viewer or service."""
    from mojive.control.errors import ControlError
    from mojive.control.operations import OPERATIONS

    if args.name is not None and args.name not in OPERATIONS:
        raise ControlError("unknown_method", f"Unknown control method: {args.name}")
    selected = [OPERATIONS[args.name]] if args.name else OPERATIONS.values()
    descriptions = [
        item.specification() for item in selected if args.scope is None or item.scope == args.scope
    ]
    if args.json:
        print(
            json.dumps(
                {
                    "schema_dialect": "https://json-schema.org/draft/2020-12/schema",
                    "operations": descriptions,
                },
                indent=2,
            )
        )
    elif args.name:
        print(json.dumps(descriptions, indent=2))
    else:
        for item in descriptions:
            action = "write" if item["mutates"] else "read"
            print(f"{item['name']:<28} {item['scope']:<8} {action:<5} {item['description']}")
        print("\nUse control describe_operations to check live availability and document identity.")
    return 0
