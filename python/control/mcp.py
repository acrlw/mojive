"""Optional MCP stdio transport for an existing Mojive RPC owner."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from functools import partial
from pathlib import Path

from mojive.control.rpc import DEFAULT_SOCKET, RpcClient, RpcError


def _result_schema(operation):
    schema = operation["output_schema"]
    if operation["name"] in {"capture", "capture_viewport"}:
        inline = deepcopy(schema)
        inline.pop("oneOf", None)
        inline["required"] = [*inline.get("required", ()), "image", "transport", "encoding"]
        inline["properties"].pop("data", None)
        inline["properties"].update(
            image={
                "type": "object",
                "properties": {"mime_type": {"const": "image/png"}},
                "required": ["mime_type"],
                "additionalProperties": False,
            },
            transport={"const": "base64"},
            encoding={"const": "png"},
        )
        schema = {"anyOf": [schema, inline]}
    return {
        "type": "object",
        "properties": {"result": schema},
        "required": ["result"],
        "additionalProperties": False,
    }


def create_server(client: RpcClient):
    """Expose the connected owner's operation catalog without duplicating scene state."""
    try:
        import anyio
        from mcp.server import Server
        from mcp.types import (
            CallToolResult,
            ImageContent,
            ListToolsResult,
            TextContent,
            Tool,
            ToolAnnotations,
        )
    except ImportError as error:
        raise RuntimeError(
            "MCP support requires the optional dependency: pip install 'mojive[mcp]'"
        ) from error

    async def list_tools(_context, _params):
        catalog = await anyio.to_thread.run_sync(partial(client.call, "describe_operations"))
        return ListToolsResult(
            tools=[
                Tool(
                    name=operation["name"],
                    description=operation["description"],
                    input_schema=operation["input_schema"],
                    output_schema=_result_schema(operation),
                    annotations=ToolAnnotations(
                        read_only_hint=not operation["mutates"]
                        and operation["name"] not in {"capture", "capture_viewport"},
                    ),
                )
                for operation in catalog["operations"]
            ]
        )

    async def call_tool(_context, params):
        try:
            result = await anyio.to_thread.run_sync(
                partial(client.call, params.name, params.arguments or {})
            )
        except RpcError as error:
            payload = {"error": error.payload()}
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
                structured_content=payload,
                is_error=True,
            )
        payload = {"result": result}
        content = []
        # Only the RPC response carries image bytes. A returned file path can
        # belong to a different host and must never become an implicit file read.
        if (
            params.name in {"capture", "capture_viewport"}
            and isinstance(result, dict)
            and result.get("transport") == "base64"
            and result.get("encoding") == "png"
        ):
            content.append(ImageContent(type="image", data=result["data"], mime_type="image/png"))
            payload = {
                "result": {
                    **{key: value for key, value in result.items() if key != "data"},
                    "image": {"mime_type": "image/png"},
                }
            }
        content.insert(0, TextContent(type="text", text=json.dumps(payload, ensure_ascii=False)))
        return CallToolResult(content=content, structured_content=payload)

    return Server(
        "mojive",
        version="0.1.0",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        instructions=(
            "Call hello and get_scene to identify the connected owner. Discover schemas and live "
            "availability with describe_operations. Inspect target IDs and send the returned document "
            "as expected_document for edits; refresh discovery after stale_document. Use edit_scene "
            "for atomic edits and undo for recovery. Capture with transport=base64 and encoding=png "
            "to receive an image. The result field contains the RPC result; PNG pixels appear only in "
            "the image content block and its result metadata contains image.mime_type. "
            "Tool failures retain error.code, message and details; do not blindly retry a timeout."
        ),
    )


def run_stdio(socket_path: Path = DEFAULT_SOCKET, *, timeout: float = 30.0) -> None:
    """Serve MCP on stdin/stdout while the existing RPC service owns all application work."""
    with RpcClient(socket_path, timeout=timeout) as client:
        server = create_server(client)
        import anyio
        from mcp.server.stdio import stdio_server

        async def serve():
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        anyio.run(serve)


def main() -> None:
    """Standalone entry point that does not import the viewer or rendering runtime."""
    parser = argparse.ArgumentParser(
        description="Expose an existing Mojive RPC service over MCP stdio"
    )
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be finite and positive")
    try:
        run_stdio(args.socket, timeout=args.timeout)
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
