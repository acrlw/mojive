"""Control a running forge-viewer RPC service from Python."""

from __future__ import annotations

import argparse
from pathlib import Path

from forge_viewer.control_rpc import DEFAULT_SOCKET, RpcClient


def parse_args() -> argparse.Namespace:
    """Parse the RPC endpoint and requested action."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--capture", type=Path)
    return parser.parse_args()


def main() -> None:
    """Pause, advance, inspect, and optionally capture the remote scene."""
    args = parse_args()
    with RpcClient(args.socket) as client:
        client.call("pause")
        client.call("step", {"count": args.steps})
        state = client.call("get_state")
        print(f"asset={state['asset']}")
        print(f"paused={state['paused']} time={state['physics']['time']:.6f}")
        if args.capture is not None:
            result = client.call("capture", {"path": str(args.capture)})
            print(result["path"])


if __name__ == "__main__":
    main()
