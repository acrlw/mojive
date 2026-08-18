"""Generate local control RPC capture and protocol acceptance artifacts."""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

from ..adapters.mujoco_adapter import MuJoCoAdapter
from ..control_rpc import ControlServer, ControlService, RpcClient


def main() -> None:
    output = Path("output/rpc")
    output.mkdir(parents=True, exist_ok=True)
    asset = Path("assets/joint_types.xml").resolve()
    service = ControlService(MuJoCoAdapter(asset), asset)
    result = {}
    with tempfile.TemporaryDirectory(prefix="fv-", dir="/tmp") as directory:
        server = ControlServer(Path(directory) / "control.sock", service)
        server.timeout = 0.1

        def run_client() -> None:
            client = RpcClient(server.socket_path)
            client.call("pause")
            objects = client.call("list_objects")
            selected = next(item for item in objects if item["object_id"])
            client.call("select_object", {"object_id": selected["object_id"]})
            captures = [
                client.call(
                    "capture",
                    {
                        "mode": mode,
                        "width": 640,
                        "height": 480,
                        "output": str(output / f"{mode}{'.png' if mode == 'rgb' else '.npy'}"),
                    },
                )
                for mode in ("rgb", "depth", "segmentation")
            ]
            result.update(
                asset=str(asset),
                selected=client.call("inspect_object", {"object_id": selected["object_id"]}),
                state=client.call("get_state"),
                captures=captures,
            )

        thread = threading.Thread(target=run_client, daemon=True)
        try:
            thread.start()
            while thread.is_alive():
                server.handle_request()
            thread.join()
            (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        finally:
            server.server_close()
            service.close()
    print(f"control RPC artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()
