"""Simulate rollout-boundary publication; viewers synchronize only on explicit request."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mojive.remote.rollout import RolloutServer, RolloutStore


def main() -> None:
    """Publish small selected CPU windows from an existing diagnostic joint archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--world-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--windows", type=int, default=120)
    parser.add_argument("--port", type=int, default=47651)
    args = parser.parse_args()
    if not 2 <= args.frames <= 512 or not 0.1 <= args.interval <= 3600 or args.windows < 1:
        parser.error("Use 2–512 frames, a 0.1–3600 second interval, and positive window count")
    metadata = json.loads((args.archive / "manifest.json").read_text())
    qpos = np.load(args.archive / "qpos.npy", mmap_mode="r", allow_pickle=False)
    ids = args.world_ids
    if (
        not 1 <= len(ids) <= 64
        or len(set(ids)) != len(ids)
        or any(i < 0 or i >= qpos.shape[1] for i in ids)
    ):
        parser.error("Choose 1–64 unique world IDs present in the archive")
    store = RolloutStore((args.archive / "model.mjb").read_bytes(), metadata)
    count = min(args.frames, len(qpos))
    try:
        with RolloutServer(store, port=args.port):
            for window in range(args.windows):
                first = (window * count) % (len(qpos) - count + 1)
                # In training, gather these worlds before copying tensors from GPU to CPU.
                store.publish(
                    qpos[first : first + count, ids, :],
                    start_step=window * count,
                    world_ids=ids,
                )
                print(f"Published window {window + 1}, source sample {window * count}", flush=True)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
