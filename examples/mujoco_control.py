"""Control and step a MuJoCo model through a Mojive session."""

from __future__ import annotations

import argparse
from pathlib import Path

from mojive import FrameNeeds, MuJoCoAdapter
from mojive import commands as cmd
from mojive.session import Session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--qpos-index", type=int)
    parser.add_argument("--qpos", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.model.expanduser().resolve()
    session = Session(MuJoCoAdapter(path), path)
    try:
        session.submit(cmd.Pause())
        if args.qpos_index is not None and args.qpos is not None:
            result = session.submit(cmd.SetQpos(args.qpos_index, args.qpos))
            if not result:
                raise RuntimeError(result.message)
        session.submit(cmd.Step(args.steps))
        frame = session.tick(FrameNeeds(qpos=True, qvel=True, poses=True))
        print(f"time={frame.time:.6f} step={frame.step}")
        print(f"qpos={frame.qpos}")
        print(f"qvel={frame.qvel}")
    finally:
        session.release()


if __name__ == "__main__":
    main()
