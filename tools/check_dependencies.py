"""Verify repository-managed dependency revisions and the optional ImGui baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Check pinned submodules; optionally require an unmodified upstream ImGui tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgui-baseline", action="store_true")
    args = parser.parse_args()
    lock = json.loads((ROOT / "thirdParty/dependencies.json").read_text())
    errors = []
    checked = 0
    for name, entry in lock.items():
        if entry.get("source") != "submodule":
            continue
        path = ROOT / entry["path"]
        if not (path / ".git").exists():
            errors.append(f"{name}: run git submodule update --init --depth 1")
            continue
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True)
        if head.strip() != entry["commit"]:
            errors.append(f"{name}: checkout differs from the dependency lock")
        changes = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True
        )
        if changes.strip():
            errors.append(f"{name}: uncommitted upstream source changes")
        checked += 1
    if args.imgui_baseline:
        baseline = json.loads((ROOT / "thirdParty/imguiBaseline.json").read_text())
        path = ROOT / lock["imgui"]["path"]
        actual = {
            p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*")
            if p.is_file()
        }
        for name in sorted(set(baseline) | set(actual)):
            if baseline.get(name) != actual.get(name):
                errors.append(f"imgui: upstream baseline differs at {name}")
        print(f"ImGui baseline: {len(baseline)} files")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Dependency revisions passed: {checked} pinned submodules")


if __name__ == "__main__":
    main()
