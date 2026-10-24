"""Read installed version and source revision once, outside the frame hot path."""

import subprocess
from contextlib import suppress
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def current_build() -> tuple[str, str | None]:
    from mojive import __version__

    root = Path(__file__).resolve().parents[1]
    revision = None
    # An installed wheel must not report the revision of an unrelated parent checkout.
    if (root / ".git").exists():
        with suppress(OSError, subprocess.SubprocessError):
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=1,
            ).strip()
    return __version__, revision
