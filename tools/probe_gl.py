"""Source-checkout entry point for the installed OpenGL probe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

from mojive.tools.probe_gl import main

if __name__ == "__main__":
    raise SystemExit(main())
