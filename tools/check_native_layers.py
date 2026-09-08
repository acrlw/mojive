"""Check that native contracts and common consumers stay independent of graphics vendors."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_INCLUDE = re.compile(
    r"#\s*include\s*[<\"](?:(?:bgfx|bx|bimg|GLFW|SDL3|nanobind|pybind11|mojive/backends)/|imgui[_.]|Python[.]h)"
)


def main() -> None:
    """Reject backend or UI dependencies in the common native boundary."""
    files = [
        *sorted((ROOT / "cpp/include/mojive").glob("*.hpp")),
        *sorted((ROOT / "cpp/src").glob("*.cpp")),
        *sorted((ROOT / "cpp/tests").glob("*.cpp")),
    ]
    errors = []
    for path in files:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            private_type = path.suffix == ".hpp" and re.search(
                r"#\s*include\s*[<\"](?:glm|spdlog)/", line
            )
            if VENDOR_INCLUDE.search(line) or private_type:
                errors.append(
                    f"{path.relative_to(ROOT)}:{number}: vendor dependency in common code"
                )
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Native layering passed: {len(files)} common headers, sources, and tests")


if __name__ == "__main__":
    main()
