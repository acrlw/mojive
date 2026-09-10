"""Compile every native shader for Vulkan without requiring a Linux GPU."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("output/cpp-build"))
    args = parser.parse_args()
    output = args.build / "shaders-spirv"
    output.mkdir(parents=True, exist_ok=True)
    shaders = sorted(Path("cpp/shaders").glob("[vcf]s_*.sc"))
    for source in shaders:
        subprocess.run(
            [
                str(args.build / "mojive_shaderc"),
                "-f",
                str(source),
                "-o",
                str(output / (source.stem + ".bin")),
                "--type",
                {"vs": "vertex", "fs": "fragment", "cs": "compute"}[source.stem[:2]],
                "--platform",
                "linux",
                "-p",
                "spirv",
                "--varyingdef",
                "cpp/shaders/varying.def.sc",
                "-i",
                "3rdparty/bgfx/src",
                "-O",
                "3",
            ],
            check=True,
            capture_output=True,
        )
    print(f"Compiled {len(shaders)} Vulkan shaders: {output}")


if __name__ == "__main__":
    main()
