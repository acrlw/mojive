"""Run native input acceptance inside an owned headless Weston 9 compositor."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(directory: Path) -> tuple[Path, Path]:
    """Extract Ubuntu 22.04's Weston without installing system packages."""
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / "root/usr"
    if not (prefix / "bin/weston").is_file():
        subprocess.run(
            ["apt-get", "download", "weston=9.0.0-4ubuntu1", "libweston-9-0=9.0.0-4ubuntu1"],
            cwd=directory,
            check=True,
            timeout=120,
        )
        for package in directory.glob("*.deb"):
            subprocess.run(["dpkg-deb", "-x", str(package), str(directory / "root")], check=True)
    source = directory / "weston-9.0.0"
    if not source.is_dir():
        archive = directory / "weston-9.0.0.tar.gz"
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--max-time",
                "60",
                "https://archive.ubuntu.com/ubuntu/pool/universe/w/weston/weston_9.0.0.orig.tar.gz",
                "--output",
                str(archive),
            ],
            check=True,
            timeout=65,
        )
        with tarfile.open(archive) as bundle:
            bundle.extractall(directory, filter="data")
    return prefix, source


def stop(process: subprocess.Popen, *, group: bool = False) -> None:
    if group:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare", action="store_true", help="Extract Weston 9 packages on Ubuntu 22.04"
    )
    parser.add_argument("--runtime", type=Path, default=ROOT / "output/wayland-runtime")
    parser.add_argument(
        "--weston-prefix", type=Path, help="Weston 9 installation prefix (usr directory)"
    )
    parser.add_argument("--weston-source", type=Path, help="Matching Weston 9 source headers")
    parser.add_argument("--output", type=Path, default=ROOT / "output/native-wayland")
    parser.add_argument("--renderer", choices=("bgfx", "opengl", "wgpu", "all"), default="all")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.weston_prefix and args.weston_source:
        prefix, source = args.weston_prefix.resolve(), args.weston_source.resolve()
    elif args.prepare:
        prefix, source = prepare(args.runtime.resolve())
    else:
        prefix, source = (
            args.runtime.resolve() / "root/usr",
            args.runtime.resolve() / "weston-9.0.0",
        )
    if not (prefix / "bin/weston").is_file() or not source.is_dir():
        parser.error(
            "Run with --prepare, or supply a Weston 9 prefix and matching source directory"
        )
    lib = next((prefix / "lib").glob("**/libweston-9.so.0")).parent
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        (str(lib / "weston"), str(lib), env.get("LD_LIBRARY_PATH", ""))
    )
    version = subprocess.check_output([str(prefix / "bin/weston"), "--version"], env=env, text=True)
    if version.strip() != "weston 9.0.0":
        parser.error(f"The private input module requires Weston 9.0.0, found {version.strip()}")
    module = output / "input.so"
    flags = shlex.split(
        subprocess.check_output(
            ["pkg-config", "--cflags", "pixman-1", "wayland-server", "xkbcommon"], text=True
        )
    )
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Werror",
            f"-I{source}/include",
            f"-I{source}/libweston",
            *flags,
            str(ROOT / "tools/wayland/input.c"),
            "-o",
            str(module),
        ],
        check=True,
    )
    # Set pyGLFW before MuJoCo can import its platform-only bundled library.
    import importlib.util

    bundle = Path(importlib.util.find_spec("imgui_bundle").origin).parent
    glfw = bundle / "libmojive_glfw.so.3"
    if not glfw.is_file():
        parser.error("Build the dual-platform ImGui dependency with make setup-imgui first")
    drag_source = output / "drag-source"
    client_flags = shlex.split(
        subprocess.check_output(["pkg-config", "--cflags", "--libs", "wayland-client"], text=True)
    )
    subprocess.run(
        [
            "cc",
            "-Wall",
            "-Werror",
            f"-I{ROOT}/thirdParty/glfw/include",
            str(ROOT / "tools/wayland/drag_source.c"),
            str(glfw),
            *client_flags,
            f"-Wl,-rpath,{bundle}",
            "-o",
            str(drag_source),
        ],
        check=True,
    )
    env["MOJIVE_TEST_DRAG_SOURCE"] = str(drag_source)
    env["PYGLFW_LIBRARY"] = str(glfw)
    env["PATH"] = str(prefix / "bin") + os.pathsep + env["PATH"]
    env["WESTON_MODULE_MAP"] = ";".join(f"{path.name}={path}" for path in lib.glob("*/*.so"))
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET", "PYOPENGL_PLATFORM", "MOJIVE_GL"):
        env.pop(name, None)
    env.update(
        XDG_SESSION_TYPE="wayland",
        MOJIVE_UI_SCALE="1",
        MOJIVE_LANGUAGE="en",
        MOJIVE_WAYLAND_OUTPUT=str(output),
    )
    env.setdefault("MOJIVE_NATIVE_BUILD", str(ROOT / "output/cpp-build"))
    renderers = ("bgfx", "opengl", "wgpu") if args.renderer == "all" else (args.renderer,)
    for renderer in renderers:
        with tempfile.TemporaryDirectory(prefix="mojive-wayland-") as runtime:
            # The controller is inherited only by this test process; there is no
            # input service or injection handle attached to the user's desktop.
            controller, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            env.update(
                XDG_RUNTIME_DIR=runtime,
                WAYLAND_DISPLAY="mojive-test",
                MOJIVE_TEST_COMPOSITOR_FD=str(server.fileno()),
                MOJIVE_TEST_INPUT_FD=str(controller.fileno()),
                MOJIVE_TEST_RENDERER=renderer,
                MOJIVE_CONFIG_DIR=runtime,
                MOJIVE_IMGUI_INI=str(Path(runtime) / "imgui.ini"),
                MOJIVE_SETTINGS=str(Path(runtime) / "settings.json"),
            )
            with (output / f"weston-{renderer}.log").open("w") as log:
                compositor = subprocess.Popen(
                    [
                        str(prefix / "bin/weston"),
                        "--no-config",
                        "--idle-time=0",
                        f"--backend={lib}/libweston-9/headless-backend.so",
                        "--use-gl",
                        f"--shell={lib}/weston/kiosk-shell.so",
                        f"--modules={module}",
                        "--socket=mojive-test",
                        "--width=1400",
                        "--height=1000",
                    ],
                    env=env,
                    pass_fds=(server.fileno(),),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                try:
                    for _ in range(100):
                        if compositor.poll() is not None:
                            raise RuntimeError(f"Weston exited: see {log.name}")
                        if Path(runtime, "mojive-test").exists():
                            break
                        time.sleep(0.05)
                    else:
                        raise RuntimeError("Weston did not create its private socket")
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "pytest",
                            "-q",
                            "python/bindingTests/test_wayland_input.py",
                            f"--junitxml={output}/{renderer}.xml",
                        ],
                        cwd=ROOT,
                        env=env,
                        pass_fds=(controller.fileno(),),
                        start_new_session=True,
                    )
                    try:
                        result = child.wait(timeout=args.timeout)
                        if result:
                            return result
                    finally:
                        stop(child, group=True)
                finally:
                    stop(compositor)
                    controller.close()
                    server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
