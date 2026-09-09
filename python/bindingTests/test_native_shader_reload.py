"""Native source reload, visible output, and last-good-program recovery."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from mojive import Scene, SceneRenderer


def test_source_reload_preserves_live_scenes_and_recovers_after_errors(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    build = Path(os.environ["MOJIVE_NATIVE_BUILD"]).resolve()
    shaderc = build / ("mojive_shaderc.exe" if sys.platform == "win32" else "mojive_shaderc")
    cmake = Path(sys.executable).parent / ("cmake.exe" if sys.platform == "win32" else "cmake")
    source, scratch = tmp_path / "source", tmp_path / "build"
    shutil.copytree(root / "cpp/shaders", source / "shaders")
    shutil.copytree(build / "shaders", scratch / "shaders")
    platform, profile = {"darwin": ("osx", "metal"), "win32": ("windows", "s_5_0")}.get(
        sys.platform, ("linux", "spirv")
    )
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\nproject(Reload NONE)\n"
        "add_custom_target(mojive_probe_shaders\n"
        f'COMMAND "{shaderc.as_posix()}" -f "${{CMAKE_SOURCE_DIR}}/shaders/fs_lit.sc" '
        '-o "${CMAKE_BINARY_DIR}/shaders/fs_lit.bin" --type fragment '
        f"--platform {platform} -p {profile} "
        '--varyingdef "${CMAKE_SOURCE_DIR}/shaders/varying.def.sc" '
        f'-i "{(root / "thirdParty/bgfx/src").as_posix()}" -O 3 VERBATIM)\n'
    )
    ninja = cmake.with_name("ninja.exe" if sys.platform == "win32" else "ninja")
    subprocess.run(
        [
            str(cmake),
            "-S",
            str(source),
            "-B",
            str(scratch),
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.setenv("MOJIVE_NATIVE_SHADER_DIR", str(scratch / "shaders"))
    scene = Scene()
    scene.box(color=(0.9, 0.1, 0.25, 1))
    with SceneRenderer(scene.source, renderer="bgfx", width=80, height=60, samples=0) as renderer:
        renderer.update(scene.frame)
        backend = renderer._backend
        backend.enable_hot_reload()
        watcher = backend.device.shaders
        peer = backend.create_peer(80, 60)
        try:
            peer.set_scene(scene.source)
            peer.set_camera(backend._camera)
            peer.set_background((0, 0, 0, 1))
            peer.update(scene.frame)

            def reloaded():
                watcher._next_check = 0
                image = renderer.render()
                deadline = time.monotonic() + 20
                while watcher._build_future is not None:
                    assert time.monotonic() < deadline, "Shader rebuild did not complete"
                    time.sleep(0.01)
                    image = renderer.render()
                return image

            base = renderer.render()
            shader = source / "shaders/litFragment.sh"
            original = shader.read_text()
            assert "gl_FragColor=vec4(rgb,alpha);" in original
            shader.write_text(original.replace("vec4(rgb,alpha)", "vec4(rgb.bgr,alpha)"))
            watcher._next_check = 0
            changed = reloaded()
            np.testing.assert_array_equal(changed, base[..., ::-1])
            peer.render()
            np.testing.assert_array_equal(peer.target.read_rgb(), changed)
            assert not watcher.error
            ticket = backend.runtime.readback(backend.target.frame, backend.api.Product.COLOR)

            shader.write_text("invalid shader source")
            watcher._next_check = 0
            np.testing.assert_array_equal(reloaded(), changed)
            assert watcher.error

            shader.write_text(original)
            binary = scratch / "shaders/fs_outline.bin"
            original_binary = binary.read_bytes()
            binary.write_bytes(b"invalid binary")
            watcher._next_check = 0
            np.testing.assert_array_equal(reloaded(), changed)
            assert "shader" in watcher.error.lower()

            binary.write_bytes(original_binary)
            watcher._next_check = 0
            np.testing.assert_array_equal(reloaded(), base)
            peer.render()
            np.testing.assert_array_equal(peer.target.read_rgb(), base)
            np.testing.assert_array_equal(backend.runtime.wait(ticket).image, changed)
            assert not watcher.error
        finally:
            peer.release()
