"""Development installs and cleanup keep generated output out of runtime dependencies."""

import importlib.machinery
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.parametrize("native", (False, True))
def test_installed_editable_shim_survives_output_and_build_cleanup(tmp_path, native):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("Editable installation acceptance requires uv")
    project = tmp_path / "project"
    source = project / "python"
    (source / "tools").mkdir(parents=True)
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, project / name)
    shutil.copy2(ROOT / "python/tools/build_hook.py", source / "tools/build_hook.py")
    (source / "__init__.py").write_text('value = "source"\n')
    (source / "cli.py").write_text('value = "cli"\n')
    env = {key: value for key, value in os.environ.items() if not key.startswith("MOJIVE_NATIVE")}
    native_package = project / "build/native/python/mojive"
    if native:
        native_package.mkdir(parents=True)
        (native_package / ("_native" + importlib.machinery.EXTENSION_SUFFIXES[0])).touch()
        env["MOJIVE_NATIVE_EDITABLE_BUILD"] = str(project / "build/native")
    installed = tmp_path / "site-packages"
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(installed),
            "--no-deps",
            "-e",
            str(project),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert (installed / "mojive.py").is_file()
    assert not (project / "output").exists()
    assert not tuple((project / "build").glob("editable-*"))
    (project / "output").mkdir()
    (project / "output/capture.png").touch()
    shutil.rmtree(project / "output")
    shutil.rmtree(project / "build")
    # The installed entry must still find edited source, without importing the package at startup.
    (source / "__init__.py").write_text('value = "edited source"\n')
    script = """
import sys, json
assert 'mojive' not in sys.modules
sys.path.insert(0, sys.argv[1])
import mojive, mojive.cli
print(json.dumps([mojive.value, mojive.cli.value, mojive.__file__, mojive.__path__]))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(installed)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    value, cli, filename, paths = json.loads(result.stdout)
    assert (value, cli, filename) == ("edited source", "cli", str(source / "__init__.py"))
    assert paths == [str(source)] + ([str(native_package)] if native else [])


@pytest.mark.integration
@pytest.mark.parametrize(
    "target,removed",
    [
        ("clean", {".pytest_cache", ".ruff_cache"}),
        ("clean-output", {"output"}),
        ("clean-build", {"build"}),
    ],
)
def test_make_cleanup_only_removes_its_named_scope(tmp_path, target, removed):
    if shutil.which("make") is None:
        pytest.skip("Makefile cleanup requires GNU Make")
    names = {".pytest_cache", ".ruff_cache", "output", "build", ".venv", "python", "out"}
    for name in names:
        (tmp_path / name).mkdir()
        (tmp_path / name / "sentinel").write_text(name)
    subprocess.run(["make", "-f", str(ROOT / "Makefile"), target], cwd=tmp_path, check=True)
    assert {name for name in names if not (tmp_path / name).exists()} == removed
    for name in names - removed:
        assert (tmp_path / name / "sentinel").read_text() == name
