"""Installed UI assets retain canonical geometry, font integrity and licenses."""

import hashlib
import json

import pytest

from mojive.tools.check_native_wheel import verify_package_resources


@pytest.fixture
def package(tmp_path):
    package = tmp_path / "installed/mojive"
    workspace = package / "ui/workspace"
    workspace.mkdir(parents=True)
    icons = {"camera": {"contours": []}, "light": {"contours": []}}
    source = json.dumps(icons).encode()
    (workspace / "icons.json").write_bytes(source)
    (workspace / "icon_meshes.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "icons": {name: [] for name in icons},
            }
        )
    )
    scene = package / "interaction/resources/scene_icons.json"
    scene.parent.mkdir(parents=True)
    scene.write_bytes(source)
    fonts = package / "text/resources/fonts"
    fonts.mkdir(parents=True)
    manifest = {}
    for name in ("MiSans-Regular.ttf", "MiSans-Semibold.ttf"):
        data = name.encode()
        (fonts / name).write_bytes(data)
        manifest[name] = hashlib.sha256(data).hexdigest()
    (fonts / "manifest.json").write_text(json.dumps(manifest))
    for path in (
        fonts / "MiSans-License.pdf",
        fonts / "README.md",
        workspace / "licenses/lucide.txt",
        workspace / "licenses/tabler.txt",
    ):
        path.parent.mkdir(exist_ok=True)
        path.write_text("fixture")
    return package


def test_installed_resources_are_verified_without_checkout_or_runtime_geometry(package):
    assert verify_package_resources(package) == {"icons": 2, "scene_icons": 2, "font_faces": 2}


@pytest.mark.parametrize("field,value", (("version", 2), ("source_sha256", "stale"), ("icons", {})))
def test_stale_or_incomplete_installed_meshes_are_rejected(package, field, value):
    path = package / "ui/workspace/icon_meshes.json"
    meshes = json.loads(path.read_text())
    meshes[field] = value
    path.write_text(json.dumps(meshes))
    with pytest.raises(ValueError, match="icon meshes"):
        verify_package_resources(package)


def test_installed_scene_subset_cannot_diverge_from_the_master(package):
    (package / "interaction/resources/scene_icons.json").write_text("{}")
    with pytest.raises(ValueError, match="scene icons"):
        verify_package_resources(package)


def test_installed_fonts_are_checked_against_their_distribution_manifest(package):
    (package / "text/resources/fonts/MiSans-Regular.ttf").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="font checksum"):
        verify_package_resources(package)


@pytest.mark.parametrize(
    "relative",
    (
        "ui/workspace/icon_meshes.json",
        "text/resources/fonts/MiSans-License.pdf",
        "ui/workspace/licenses/lucide.txt",
        "ui/workspace/licenses/tabler.txt",
    ),
)
def test_missing_installed_assets_fail_before_the_renderer_probe(package, relative):
    (package / relative).unlink()
    with pytest.raises(FileNotFoundError):
        verify_package_resources(package)
