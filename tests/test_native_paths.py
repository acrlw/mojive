from types import SimpleNamespace

from mojive.native import native_shader_directory


def test_native_shaders_follow_wheel_and_editable_extension(tmp_path, monkeypatch):
    monkeypatch.delenv("MOJIVE_NATIVE_BUILD", raising=False)
    monkeypatch.delenv("MOJIVE_NATIVE_SHADER_DIR", raising=False)
    package = tmp_path / "python" / "mojive"
    package.mkdir(parents=True)
    module = SimpleNamespace(__file__=str(package / "_native.so"))
    assert native_shader_directory(module) == package / "shaders"
    (tmp_path / "CMakeCache.txt").write_text("CMAKE_PROJECT_NAME:STATIC=mojive\n")
    assert native_shader_directory(module) == tmp_path / "shaders"


def test_explicit_native_shader_paths_take_precedence(tmp_path, monkeypatch):
    module = SimpleNamespace(__file__=str(tmp_path / "mojive" / "_native.so"))
    monkeypatch.setenv("MOJIVE_NATIVE_BUILD", str(tmp_path / "build"))
    monkeypatch.delenv("MOJIVE_NATIVE_SHADER_DIR", raising=False)
    assert native_shader_directory(module) == tmp_path / "build" / "shaders"
    monkeypatch.setenv("MOJIVE_NATIVE_SHADER_DIR", str(tmp_path / "custom-shaders"))
    assert native_shader_directory(module) == tmp_path / "custom-shaders"
