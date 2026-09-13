"""Expose the installed package name to static API documentation analysis."""

from pathlib import Path

from mkdocs.plugins import event_priority


@event_priority(100)
def on_config(config):
    """Map source files without copying them or importing the rendering stack."""
    root = Path(config.config_file_path).resolve().parent
    source = root / "python"
    package = root / "output/docs_api/mojive"
    package.parent.mkdir(parents=True, exist_ok=True)
    if package.is_symlink() and package.resolve() != source:
        package.unlink()
    if not package.exists():
        package.symlink_to(source, target_is_directory=True)
    elif not package.is_symlink():
        raise RuntimeError(f"Expected generated API source link at {package}")
