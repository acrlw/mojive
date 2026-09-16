"""Managed and passive launch signatures compatible with mujoco.viewer."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco

from .handle import Handle, _ui_owner
from .panels import install_key_callback, viewer_config


def _validate(model, data):
    if not isinstance(model, mujoco.MjModel):
        raise ValueError("model must be a mujoco.MjModel")
    if data is not None and not isinstance(data, mujoco.MjData):
        raise ValueError("data must be a mujoco.MjData")
    if data is not None and data.model is not model:
        raise ValueError("data was created for a different MuJoCo model")


def launch(
    model=None,
    data=None,
    *,
    loader=None,
    show_left_ui=True,
    show_right_ui=True,
    renderer=None,
    config=None,
) -> None:
    """Block until the full viewer closes, advancing the supplied model/data.

    Omit model/data for an empty editable workspace. ``loader`` returns a
    model/data pair. Mojive's extra renderer/config arguments are keyword-only.
    """
    from mojive.app.composition import build, build_editor

    if model is None and data is not None:
        raise ValueError("data requires a model")
    if loader is not None:
        if model is not None or data is not None:
            raise ValueError("Specify either model/data or loader")
        if not callable(loader):
            raise ValueError("loader must be callable")
        model, data = loader()
    config = viewer_config(config, show_left_ui, show_right_ui)
    if model is not None:
        _validate(model, data)
    with _ui_owner():
        if model is None:
            viewer = build_editor(renderer=renderer, config=config)
        else:
            viewer = build(
                model=model,
                data=data,
                external_clock=False,
                paused=False,
                renderer=renderer,
                config=config,
            )
        with viewer:
            install_key_callback(viewer, None)
            viewer.run()


def launch_from_path(path: str) -> None:
    """Launch a managed viewer for an XML/URDF or compiled MJB model."""
    from mojive.app.composition import build

    if Path(path).suffix.lower() == ".mjb":
        launch(mujoco.MjModel.from_binary_path(str(path)))
        return
    with _ui_owner(), build(path, paused=False) as viewer:
        install_key_callback(viewer, None)
        viewer.run()


def launch_passive(
    model,
    data,
    *,
    key_callback=None,
    show_left_ui=True,
    show_right_ui=True,
    renderer=None,
    config=None,
) -> Handle:
    """Return a MuJoCo-style handle without taking ownership of physics stepping.

    Linux and Windows use a UI thread and do not need a multiprocessing main
    guard. macOS currently requires the process-based ``mojive.launch_passive``
    extension because Mojive has no mjpython UI-thread dispatcher.
    """
    _validate(model, data)
    if data is None:
        raise ValueError("data must be a mujoco.MjData")
    if key_callback is not None and not callable(key_callback):
        raise ValueError("key_callback must be callable")
    config = viewer_config(config, show_left_ui, show_right_ui)
    if sys.platform == "darwin":
        raise NotImplementedError(
            "mojive.viewer.launch_passive has no macOS UI-thread dispatcher yet; "
            "use mojive.launch_passive from a guarded script for process-based viewing"
        )
    mujoco.mj_forward(model, data)
    return Handle(
        model,
        data,
        key_callback=key_callback,
        options={"renderer": renderer, "config": config, "vsync": False},
    )
