"""Exchange caller-owned MuJoCo state only at explicit sync boundaries."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from mojive.adapters.mujoco.updates import camera_fields_changed


def _arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    # Most frames have no edits or NaNs; avoid constructing NaN masks on that path.
    return np.array_equal(left, right) or (
        left.dtype.kind in "fc" and np.array_equal(left, right, equal_nan=True)
    )


@dataclass
class _Field:
    caller: object
    display: object
    name: str
    previous: object
    path: str = ""

    def sync(self, publish: bool, selection: slice | None = None) -> bool:
        caller = getattr(self.caller, self.name)
        display = getattr(self.display, self.name)
        if isinstance(caller, np.ndarray):
            previous = self.previous
            if selection is not None:
                caller, display, previous = (
                    caller[selection],
                    display[selection],
                    previous[selection],
                )
            # Apply only UI edits, not the rest of a stale display snapshot.
            edited = not _arrays_equal(display, previous)
            if edited:
                mask = display != previous
                if display.dtype.kind in "fc":
                    mask &= ~(np.isnan(display) & np.isnan(previous))
                np.copyto(caller, display, where=mask)
            changed = publish and not _arrays_equal(caller, display)
            if changed:
                np.copyto(display, caller)
            if edited or changed:
                np.copyto(previous, display)
        else:
            if display != self.previous:
                setattr(self.caller, self.name, display)
                caller = display
            changed = publish and caller != display
            if changed:
                setattr(self.display, self.name, caller)
            self.previous = caller if publish else display
        return bool(changed)


class StructExchange:
    """Mirror writable MuJoCo fields, including nested option/visual structures."""

    def __init__(self, caller, display, *, exclude: tuple[str, ...] = (), prefix: str = ""):
        self.fields: list[_Field] = []
        self.changed_fields: set[str] = set()
        for name, descriptor in vars(type(caller)).items():
            if name in exclude or not isinstance(descriptor, property):
                continue
            value = getattr(caller, name)
            if descriptor.fset is not None and isinstance(value, (int, float, np.ndarray)):
                if not isinstance(value, np.ndarray) or value.size:
                    self.fields.append(
                        _Field(caller, display, name, copy.copy(value), prefix + name)
                    )
            elif type(value).__module__.startswith("mujoco"):
                self.fields.extend(
                    StructExchange(value, getattr(display, name), prefix=prefix + name + ".").fields
                )

    def sync(self, *, publish: bool = True) -> bool:
        self.changed_fields.clear()
        for field in self.fields:
            if field.sync(publish):
                self.changed_fields.add(field.path)
        return bool(self.changed_fields)

    def sync_arrays(self, selections: dict[str, slice]) -> None:
        for field in self.fields:
            if field.name in selections:
                field.sync(True, selections[field.name])


class StateExchange:
    """Keep rendering isolated from arbitrary caller writes between sync calls."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.display_model = copy.copy(model)
        self.display_data = mujoco.MjData(self.display_model)
        mujoco.mj_copyData(self.display_data, self.display_model, data)
        self.model_fields = StructExchange(model, self.display_model, exclude=("opt",))
        # Solver/physics options affect forward dynamics, not scene structure or GPU assets.
        self.physics_options = StructExchange(model.opt, self.display_model.opt)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        self.opt = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.display_cam = copy.copy(self.cam)
        self.display_opt = copy.copy(self.opt)
        self.display_perturb = copy.copy(self.perturb)
        self.camera_fields = StructExchange(self.cam, self.display_cam)
        self.option_fields = StructExchange(self.opt, self.display_opt)
        self.perturb_fields = StructExchange(self.perturb, self.display_perturb)
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        size = mujoco.mj_stateSize(model, self.state_spec)
        self.caller_state = np.empty(size)
        self.display_state = np.empty(size)
        self.previous_state = np.empty(size)
        mujoco.mj_getState(model, data, self.previous_state, self.state_spec)
        self.camera_changed = True
        self.options_changed = True
        self.pending_model_fields: set[str] = set()

    def sync_resource(self, category: str, index: int) -> None:
        """Publish one compiled resource's payload without publishing other model edits."""
        m, dm = self.model, self.display_model
        if category == "tex":
            layout = ("tex_adr", "tex_width", "tex_height", "tex_type")
            channels = int(m.tex_nchannel[index]) if hasattr(m, "tex_nchannel") else 3
            if hasattr(m, "tex_nchannel"):
                layout += ("tex_nchannel",)
            start = int(dm.tex_adr[index])
            end = start + int(dm.tex_width[index]) * int(dm.tex_height[index]) * channels
            selections = {"tex_data" if hasattr(m, "tex_data") else "tex_rgb": slice(start, end)}
        elif category == "hfield":
            layout = ("hfield_adr", "hfield_nrow", "hfield_ncol")
            start = int(dm.hfield_adr[index])
            end = start + int(dm.hfield_nrow[index]) * int(dm.hfield_ncol[index])
            selections = {"hfield_data": slice(start, end)}
        else:
            layout = ()
            selections = {}
            for name in ("vert", "normal", "texcoord", "face"):
                address, count = f"mesh_{name}adr", f"mesh_{name}num"
                layout += (address, count)
                start = int(getattr(dm, address)[index])
                stop = start + int(getattr(dm, count)[index])
                selections[f"mesh_{name}"] = slice(start, stop)
                if name == "face":
                    selections["mesh_facenormal"] = selections["mesh_facetexcoord"] = slice(
                        start, stop
                    )
        # Compiled allocation/offset changes cannot be uploaded safely into the private model.
        if any(getattr(m, name)[index] != getattr(dm, name)[index] for name in layout):
            raise ValueError("Resource layout changed; close and relaunch with the new model")
        self.model_fields.sync_arrays(selections)

    def sync(self, state_only: bool = False) -> None:
        if self.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
            if not 0 <= self.cam.fixedcamid < self.model.ncam:
                raise ValueError("cam.fixedcamid is outside the model camera range")
        elif self.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
            if not 0 <= self.cam.trackbodyid < self.model.nbody:
                raise ValueError("cam.trackbodyid is outside the model body range")
        elif self.cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
            raise NotImplementedError("Only free, fixed and tracking MjvCamera types are supported")
        self.model_fields.sync(publish=not state_only)
        self.physics_options.sync(publish=not state_only)
        self.pending_model_fields.update(self.model_fields.changed_fields)
        self.camera_changed |= camera_fields_changed(self.model_fields.changed_fields)
        self.camera_changed |= self.camera_fields.sync()
        self.options_changed |= self.option_fields.sync()
        self.perturb_fields.sync()
        m, d = self.model, self.data
        dm, dd = self.display_model, self.display_data
        mujoco.mj_getState(m, d, self.caller_state, self.state_spec)
        mujoco.mj_getState(dm, dd, self.display_state, self.state_spec)
        edited = self.display_state != self.previous_state
        np.copyto(self.caller_state, self.display_state, where=edited)
        mujoco.mj_setState(m, d, self.caller_state, self.state_spec)
        if self.perturb.active or self.perturb.active2:
            mujoco.mjv_applyPerturbPose(m, d, self.perturb, 0)
            mujoco.mjv_applyPerturbForce(m, d, self.perturb)
            mujoco.mj_getState(m, d, self.caller_state, self.state_spec)
        if not state_only:
            mujoco.mj_copyData(dd, dm, d)
        mujoco.mj_setState(dm, dd, self.caller_state, self.state_spec)
        mujoco.mj_forward(dm, dd)
        np.copyto(self.previous_state, self.caller_state)
