"""World-wide MuJoCo options, edited between steps without rebuilding the model."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from functools import lru_cache

import numpy as np

from ..base import PhysicsOption
from .engine import mujoco

# Keep schema discovery at the engine boundary: older MuJoCo versions simply
# omit options and flags they do not implement.
_FIELDS = (
    (
        "Algorithms",
        (
            ("integrator", "Integrator"),
            ("cone", "Friction cone"),
            ("jacobian", "Jacobian"),
            ("solver", "Solver"),
        ),
    ),
    (
        "Algorithmic parameters",
        (
            ("timestep", "Timestep (s)"),
            ("iterations", "Iterations"),
            ("tolerance", "Tolerance"),
            ("ls_iterations", "Line search iterations"),
            ("ls_tolerance", "Line search tolerance"),
            ("noslip_iterations", "Noslip iterations"),
            ("noslip_tolerance", "Noslip tolerance"),
            ("ccd_iterations", "CCD iterations"),
            ("ccd_tolerance", "CCD tolerance"),
            ("sleep_tolerance", "Sleep tolerance"),
            ("sdf_iterations", "SDF iterations"),
            ("sdf_initpoints", "SDF initial points"),
        ),
    ),
    (
        "Physical parameters",
        (
            ("gravity", "Gravity"),
            ("wind", "Wind"),
            ("magnetic", "Magnetic field"),
            ("density", "Density"),
            ("viscosity", "Viscosity"),
            ("impratio", "Impedance ratio"),
            ("disableactuator", "Disabled actuator groups"),
        ),
    ),
    (
        "Contact override",
        (
            ("o_margin", "Margin"),
            ("o_solref", "Solver reference"),
            ("o_solimp", "Solver impedance"),
            ("o_friction", "Friction"),
        ),
    ),
)
_GROUP_DESCRIPTIONS = {
    "Disable flags": "Checked flags disable these features.",
    "Enable flags": "Checked flags enable these features.",
    "Contact override": "Used when Contact override is enabled under Enable flags.",
}
_ENUMS = {
    "integrator": "mjtIntegrator",
    "cone": "mjtCone",
    "jacobian": "mjtJacobian",
    "solver": "mjtSolver",
}
_LABELS = {
    "EULER": "Euler",
    "RK4": "RK4",
    "IMPLICIT": "Implicit",
    "IMPLICITFAST": "Implicit fast",
    "CG": "CG",
    "PGS": "PGS",
    "FRICTIONLOSS": "Friction loss",
    "CLAMPCTRL": "Control clamping",
    "WARMSTART": "Warm start",
    "FILTERPARENT": "Parent collision filtering",
    "REFSAFE": "Reference safety",
    "EULERDAMP": "Euler damping",
    "AUTORESET": "Automatic reset",
    "NATIVECCD": "Native CCD",
    "MULTICCD": "Multiple CCD",
    "OVERRIDE": "Contact override",
    "FWDINV": "Forward/inverse comparison",
    "INVDISCRETE": "Discrete inverse dynamics",
    "DIAGEXACT": "Exact solver diagonal",
}


def _value(option, name):
    value = getattr(option, name)
    return tuple(float(v) for v in value) if isinstance(value, np.ndarray) else value


def _assign(option, name, value):
    current = getattr(option, name)
    if isinstance(current, np.ndarray):
        current[:] = value
    else:
        setattr(option, name, value)


@lru_cache(maxsize=1)
def _schema():
    default = mujoco.MjOption()
    fields = []
    for group, entries in _FIELDS:
        for key, label in entries:
            if not hasattr(default, key):
                continue
            value = _value(default, key)
            kind = (
                "vector"
                if isinstance(value, tuple)
                else "int"
                if isinstance(value, int)
                else "float"
            )
            choices = ()
            if key in _ENUMS:
                kind = "choice"
                choices = tuple(
                    (_LABELS.get(name.split("_", 1)[1], name.split("_", 1)[1].title()), int(value))
                    for name, value in getattr(mujoco, _ENUMS[key]).__members__.items()
                )
            description = (
                "Bit mask of disabled actuator groups: 1 disables group 0, 2 disables group 1, 3 disables both."
                if key == "disableactuator"
                else ""
            )
            fields.append(
                PhysicsOption(
                    key,
                    label,
                    group,
                    kind,
                    value,
                    choices,
                    description,
                    _GROUP_DESCRIPTIONS.get(group, ""),
                )
            )
    bits = {}
    for field, group, enum, prefix in (
        ("disableflags", "Disable flags", mujoco.mjtDisableBit, "mjDSBL_"),
        ("enableflags", "Enable flags", mujoco.mjtEnableBit, "mjENBL_"),
    ):
        for name, bit in enum.__members__.items():
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            key = f"{field}.{suffix.lower()}"
            bits[key] = (field, int(bit))
            fields.append(
                PhysicsOption(
                    key,
                    _LABELS.get(suffix, suffix.title()),
                    group,
                    "bool",
                    False,
                    group_description=_GROUP_DESCRIPTIONS[group],
                )
            )
    return tuple(fields), bits


def _validated_value(field, value):
    if field.kind == "bool":
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{field.label} requires a boolean")
        return bool(value)
    array = np.asarray(value, dtype=np.float64)
    shape = (len(field.value),) if field.kind == "vector" else ()
    if array.shape != shape or not np.isfinite(array).all() or isinstance(value, bool):
        raise ValueError(f"{field.label} requires finite values with shape {shape}")
    if field.kind in ("int", "choice"):
        if float(array) != int(array) or not 0 <= int(array) <= 2**31 - 1:
            raise ValueError(f"{field.label} requires a non-negative 32-bit integer")
        value = int(array)
        if field.choices and value not in {choice for _, choice in field.choices}:
            raise ValueError(f"Unknown {field.label} value: {value}")
    else:
        value = tuple(float(v) for v in array) if shape else float(array)
    if (
        field.kind == "float"
        and field.key != "o_margin"
        and (value < 0 or (field.key in ("timestep", "impratio") and value == 0))
    ):
        raise ValueError(
            f"{field.label} must be {'positive' if field.key in ('timestep', 'impratio') else 'non-negative'}"
        )
    if field.key == "o_friction" and min(value) < 0:
        raise ValueError("Contact friction must be non-negative")
    if field.key == "o_solref" and not (min(value) > 0 or max(value) <= 0):
        raise ValueError("Solver reference requires two positive values or two non-positive values")
    if field.key == "o_solimp" and not (
        0 <= value[0] <= 1
        and 0 <= value[1] <= 1
        and value[2] >= 0
        and 0 <= value[3] <= 1
        and value[4] >= 1
    ):
        raise ValueError(
            "Solver impedance requires dmin/dmax in [0,1], width >= 0, midpoint in [0,1], power >= 1"
        )
    return value


class _PhysicsOptions:
    """Private option editing; the adapter remains the sole model and data owner."""

    def physics_options(self) -> tuple[PhysicsOption, ...]:
        fields, bits = _schema()
        values = tuple(
            bool(getattr(self._m.opt, bits[f.key][0]) & bits[f.key][1])
            if f.key in bits
            else _value(self._m.opt, f.key)
            for f in fields
        )
        if self._physics_options_cache is None or values != self._physics_options_cache[0]:
            self._physics_options_cache = (
                values,
                tuple(replace(f, value=v) for f, v in zip(fields, values, strict=True)),
            )
        return self._physics_options_cache[1]

    def set_physics_options(self, values: dict[str, object]) -> bool:
        if not self.caps.supports("physics.options"):
            return False
        fields, bits = _schema()
        schema = {f.key: f for f in fields}
        candidate = copy(self._m.opt)
        for key, raw in values.items():
            if key not in schema:
                raise ValueError(f"Unknown physics option: {key}")
            value = _validated_value(schema[key], raw)
            if key in bits:
                name, bit = bits[key]
                old = getattr(candidate, name)
                setattr(candidate, name, (old | bit) if value else (old & ~bit))
            else:
                _assign(candidate, key, value)
        names = (*(f.key for f in fields if f.key not in bits), "disableflags", "enableflags")
        previous = {name: _value(self._m.opt, name) for name in names}
        authored = (
            None
            if self._root_spec is None
            else {name: _value(self._root_spec.option, name) for name in names}
        )
        saved_data = mujoco.MjData(self._m)
        mujoco.mj_copyData(saved_data, self._m, self._d)
        try:
            for name in names:
                _assign(self._m.opt, name, _value(candidate, name))
            # Refresh accelerations, contacts and diagnostics without advancing time
            # or replacing the model/data objects used by the simulation worker.
            mujoco.mj_forward(self._m, self._d)
            if self._root_spec is not None:
                for name in names:
                    _assign(self._root_spec.option, name, _value(candidate, name))
        except Exception:
            for name, value in previous.items():
                _assign(self._m.opt, name, value)
            if authored is not None:
                for name, value in authored.items():
                    _assign(self._root_spec.option, name, value)
            mujoco.mj_copyData(self._d, self._m, saved_data)
            raise
        if self._root_spec is not None:
            self._root_options_explicit = True
            self._root_edited = True
        self._physics_options_cache = None
        return True
