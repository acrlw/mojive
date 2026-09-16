"""Classify presentation roles without changing MuJoCo collision behavior."""

import numpy as np

from mojive.types import GeometryRole

from .engine import mujoco
from .spec import _compiled_text_names


def geometry_roles(model) -> tuple[np.ndarray, np.ndarray]:
    """Return role bits and collision candidates, including explicit contact pairs.

    A collidable geom is shared unless its body has a separate
    non-colliding appearance. World geometry stays shared: unrelated decorations
    attached to the world must not hide floors. Custom text lists override this
    presentation inference without modifying collision masks.
    """
    collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    collision[model.pair_geom1] = True
    collision[model.pair_geom2] = True
    welded = model.body_weldid[model.geom_bodyid]
    # Fixed bodies can have their own visual/proxy pairs despite being welded
    # to the world. Only world-attached geoms share its global namespace.
    welded = np.where(welded == 0, model.geom_bodyid, welded)
    has_visual = np.zeros(model.nbody, bool)
    has_visual[welded[~collision]] = True
    shared = collision & ((model.geom_bodyid == 0) | ~has_visual[welded])
    roles = np.where(collision, int(GeometryRole.COLLISION), int(GeometryRole.VISUAL)).astype(
        np.uint8
    )
    roles[shared] = int(GeometryRole.BOTH)
    assigned: set[int] = set()
    for name, role in (
        ("mojive_visual_geoms", GeometryRole.VISUAL),
        ("mojive_collision_geoms", GeometryRole.COLLISION),
        ("mojive_shared_geoms", GeometryRole.BOTH),
    ):
        for prefix, names in _compiled_text_names(model, name, strict=True):
            for raw in names:
                geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + raw)
                if geom < 0:
                    raise ValueError(f"{name} references unknown geom {prefix + raw!r}")
                if geom in assigned:
                    raise ValueError(f"Geom {prefix + raw!r} has conflicting presentation roles")
                assigned.add(geom)
                roles[geom] = int(role)
    return roles, collision
