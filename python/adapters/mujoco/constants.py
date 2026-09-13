"""MuJoCo visual categories and native data encodings."""

from __future__ import annotations

import numpy as np

DEFAULT_GEOM_GROUPS: tuple[int, ...] = (0, 1, 2)


VISUAL_GROUP_CATEGORIES = ("geom", "site", "joint", "tendon", "actuator", "flex", "skin")


_GEOM_RGBA_DEFAULT = np.array([0.5, 0.5, 0.5, 1.0], np.float32)


_TEXROLE_RGB = 1


_TEXROLE_RGBA = 8


_MOJIVE_AMBIENT_NUMERIC = "mojive.environment.ambient"


_MOJIVE_HAZE_NUMERIC = "mojive.environment.horizon_haze"


_MOJIVE_AREA_LIGHTS_TEXT = "mojive.light.area"


_URDF_MIN_POSITIVE_INERTIA = 1e-14


_BVH_POSE_BODY = 0


_BVH_POSE_GEOM = 1


_BVH_POSE_DYNAMIC = 2


_ACTUATOR_POSE_JOINT_AXIS = 0


_ACTUATOR_POSE_JOINT_BODY = 1


_ACTUATOR_POSE_SITE = 2


_ACTUATOR_POSE_GEOM = 3
