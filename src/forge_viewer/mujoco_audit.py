"""MuJoCo model coverage audit for forge's current scene contract."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from .adapters.mujoco_adapter import DEFAULT_GEOM_GROUPS, mujoco


@dataclass(frozen=True)
class Finding:
    feature: str
    status: str
    count: int
    detail: str


@dataclass(frozen=True)
class Coverage:
    feature: str
    status: str
    detail: str


_RND_COVERAGE = (
    Coverage("mjRND_SHADOW", "supported", "forge shadow passes"),
    Coverage("mjRND_WIREFRAME", "supported", "render flag and wireframe debug view"),
    Coverage("mjRND_REFLECTION", "supported", "planar reflections"),
    Coverage("mjRND_ADDITIVE", "unsupported", "no additive visualization pass"),
    Coverage("mjRND_SKYBOX", "supported", "skybox texture pass"),
    Coverage("mjRND_FOG", "supported", "independent linear fog flag"),
    Coverage("mjRND_HAZE", "supported", "independent exponential haze flag"),
    Coverage("mjRND_DEPTH", "supported", "depth debug view"),
    Coverage("mjRND_SEGMENT", "supported", "R32UI segment debug view"),
    Coverage("mjRND_IDCOLOR", "supported", "R32UI id-color debug view"),
    Coverage("mjRND_CULL_FACE", "supported", "render flag"),
)

_VIS_COVERAGE = (
    Coverage("mjVIS_CONVEXHULL", "unsupported", "no separate convex-hull overlay"),
    Coverage("mjVIS_TEXTURE", "supported", "2D color textures and skyboxes"),
    Coverage("mjVIS_JOINT", "supported", "solid free, ball, slide and hinge markers"),
    Coverage(
        "mjVIS_CAMERA",
        "degraded",
        "named cameras drive the viewport; camera icons are not drawn",
    ),
    Coverage(
        "mjVIS_ACTUATOR",
        "degraded",
        "spatial-tendon transmission paths; joint/site/body icons are not drawn",
    ),
    Coverage(
        "mjVIS_ACTIVATION",
        "degraded",
        "MuJoCo range and color interpolation on spatial-tendon actuator paths",
    ),
    Coverage(
        "mjVIS_LIGHT",
        "degraded",
        "lights are editable Forge entities; viewport icons are not drawn",
    ),
    Coverage("mjVIS_TENDON", "supported", "dynamic tendon paths"),
    Coverage("mjVIS_RANGEFINDER", "unsupported", "rangefinder rays are not drawn"),
    Coverage("mjVIS_CONSTRAINT", "unsupported", "constraint overlays are not drawn"),
    Coverage("mjVIS_INERTIA", "supported", "body inertia boxes"),
    Coverage("mjVIS_SCLINERTIA", "supported", "constant-density inertia boxes"),
    Coverage("mjVIS_PERTFORCE", "degraded", "forge-native perturbation feedback"),
    Coverage("mjVIS_PERTOBJ", "degraded", "forge-native perturbation feedback"),
    Coverage("mjVIS_CONTACTPOINT", "supported", "contact point debug layer"),
    Coverage("mjVIS_ISLAND", "unsupported", "constraint islands are not colored"),
    Coverage("mjVIS_CONTACTFORCE", "supported", "contact force debug layer"),
    Coverage("mjVIS_CONTACTSPLIT", "unsupported", "split contact components are not drawn"),
    Coverage("mjVIS_TRANSPARENT", "supported", "transparent material pass and flag"),
    Coverage("mjVIS_AUTOCONNECT", "unsupported", "auto-connect lines are not drawn"),
    Coverage("mjVIS_COM", "supported", "root subtree center-of-mass markers"),
    Coverage("mjVIS_SELECT", "degraded", "GPU picking and forge outline replace MuJoCo select"),
    Coverage("mjVIS_STATIC", "degraded", "static bodies render, but have no separate filter flag"),
    Coverage("mjVIS_SKIN", "supported", "dynamic skinned meshes"),
    Coverage("mjVIS_FLEXVERT", "unsupported", "flex vertex markers are not drawn"),
    Coverage("mjVIS_FLEXEDGE", "unsupported", "flex edge debug overlays are not drawn"),
    Coverage("mjVIS_FLEXFACE", "supported", "dynamic flex surfaces"),
    Coverage("mjVIS_FLEXSKIN", "degraded", "surface skinning renders without an independent flag"),
    Coverage("mjVIS_BODYBVH", "unsupported", "body BVH overlays are not drawn"),
    Coverage("mjVIS_MESHBVH", "unsupported", "mesh BVH overlays are not drawn"),
    Coverage("mjVIS_SDFITER", "unsupported", "SDF iteration overlays are not drawn"),
)


def visual_coverage() -> dict[str, list[dict]]:
    """Return forge coverage for every MuJoCo visualization flag."""
    return {
        "mjtRndFlag": [asdict(item) for item in _RND_COVERAGE],
        "mjtVisFlag": [asdict(item) for item in _VIS_COVERAGE],
    }


def audit_model(model) -> dict:
    supported = {
        int(mujoco.mjtGeom.mjGEOM_PLANE),
        int(mujoco.mjtGeom.mjGEOM_HFIELD),
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_BOX),
        int(mujoco.mjtGeom.mjGEOM_MESH),
        int(mujoco.mjtGeom.mjGEOM_SDF),
    }
    groups = set(DEFAULT_GEOM_GROUPS)
    visible = [i for i in range(model.ngeom) if int(model.geom_group[i]) in groups]
    types = Counter(int(model.geom_type[i]) for i in visible)
    findings: list[Finding] = []

    for kind, count in sorted(types.items()):
        if kind not in supported:
            findings.append(
                Finding(
                    _enum_name(mujoco.mjtGeom, kind), "unsupported", count, "geometry is skipped"
                )
            )
    hidden = model.ngeom - len(visible)
    if hidden:
        findings.append(
            Finding(
                "visual groups 3-5",
                "hidden",
                hidden,
                "hidden by MuJoCo's default; enable them in Settings > visual groups",
            )
        )
    if model.nhfield:
        findings.append(
            Finding("heightfield", "supported", model.nhfield, "closed indexed mesh with materials")
        )
    if model.nsite:
        shown = sum(int(g) in groups for g in model.site_group)
        findings.append(
            Finding("site", "supported", shown, "rendered as regular shaded scene instances")
        )
    if model.ntendon:
        findings.append(
            Finding(
                "tendon",
                "supported",
                model.ntendon,
                "instanced 3D capsules with model color, world width, wrap half-width and depth",
            )
        )
        materialized = int(np.count_nonzero(np.asarray(model.tendon_matid) >= 0))
        if materialized:
            findings.append(
                Finding(
                    "tendon material",
                    "supported",
                    materialized,
                    "RGBA override, lighting scalars, texture, repeat and transparency",
                )
            )
    if model.ncam:
        findings.append(
            Finding(
                "model camera",
                "supported",
                model.ncam,
                "named perspective and orthographic cameras can drive the viewport",
            )
        )
        intrinsic = np.asarray(getattr(model, "cam_intrinsic", np.zeros((model.ncam, 4))))
        shifted = int(np.count_nonzero(np.any(np.abs(intrinsic[:, 2:4]) > 1e-9, axis=1)))
        if shifted:
            findings.append(
                Finding(
                    "camera principal point",
                    "degraded",
                    shifted,
                    "off-center projection is rendered on the centered forge projection",
                )
            )
    if model.nkey:
        findings.append(
            Finding(
                "keyframe",
                "supported",
                model.nkey,
                "listed and loadable from the Control panel while paused",
            )
        )
    if model.nsensor:
        findings.append(
            Finding(
                "sensor",
                "supported",
                model.nsensor,
                "metadata and live values are available in the Sensors panel (F11)",
            )
        )
    if model.nu:
        findings.append(
            Finding(
                "actuator control",
                "supported",
                model.nu,
                "live control values and model ranges are available in the Joints panel",
            )
        )
        tendon_actuators = int(
            np.count_nonzero(np.asarray(model.actuator_trntype) == int(mujoco.mjtTrn.mjTRN_TENDON))
        )
        if tendon_actuators:
            findings.append(
                Finding(
                    "actuator visualization",
                    "supported",
                    tendon_actuators,
                    "spatial tendon path with ctrl/activation color interpolation",
                )
            )
        if tendon_actuators < model.nu:
            findings.append(
                Finding(
                    "actuator visualization",
                    "degraded",
                    model.nu - tendon_actuators,
                    "joint/site/body transmission icons are not drawn",
                )
            )
    if model.nflex:
        findings.append(
            Finding(
                "flex",
                "supported",
                model.nflex,
                "1D cables and 2D/3D surfaces update through the generic dynamic-mesh contract",
            )
        )
    if model.nskin:
        findings.append(
            Finding(
                "skin",
                "supported",
                model.nskin,
                "weighted bones, smooth normals, inflation, materials and UVs",
            )
        )
    image_kind = getattr(mujoco.mjtLightType, "mjLIGHT_IMAGE", None)
    if model.nlight:
        findings.append(
            Finding(
                "light entity",
                "supported",
                model.nlight,
                "selectable in Hierarchy and editable in Inspector; body attachment stays dynamic",
            )
        )
    if image_kind is not None:
        images = sum(int(x) == int(image_kind) for x in model.light_type)
        if images:
            findings.append(
                Finding("image light", "degraded", images, "rendered as a directional light")
            )
    if model.nlight > 16:
        findings.append(
            Finding("lights", "degraded", model.nlight - 16, "forge shades the first 16 lights")
        )

    unsupported = sum(x.count for x in findings if x.status == "unsupported")
    degraded = sum(x.count for x in findings if x.status == "degraded")
    return {
        "counts": {
            "geom": int(model.ngeom),
            "geom_visible": len(visible),
            "site": int(model.nsite),
            "tendon": int(model.ntendon),
            "camera": int(model.ncam),
            "keyframe": int(model.nkey),
            "sensor": int(model.nsensor),
            "actuator": int(model.nu),
            "light": int(model.nlight),
            "heightfield": int(model.nhfield),
            "flex": int(model.nflex),
            "skin": int(model.nskin),
        },
        "unsupported": unsupported,
        "degraded": degraded,
        "findings": [asdict(x) for x in findings],
        "coverage": visual_coverage(),
    }


def _enum_name(enum_type, value: int) -> str:
    try:
        return str(enum_type(value)).split(".")[-1]
    except ValueError:
        return f"unknown({value})"
