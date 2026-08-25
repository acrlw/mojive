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
    Coverage("mjRND_SHADOW", "exact", "MuJoCo light shadow flags drive forge shadow passes"),
    Coverage("mjRND_WIREFRAME", "equivalent", "forge wireframe rasterization"),
    Coverage("mjRND_REFLECTION", "equivalent", "forge planar reflection passes"),
    Coverage("mjRND_ADDITIVE", "exact", "additive blending for transparent geometry"),
    Coverage("mjRND_SKYBOX", "exact", "skybox texture and enable flag"),
    Coverage("mjRND_FOG", "equivalent", "forge linear fog"),
    Coverage("mjRND_HAZE", "equivalent", "horizon haze over infinite planes"),
    Coverage("mjRND_DEPTH", "equivalent", "metric depth debug view"),
    Coverage("mjRND_SEGMENT", "exact", "MuJoCo object ID and object type segmentation"),
    Coverage("mjRND_IDCOLOR", "equivalent", "stable forge object-ID colors"),
    Coverage("mjRND_CULL_FACE", "exact", "back-face culling flag"),
)

_VIS_COVERAGE = (
    Coverage(
        "mjVIS_CONVEXHULL",
        "exact",
        "collision mesh and SDF geoms switch to MuJoCo compiled convex hulls",
    ),
    Coverage("mjVIS_TEXTURE", "exact", "2D and cube color textures plus skyboxes"),
    Coverage("mjVIS_JOINT", "equivalent", "solid free, ball, slide and hinge markers"),
    Coverage(
        "mjVIS_CAMERA",
        "equivalent",
        "named cameras, editable scene entities and viewport frustum icons",
    ),
    Coverage(
        "mjVIS_ACTUATOR",
        "equivalent",
        "joint, joint-in-parent, site, body, tendon and slider-crank visuals",
    ),
    Coverage(
        "mjVIS_ACTIVATION",
        "exact",
        "control and activation ranges drive MuJoCo actuator colors",
    ),
    Coverage(
        "mjVIS_LIGHT",
        "equivalent",
        "editable scene entities with point and direction viewport icons",
    ),
    Coverage("mjVIS_TENDON", "exact", "dynamic tendon paths, widths, colors and materials"),
    Coverage(
        "mjVIS_RANGEFINDER",
        "equivalent",
        "site and camera rays, hit points and surface normals",
    ),
    Coverage(
        "mjVIS_CONSTRAINT",
        "equivalent",
        "connect and weld equality endpoint markers",
    ),
    Coverage("mjVIS_INERTIA", "exact", "body inertia boxes"),
    Coverage("mjVIS_SCLINERTIA", "exact", "constant-density inertia boxes"),
    Coverage("mjVIS_PERTFORCE", "equivalent", "forge translation perturbation feedback"),
    Coverage("mjVIS_PERTOBJ", "equivalent", "forge rotation perturbation feedback"),
    Coverage("mjVIS_CONTACTPOINT", "exact", "contact points use MuJoCo visualization data"),
    Coverage(
        "mjVIS_ISLAND",
        "exact",
        "official island colors for moving geoms, flexes, tendons, and contacts",
    ),
    Coverage("mjVIS_CONTACTFORCE", "exact", "contact forces use mj_contactForce"),
    Coverage(
        "mjVIS_CONTACTSPLIT",
        "exact",
        "normal and friction contact-force components use separate arrows",
    ),
    Coverage("mjVIS_TRANSPARENT", "exact", "visual alpha multiplier and transparent pass"),
    Coverage(
        "mjVIS_AUTOCONNECT",
        "exact",
        "body centers connect through reverse joint order to parent centers",
    ),
    Coverage("mjVIS_COM", "exact", "root subtree center-of-mass markers"),
    Coverage("mjVIS_SELECT", "equivalent", "GPU picking and forge outline"),
    Coverage("mjVIS_STATIC", "exact", "independent static-body filter"),
    Coverage("mjVIS_SKIN", "exact", "dynamic skinned meshes with an independent flag"),
    Coverage("mjVIS_FLEXVERT", "exact", "GPU point overlay follows dynamic flex vertices"),
    Coverage("mjVIS_FLEXEDGE", "exact", "GPU line overlay follows dynamic flex topology"),
    Coverage("mjVIS_FLEXFACE", "exact", "flat element faces selected independently"),
    Coverage("mjVIS_FLEXSKIN", "exact", "smooth shell surfaces selected independently"),
    Coverage("mjVIS_BODYBVH", "exact", "body BVH boxes with depth and active-node colors"),
    Coverage(
        "mjVIS_MESHBVH",
        "exact",
        "mesh, octree and flex BVH boxes plus interpolated-flex control cages",
    ),
    Coverage(
        "mjVIS_SDFITER",
        "deferred",
        "planned after a dedicated SDF iteration data path",
    ),
)


def visual_coverage() -> dict[str, list[dict]]:
    """Return forge coverage for every MuJoCo visualization flag."""
    return {
        "mjtRndFlag": [asdict(item) for item in _RND_COVERAGE],
        "mjtVisFlag": [asdict(item) for item in _VIS_COVERAGE],
    }


def audit_model(model) -> dict:
    """Report Forge coverage and model usage for one compiled MuJoCo model."""

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
                "named perspective and orthographic cameras are editable viewport entities",
            )
        )
        intrinsic = np.asarray(getattr(model, "cam_intrinsic", np.zeros((model.ncam, 4))))
        shifted = int(np.count_nonzero(np.any(np.abs(intrinsic[:, 2:4]) > 1e-9, axis=1)))
        if shifted:
            findings.append(
                Finding(
                    "camera principal point",
                    "supported",
                    shifted,
                    "physical camera intrinsics drive an off-center projection",
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
    if model.nmocap:
        findings.append(
            Finding(
                "mocap body",
                "supported",
                model.nmocap,
                "editable through the shared transform inspector and gizmo",
            )
        )
    if model.neq:
        findings.append(
            Finding(
                "equality constraint",
                "supported",
                model.neq,
                "runtime enable state is editable in the Control panel",
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
        rangefinder_kind = int(mujoco.mjtSensor.mjSENS_RANGEFINDER)
        rangefinders = int(
            np.count_nonzero(np.asarray(model.sensor_type, np.int32) == rangefinder_kind)
        )
        if rangefinders:
            findings.append(
                Finding(
                    "rangefinder visualization",
                    "supported",
                    rangefinders,
                    "site and camera rays include requested hit points and normals",
                )
            )
    if model.nactuator:
        findings.append(
            Finding(
                "actuator control",
                "supported",
                model.nactuator,
                "live control values and model ranges are available in the Joints panel",
            )
        )
        visual_transmissions = {
            int(mujoco.mjtTrn.mjTRN_JOINT),
            int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),
            int(mujoco.mjtTrn.mjTRN_TENDON),
            int(mujoco.mjtTrn.mjTRN_SITE),
            int(mujoco.mjtTrn.mjTRN_BODY),
        }
        visualized = int(
            np.count_nonzero(
                np.isin(np.asarray(model.actuator_trntype, np.int32), tuple(visual_transmissions))
            )
        )
        if visualized:
            findings.append(
                Finding(
                    "actuator visualization",
                    "supported",
                    visualized,
                    "joint, site, body and tendon transmissions use ctrl/activation colors",
                )
            )
        slider_crank = int(
            np.count_nonzero(
                np.asarray(model.actuator_trntype) == int(mujoco.mjtTrn.mjTRN_SLIDERCRANK)
            )
        )
        if slider_crank:
            findings.append(
                Finding(
                    "slider-crank visualization",
                    "supported",
                    slider_crank,
                    "slider cylinder, connecting rod and broken-crank color follow MuJoCo",
                )
            )
    if model.nflex:
        findings.append(
            Finding(
                "flex",
                "supported",
                model.nflex,
                "1D cables, 2D/3D surfaces, vertices, and edges update through generic scene data",
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
                "selectable, editable and visible as a viewport icon; body attachment stays dynamic",
            )
        )
    if image_kind is not None:
        images = sum(int(x) == int(image_kind) for x in model.light_type)
        if images:
            findings.append(
                Finding(
                    "image light",
                    "supported",
                    images,
                    "cube-map diffuse and roughness-aware specular environment lighting",
                )
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
            "mocap": int(model.nmocap),
            "equality": int(model.neq),
            "sensor": int(model.nsensor),
            "actuator": int(model.nactuator),
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
