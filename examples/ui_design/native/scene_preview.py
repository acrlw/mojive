"""Render the reference document with Mojive's real scene and camera contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mojive import CameraView, Scene, math3d
from mojive.interaction.gizmo import SIZE_PT, GizmoFrame, GizmoHandle, GizmoMode, GizmoSpace
from mojive.types import Light, LightSet, Material, TextureData, TextureType
from mojive.ui.camera import OrbitCamera

ROOT = Path(__file__).parent


def rgba(value):
    return (*[int(value[n : n + 2], 16) / 255 for n in (1, 3, 5)], 1.0)


def rotation(degrees):
    return math3d.euler_xyz_to_mat3(np.radians(degrees))


class Preview:
    def __init__(self, window, backend_name):
        if backend_name == "bgfx":
            from mojive.render.native.backend import NativeBackend

            self.backend = NativeBackend(800, 700, 4)
        else:
            from mojive.render.opengl.backend import OpenGLBackend

            window.make_current()
            self.backend = OpenGLBackend(None, 800, 700, 4)
        self.document = json.loads((ROOT / "document.json").read_text())
        self.entities = self.document["entities"]
        self.selected = "hinge_body"
        self.parts = {}
        self.identities = {}
        self.scene = Scene(
            lights=LightSet(
                lights=(
                    Light(
                        direction=np.array([0.3, 0.5, -1], np.float32),
                        diffuse=np.array([0.85, 0.81, 0.75], np.float32),
                        ambient=np.full(3, 0.18, np.float32),
                    ),
                ),
                headlight=Light(diffuse=np.full(3, 0.35, np.float32), cast_shadow=False),
                ambient=np.full(3, 0.35, np.float32),
                fog_color=np.array([0.165, 0.188, 0.22], np.float32),
                fog_start=17,
                fog_end=46,
            )
        )
        pixels = np.zeros((128, 128, 3), np.uint8)
        yy, xx = np.indices((128, 128))
        pixels[:] = [52, 59, 69]
        pixels[((xx // 64 + yy // 64) % 2) == 1] = [65, 75, 88]
        self.scene.add_texture(TextureData("checker", TextureType.TWO_D, pixels))
        self.floor = self.scene.plane(
            name="floor",
            size=(40, 40, 0.01),
            position=(0, 0, -0.015),
            material=Material(
                rgba=np.ones(4, np.float32),
                texture="checker",
                tex_repeat=np.array([1.0, 1.0], np.float32),
                tex_uniform=True,
                specular=0,
                roughness=1,
            ),
        )
        dark = rgba("#535c67")
        for e in self.entities:
            self.parts[e["id"]] = []
            if e["type"] != "link":
                continue
            color = rgba(e["color"])

            def add(shape, size, offset=(0, 0, 0), r=(0, 0, 0), tint=color, entity=e):
                obj = getattr(self.scene, shape)(
                    name=entity["id"] + "_" + str(len(self.parts[entity["id"]])),
                    size=size,
                    color=tint,
                    material=Material(specular=0.2, roughness=0.55),
                )
                self.parts[entity["id"]].append(
                    (obj, np.array(offset), rotation(r), np.array(size), tint)
                )
                self.identities[obj.object_id] = entity["id"]

            shape = e["shape"]
            if shape == "box":
                add("box", (0.25, 0.25, 0.25))
                add("cylinder", (0.07, 0.07, 0.16), (0, 0, 0.36))
            elif shape == "ball":
                add("box", (0.14, 0.14, 0.125), (0, 0, 0.42), tint=dark)
                add("cylinder", (0.068, 0.068, 0.375), (0, 0, -0.05), (0, -11.46, 0))
                add("sphere", (0.14, 0.14, 0.14), (-0.07, 0, -0.43))
            elif shape == "slide":
                add("cylinder", (0.029, 0.029, 0.65), r=(90, 0, 0), tint=dark)
                add("box", (0.175, 0.175, 0.175))
            elif shape == "hinge":
                add("cylinder", (0.15, 0.15, 0.075), tint=dark)
                add("cylinder", (0.072, 0.072, 0.425), (0, 0.4, 0), (90, 0, 0))
                add("sphere", (0.072, 0.072, 0.072), (0, 0.83, 0))
            elif shape == "chain":
                add("cylinder", (0.055, 0.055, 0.55), (0, 0, 0.38))
                add("box", (0.12, 0.12, 0.12), (0, 0, -0.2), tint=dark)
                add("sphere", (0.058, 0.058, 0.058), (0, 0, 0.93))
        self.camera = OrbitCamera()
        self.camera.adopt(
            CameraView(
                eye=np.array([4.3, -5.9, 4.2]),
                target=np.array([0, 0, 0.55]),
                fov_y=np.radians(42),
                near=0.05,
                far=100,
            )
        )
        self.camera.attach(self.backend)
        self._signature = None
        self.size = (800, 700)
        self.tool = "rotate"
        self.space = "World"
        self.hovered = self.active = GizmoHandle.NONE
        self.gizmos = True
        self.shadows = True
        self.visible = True
        self.sync()
        self.backend.set_scene(self.scene.source)
        self.backend.set_background((0.165, 0.188, 0.22, 1))

    @property
    def entity(self):
        return next((e for e in self.entities if e["id"] == self.selected), None)

    def sync(self):
        signature = json.dumps([self.entities, self.document["pose"]], sort_keys=True)
        if signature == self._signature:
            return
        self._signature = signature
        pose = self.document["pose"]
        for e in self.entities:
            p = np.array(e["position"], dtype=float)
            r = list(e["rotation"])
            if e["id"] == "hinge_body":
                r[0] += pose["hinge"]
            if e["id"] == "ball_body":
                r[1] += pose["ball"]
            if e["id"] == "slide_body":
                p[1] += pose["slide"]
            rot = rotation(r)
            scale = np.array(e["scale"])
            for obj, offset, local, size, tint in self.parts[e["id"]]:
                obj.set_pose(p + rot @ (offset * scale), rot @ local)
                obj.set_size(size * scale)
                c = rgba(e["color"]) if tint != rgba("#535c67") else tint
                obj.set_color((*c[:3], 0 if e["hidden"] else 1))

    def render(self, width, height, scale):
        size = (max(32, int(width)), max(32, int(height)))
        if size != self.size:
            self.backend.resize(*size)
            self.size = size
        self.sync()
        self.camera.set_aspect(size[0] / size[1])
        self.camera.advance(1 / 60, self.backend)
        self.backend.set_camera(self.camera.view())
        self.backend.update(self.scene.frame)
        e = self.entity
        parts = self.parts.get(self.selected, [])
        self.backend.highlight(parts[0][0].object_id if parts else 0, fill=False, outline=True)
        if e and self.gizmos and self.tool != "select" and not e["hidden"]:
            mode = {
                "move": GizmoMode.TRANSLATE,
                "rotate": GizmoMode.ROTATE,
                "scale": GizmoMode.DIMENSIONS,
            }[self.tool]
            self.backend.set_gizmo(
                GizmoFrame(
                    mode=mode,
                    space=GizmoSpace.WORLD if self.space == "World" else GizmoSpace.BODY,
                    rotation=rotation(e["rotation"]) if self.space == "Body" else np.eye(3),
                    hovered=self.hovered,
                    active=self.active,
                    position=np.array(e["position"], np.float32),
                    size_px=SIZE_PT * scale,
                )
            )
        else:
            self.backend.set_gizmo(None)
        return self.backend.render()

    def pick(self, x, y):
        oid = self.backend.pick(int(x), int(y))
        self.selected = self.identities.get(oid)

    def close(self):
        self.backend.release()
