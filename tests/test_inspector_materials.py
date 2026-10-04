"""Inspector material edits retain command ordering, capability gates, and undo."""

from dataclasses import replace

import numpy as np
import pytest
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.adapters.base import NodeType
from mojive.adapters.static import StaticSceneAdapter
from mojive.render.backend import NullBackend
from mojive.scene import Scene
from mojive.session import Session
from mojive.types import Material
from mojive.ui.panels import PanelContext
from mojive.ui.panels.inspector import InspectorPanel, geometry, materials


class _MaterialInputs:
    """Supply widget results while executing the production panel and Session commands."""

    def __init__(self):
        self.edits = {}
        self.choices = {}
        self.buttons = set()
        self.active = False
        self.dimensions = False
        self.disabled = 0
        self.combo = ""

    def edit(self, label, value, *_args, **_kwargs):
        if not self.disabled and label in self.edits:
            return True, self.edits.pop(label)
        return False, value

    def begin_combo(self, label, *_args):
        self.combo = label
        return not self.disabled and label in self.choices

    def selectable(self, label, selected):
        return self.choices.get(self.combo) == label, selected

    def button_row(self, _ctx, _label, labels):
        return tuple(not self.disabled and label in self.buttons for label in labels)

    def begin_disabled(self):
        self.disabled += 1

    def end_disabled(self):
        self.disabled -= 1
        assert self.disabled >= 0


@pytest.fixture
def inputs(monkeypatch):
    values = _MaterialInputs()
    for module in (geometry, materials):
        monkeypatch.setattr(module, "_property_control_row", lambda *_args: None)
        monkeypatch.setattr(module, "_property_color_edit4", lambda _ctx, *args: values.edit(*args))
        monkeypatch.setattr(module, "_property_button_row", values.button_row)
    monkeypatch.setattr(geometry, "_begin_property_table", lambda *_args: True)
    monkeypatch.setattr(
        geometry,
        "_property_section",
        lambda _ctx, label: label != "geometry dimensions" or values.dimensions,
    )
    for name in (
        "push_id",
        "pop_id",
        "end_table",
        "text",
        "text_disabled",
        "set_item_tooltip",
        "align_text_to_frame_padding",
        "end_combo",
    ):
        monkeypatch.setattr(imgui, name, lambda *_args: None)
    monkeypatch.setattr(imgui, "collapsing_header", lambda *_args: True)
    monkeypatch.setattr(imgui, "is_any_item_active", lambda: values.active)
    monkeypatch.setattr(imgui, "begin_disabled", values.begin_disabled)
    monkeypatch.setattr(imgui, "end_disabled", values.end_disabled)
    monkeypatch.setattr(imgui, "begin_combo", values.begin_combo)
    monkeypatch.setattr(imgui, "selectable", values.selectable)
    for name in ("drag_float", "drag_float2", "checkbox"):
        monkeypatch.setattr(imgui, name, values.edit)
    yield values
    assert values.disabled == 0


class _ModelMaterialsAdapter(StaticSceneAdapter):
    """Small editable material catalog using real Scene storage and checkpoints."""

    caps = replace(StaticSceneAdapter.caps, model_assets=True)

    def scene_source(self):
        source = super().scene_source()
        for node in source.nodes:
            if node.type is NodeType.GEOM:
                node.model_id = 0
                node.source_editable = True
        return source

    def model_material_indices(self, model_id):
        return tuple(range(len(self.scene.source.materials))) if model_id == 0 else ()

    def set_geometry_material(self, node_id, material_index):
        source = self.scene.source
        instance = int(np.flatnonzero(source.geom_node == node_id)[0])
        self.scene.set_object_material(
            int(source.geom_object_id[instance]), source.materials[material_index]
        )
        return True

    def add_model_material(self, node_id, name, copy_from=-1):
        source = self.scene.source
        original = source.materials[copy_from] if copy_from >= 0 else Material()
        material = replace(original, name=f"opengl_0_{name}")
        instance = int(np.flatnonzero(source.geom_node == node_id)[0])
        self.scene.set_object_material(int(source.geom_object_id[instance]), material)
        return next(
            i for i, value in enumerate(self.scene.source.materials) if value.name == material.name
        )


@pytest.fixture
def editor(request):
    scene = Scene()
    scene.sphere(name="target", material=Material(name="opengl_0_material", specular=0.2))
    scene.box(name="other", material=Material(name="opengl_0_other", specular=0.8))
    adapter = (
        _ModelMaterialsAdapter(scene)
        if getattr(request, "param", False)
        else StaticSceneAdapter(scene)
    )
    session = Session(adapter)
    panel = InspectorPanel()
    ctx = PanelContext(session=session, backend=NullBackend())
    node_id = next(node.node_id for node in session.nodes if node.name == "target.geom")
    yield panel, ctx, node_id
    session.release()


def _draw(editor):
    panel, ctx, node_id = editor
    panel._material(ctx, ctx.session.node(node_id))
    panel.finish_frame(ctx)


def _material(editor):
    _, ctx, node_id = editor
    source = ctx.session.source
    instance = int(np.flatnonzero(source.geom_node == node_id)[0])
    return source.materials[source.geom_material[instance]]


def test_idle_material_panel_does_not_create_history(editor, inputs):
    _, ctx, _ = editor
    before = ctx.session.document_revision
    _draw(editor)
    _draw(editor)
    assert not ctx.session.can_undo
    assert ctx.session.document_revision == before


def test_surface_and_texture_drag_share_one_undo_transaction(editor, inputs):
    _, ctx, _ = editor
    original = _material(editor)
    inputs.active = True
    inputs.edits = {"##material_specular": 0.45, "##material_texture_repeat": (2.0, 3.0)}
    _draw(editor)
    inputs.edits = {"##material_specular": 0.65}
    _draw(editor)
    assert _material(editor).specular == pytest.approx(0.65)
    assert _material(editor).tex_repeat == pytest.approx((2.0, 3.0))
    assert ctx.session.editing
    inputs.active = False
    _draw(editor)
    assert not ctx.session.editing
    assert ctx.submit(cmd.Undo()).ok
    assert _material(editor).specular == original.specular
    assert _material(editor).tex_repeat == pytest.approx(original.tex_repeat)
    assert not ctx.session.can_undo


def test_preset_and_override_controls_preserve_the_disabled_sentinel(editor, inputs):
    inputs.choices = {"##material_preset": "Metal"}
    inputs.edits = {"##material_metallic_override": True, "##material_roughness_override": True}
    _draw(editor)
    value = _material(editor)
    assert (value.specular, value.shininess, value.reflectance) == pytest.approx((0.9, 0.9, 0.65))
    assert value.metallic == 0.0 and value.roughness == 0.5
    inputs.choices.clear()
    inputs.edits = {"##material_metallic_override": False, "##material_roughness_override": False}
    _draw(editor)
    assert _material(editor).metallic == -1.0 and _material(editor).roughness == -1.0


def test_geometry_dimensions_edit_without_physics_and_undo(editor, inputs):
    _, ctx, _ = editor
    before = ctx.session.source.geom_size.copy()
    inputs.dimensions = True
    inputs.edits = {"##geometry_dimension": 2.0}
    _draw(editor)
    assert ctx.session.source.geom_size[0] == pytest.approx((2.0, 2.0, 2.0))
    assert ctx.submit(cmd.Undo()).ok
    assert ctx.session.source.geom_size == pytest.approx(before)


@pytest.mark.parametrize("editor", [True], indirect=True)
def test_assignment_discards_old_material_edits_from_the_same_frame(editor, inputs):
    inputs.choices = {"##assigned_material": "opengl_0_other"}
    inputs.edits = {"##material_specular": 0.1}
    _draw(editor)
    assert _material(editor).name == "opengl_0_other"
    assert _material(editor).specular == 0.8
    assert editor[1].submit(cmd.Undo()).ok
    assert _material(editor).name == "opengl_0_material"
    assert _material(editor).specular == 0.2


@pytest.mark.parametrize("editor", [True], indirect=True)
def test_duplicate_uses_original_material_and_returns_before_property_writes(editor, inputs):
    inputs.buttons = {"Duplicate material"}
    inputs.edits = {"##material_specular": 0.9}
    _draw(editor)
    assert _material(editor).name == "opengl_0_material2"
    assert _material(editor).specular == 0.2


@pytest.mark.parametrize("editor", [True], indirect=True)
def test_texture_import_actions_keep_their_target_and_kind(editor, inputs):
    _, ctx, node_id = editor
    imports = []
    ctx.request_texture_import = lambda *args: imports.append(args)
    source = ctx.session.source
    instance = int(np.flatnonzero(source.geom_node == node_id)[0])
    material_index = source.geom_material[instance]
    inputs.buttons = {"Import texture", "Import cube texture", "Import skybox texture"}
    _draw(editor)
    assert imports == [(0, material_index), (0, -1, "cube"), (0, -1, "skybox")]
    assert not ctx.session.can_undo


@pytest.mark.parametrize("editor", [True], indirect=True)
def test_running_simulation_disables_asset_actions_but_keeps_appearance_editable(editor, inputs):
    _, ctx, _ = editor
    imports = []
    ctx.request_texture_import = lambda *args: imports.append(args)
    ctx.session.adapter.caps = replace(ctx.session.adapter.caps, simulation=True)
    assert ctx.submit(cmd.Play()).ok
    inputs.buttons = {
        "New material",
        "Duplicate material",
        "Import texture",
        "Import cube texture",
        "Import skybox texture",
    }
    inputs.edits = {"##material_specular": 0.7}
    _draw(editor)
    assert _material(editor).name == "opengl_0_material"
    assert _material(editor).specular == 0.7
    assert not imports
