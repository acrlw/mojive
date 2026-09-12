"""Topology, component and asset edits through editable model specs."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ...commands import (
    AddModelElementEdit,
    ModelEdit,
    ModelElementRef,
    RemoveModelElementEdit,
    RenameModelElementEdit,
)
from ...types import (
    MATERIAL_TEXTURE_ROLES,
)
from ..base import (
    ModelAssetInfo,
    ModelComponentField,
    ModelComponentInfo,
    ModelComponentPathItem,
    NodeType,
)
from .engine import mujoco
from .schema import (
    _MJCF_SCHEMA_ATTRIBUTES,
    _MODEL_ASSET_REFERENCE_FIELDS,
    _MODEL_ASSET_TYPES,
    _MODEL_COMPONENT_CATEGORIES,
    _REFERENCE_ELEMENT,
    _component_fields,
    _component_path_fields,
    _component_path_presets,
    _component_presets,
    _ComponentReferences,
    _ensure_model_asset_section,
    _model_asset_element,
    _model_asset_references,
)
from .spec import (
    _component_section,
    _component_xml,
    _editable_spec_xml,
    _format_mjcf_values,
    _serialize_component_xml,
)
from .state import _ModelComponentEntry

_TOPOLOGY_XML_KIND = {
    "body": "body",
    "geom": "geom",
    "joint": "joint",
    "freejoint": "joint",
    "site": "site",
    "camera": "camera",
    "light": "light",
}


_TOPOLOGY_NODE_TAGS = {
    NodeType.ROBOT: ("body",),
    NodeType.LINK: ("body",),
    NodeType.GEOM: ("geom",),
    NodeType.JOINT: ("joint", "freejoint"),
    NodeType.SITE: ("site",),
    NodeType.CAMERA: ("camera",),
    NodeType.LIGHT: ("light",),
}


def _next_topology_copy_name(name: str, occupied: set[str]) -> str:
    match = re.fullmatch(r"(.+)_copy(?:(\d+))?", name)
    if match is None:
        stem = f"{name}_copy"
        index = 1
    else:
        stem = f"{match.group(1)}_copy"
        index = int(match.group(2) or 1) + 1
    while True:
        candidate = stem if index == 1 else f"{stem}{index}"
        if candidate not in occupied:
            occupied.add(candidate)
            return candidate
        index += 1


def _duplicate_topology_xml(
    root: ET.Element, node_type: NodeType, name: str
) -> tuple[ET.Element, str] | None:
    """Clone one normalized topology element and namespace its owned subtree.

    MuJoCo object names are unique per object type, not across the whole model.
    Keep asset/default references shared while remapping references to bodies and
    other topology elements that are part of the copied body subtree.
    """

    tags = _TOPOLOGY_NODE_TAGS.get(node_type, ())
    source = next(
        (
            element
            for tag in tags
            for element in root.iter(tag)
            if str(element.attrib.get("name", "")).strip() == name
        ),
        None,
    )
    if source is None:
        return None
    parent = next((element for element in root.iter() if source in tuple(element)), None)
    if parent is None:
        return None

    occupied: dict[str, set[str]] = {kind: set() for kind in set(_TOPOLOGY_XML_KIND.values())}
    for element in root.iter():
        kind = _TOPOLOGY_XML_KIND.get(element.tag)
        value = str(element.attrib.get("name", "")).strip()
        if kind and value:
            occupied[kind].add(value)

    duplicate = deepcopy(source)
    renamed: dict[tuple[str, str], str] = {}
    for element in duplicate.iter():
        kind = _TOPOLOGY_XML_KIND.get(element.tag)
        value = str(element.attrib.get("name", "")).strip()
        if not kind or not value:
            continue
        copied = _next_topology_copy_name(value, occupied[kind])
        element.set("name", copied)
        renamed[(kind, value)] = copied

    reference_kinds = {**_REFERENCE_ELEMENT, "target": "body", "targetbody": "body"}
    for element in duplicate.iter():
        for field, value in tuple(element.attrib.items()):
            kind = reference_kinds.get(field)
            if field == "objname":
                kind = element.attrib.get("objtype", "")
            elif field == "refname":
                kind = element.attrib.get("reftype", "")
            copied = renamed.get((str(kind), value)) if kind else None
            if copied is not None:
                element.set(field, copied)

    source_index = tuple(parent).index(source)
    parent.insert(source_index + 1, duplicate)
    source_kind = _TOPOLOGY_XML_KIND[source.tag]
    return root, renamed[(source_kind, name)]


class _ModelEditing:
    """Topology, component and asset edits through editable model specs.

    Private implementation of MuJoCoAdapter; owns no independent model or lifecycle.
    """

    def add_model_element(self, parent_node_id: int, element_type: str, name: str) -> int:
        if self._model_parent(int(parent_node_id)) is None or not str(name).strip():
            return -1
        results = self.apply_model_edit_batch(
            (AddModelElementEdit(ModelElementRef(node_id=int(parent_node_id)), element_type, name),)
        )
        return results[0] if results else -1

    def duplicate_model_element(self, node_id: int) -> int:
        identity = self._node_element.get(int(node_id))
        if identity is None or identity[1] not in _TOPOLOGY_NODE_TAGS:
            return -1
        model_id, node_type, name = identity
        spec = self._spec_for_model(model_id)
        if spec is None:
            return -1
        root, _xml = _component_xml(spec)
        duplicated = _duplicate_topology_xml(root, node_type, name)
        if duplicated is None:
            return -1
        root, duplicate_name = duplicated
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        if not self._replace_model_spec(model_id, edited):
            return -1
        if node_type is NodeType.GEOM:
            self._geometry_object_id(model_id, duplicate_name)
        self.nodes()
        body_types = {NodeType.ROBOT, NodeType.LINK}
        return next(
            (
                candidate_id
                for candidate_id, candidate in self._node_element.items()
                if candidate[0] == model_id
                and candidate[2] == duplicate_name
                and (
                    candidate[1] is node_type
                    or (candidate[1] in body_types and node_type in body_types)
                )
            ),
            -1,
        )

    def remove_model_element(self, node_id: int) -> bool:
        identity = self._node_element.get(int(node_id))
        if identity is None or identity[1] not in {
            NodeType.LINK,
            NodeType.ROBOT,
            NodeType.GEOM,
            NodeType.JOINT,
            NodeType.SITE,
            NodeType.CAMERA,
            NodeType.LIGHT,
        }:
            return False
        return bool(
            self.apply_model_edit_batch(
                (RemoveModelElementEdit(ModelElementRef(node_id=int(node_id))),)
            )
        )

    def rename_model_element(self, node_id: int, name: str) -> bool:
        identity = self._node_element.get(int(node_id))
        value = str(name).strip()
        if identity is None or not value:
            return False
        if identity[1] in (NodeType.WORLD, NodeType.MODEL) or value == identity[2]:
            return value == identity[2]
        return bool(
            self.apply_model_edit_batch(
                (RenameModelElementEdit(ModelElementRef(node_id=int(node_id)), name),)
            )
        )

    def apply_model_edit_batch(self, edits: tuple[ModelEdit, ...]) -> tuple[int, ...]:
        operations = tuple(edits)
        if self._root_spec is None or not operations:
            return ()
        if not all(
            isinstance(
                edit,
                (AddModelElementEdit, RemoveModelElementEdit, RenameModelElementEdit),
            )
            for edit in operations
        ):
            raise ValueError("Unsupported model edit operation")

        # Node IDs describe the currently installed hierarchy. Keep their semantic
        # identities while edits rename elements and before the hierarchy is rebuilt.
        nodes_by_id = {node.node_id: node for node in self.nodes()}
        node_identities: dict[int, tuple[int, NodeType, str]] = {}
        batch_identities: dict[str, tuple[int, NodeType, str]] = {}
        batch_elements: dict[str, Any] = {}
        result_identities: list[tuple[int, NodeType, str] | None] = []

        def node_identity(node_id: int) -> tuple[int, NodeType, str]:
            value = node_identities.get(int(node_id))
            if value is None:
                value = self._node_element.get(int(node_id))
                if value is None:
                    raise ValueError(f"Unknown model node_id={node_id}")
                node_identities[int(node_id)] = value
            return value

        def identity(ref: ModelElementRef) -> tuple[int, NodeType, str]:
            key = str(ref.batch_key).strip()
            has_node = int(ref.node_id) >= 0
            if has_node == bool(key):
                raise ValueError("A model element reference must use one node ID or batch key")
            if key:
                value = batch_identities.get(key)
                if value is None:
                    raise ValueError(f"Unknown model edit batch key {key!r}")
                return value
            return node_identity(int(ref.node_id))

        def parent_identity(ref: ModelElementRef) -> tuple[int, NodeType, str]:
            if str(ref.batch_key).strip():
                return identity(ref)
            node = nodes_by_id.get(int(ref.node_id))
            while node is not None:
                value = node_identities.get(node.node_id)
                if value is None:
                    value = self._node_element.get(node.node_id)
                    if value is not None:
                        node_identities[node.node_id] = value
                if value is not None and value[1] in {
                    NodeType.MODEL,
                    NodeType.WORLD,
                    NodeType.LINK,
                    NodeType.ROBOT,
                }:
                    return value
                node = nodes_by_id.get(node.parent)
            raise ValueError(f"Model node_id={ref.node_id} cannot own topology children")

        previous_root = self._root_spec
        previous_models = self._attached_models
        previous_root_edited = self._root_edited
        previous_object_ids = dict(self._geometry_object_ids)
        previous_next_object_id = self._next_geometry_object_id
        self._root_spec = previous_root.copy()
        self._attached_models = [replace(item, spec=item.spec.copy()) for item in previous_models]
        changed_models: set[int] = set()

        try:
            for edit in operations:
                if isinstance(edit, AddModelElementEdit):
                    model_id, parent_type, parent_name = parent_identity(edit.parent)
                    spec = self._spec_for_model(model_id)
                    if spec is None:
                        raise ValueError(f"Model {model_id} is unavailable")
                    if parent_type in (NodeType.MODEL, NodeType.WORLD):
                        body = spec.worldbody
                    elif parent_type in (NodeType.LINK, NodeType.ROBOT):
                        parent_key = str(edit.parent.batch_key).strip()
                        body = (
                            batch_elements.get(parent_key) if parent_key else spec.body(parent_name)
                        )
                    else:
                        body = None
                    if body is None:
                        raise ValueError(f"Model parent {parent_name!r} is unavailable")

                    value = str(edit.name).strip()
                    if not value:
                        raise ValueError("A model element name cannot be empty")
                    type_name, _, subtype = str(edit.element_type).partition(":")
                    if self._element(model_id, type_name, value) is not None:
                        raise ValueError(f"{type_name} {value!r} already exists")
                    if type_name == "body":
                        element = body.add_body(name=value)
                    elif type_name == "geom":
                        element = body.add_geom(name=value)
                        self._geometry_object_id(model_id, value)
                        element.type = {
                            "box": mujoco.mjtGeom.mjGEOM_BOX,
                            "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
                            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                            "plane": mujoco.mjtGeom.mjGEOM_PLANE,
                        }.get(subtype, mujoco.mjtGeom.mjGEOM_SPHERE)
                        element.size = [4.0, 4.0, 0.02] if subtype == "plane" else [0.1, 0.1, 0.1]
                    elif type_name == "joint":
                        element = body.add_joint(name=value)
                        element.type = {
                            "slide": mujoco.mjtJoint.mjJNT_SLIDE,
                            "ball": mujoco.mjtJoint.mjJNT_BALL,
                            "free": mujoco.mjtJoint.mjJNT_FREE,
                        }.get(subtype, mujoco.mjtJoint.mjJNT_HINGE)
                    elif type_name == "site":
                        element = body.add_site(name=value)
                        element.type = mujoco.mjtGeom.mjGEOM_SPHERE
                        element.size = [0.03, 0.03, 0.03]
                    elif type_name == "camera":
                        element = body.add_camera(name=value)
                    elif type_name == "light":
                        element = body.add_light(name=value)
                    else:
                        raise ValueError(f"Unsupported model element type {type_name!r}")
                    target_type = NodeType(type_name if type_name != "body" else "link")
                    target = (model_id, target_type, value)
                    key = str(edit.key).strip()
                    if key:
                        if key in batch_identities:
                            raise ValueError(f"Duplicate model edit batch key {key!r}")
                        batch_identities[key] = target
                        batch_elements[key] = element
                    result_identities.append(target)
                    changed_models.add(model_id)
                    continue

                target = identity(edit.target)
                model_id, node_type, current = target
                target_key = str(edit.target.batch_key).strip()
                element = (
                    batch_elements.get(target_key)
                    if target_key
                    else self._element(model_id, node_type.value, current)
                )
                spec = self._spec_for_model(model_id)
                if spec is None or element is None:
                    raise ValueError(f"{node_type.value} {current!r} is unavailable")

                if isinstance(edit, RemoveModelElementEdit):
                    if node_type not in {
                        NodeType.LINK,
                        NodeType.ROBOT,
                        NodeType.GEOM,
                        NodeType.JOINT,
                        NodeType.SITE,
                        NodeType.CAMERA,
                        NodeType.LIGHT,
                    }:
                        raise ValueError(f"{node_type.value} {current!r} cannot be removed")
                    if node_type is NodeType.GEOM:
                        self._geometry_object_ids.pop((model_id, current), None)
                    spec.delete(element)
                    result_identities.append(None)
                    changed_models.add(model_id)
                    continue

                value = str(edit.name).strip()
                if not value:
                    raise ValueError("A model element name cannot be empty")
                if node_type in (NodeType.WORLD, NodeType.MODEL):
                    raise ValueError(f"{node_type.value} {current!r} cannot be renamed")
                if value == current:
                    result_identities.append(target)
                    continue
                if self._element(model_id, node_type.value, value) is not None:
                    raise ValueError(f"{node_type.value} {value!r} already exists")
                if node_type is NodeType.GEOM:
                    object_id = self._geometry_object_ids.pop((model_id, current), None)
                    if object_id is not None:
                        self._geometry_object_ids[(model_id, value)] = object_id
                element.name = value
                renamed = (model_id, node_type, value)
                batch_identities.update(
                    (key, renamed if item == target else item)
                    for key, item in tuple(batch_identities.items())
                )
                node_identities.update(
                    (node_id, renamed if item == target else item)
                    for node_id, item in tuple(node_identities.items())
                )
                result_identities = [
                    renamed if item == target else item for item in result_identities
                ]
                result_identities.append(renamed)
                changed_models.add(model_id)

            if not changed_models:
                self._root_spec = previous_root
                self._attached_models = previous_models
            else:
                for model_id in changed_models:
                    self._mark_model_edited(model_id)
                self._recompile_topology()
                self.nodes()
        except Exception:
            self._root_spec = previous_root
            self._attached_models = previous_models
            self._root_edited = previous_root_edited
            self._geometry_object_ids = previous_object_ids
            self._next_geometry_object_id = previous_next_object_id
            raise

        by_identity = {identity: node_id for node_id, identity in self._node_element.items()}
        return tuple(
            by_identity.get(item, -1) if item is not None else -1 for item in result_identities
        )

    def scene_model_xml(self, model_id: int) -> str | None:
        spec = self._spec_for_model(model_id)
        if spec is None:
            return None
        if model_id == 0:
            return _editable_spec_xml(spec) if self._root_edited else None
        item = next((item for item in self._attached_models if item.model_id == model_id), None)
        return _editable_spec_xml(spec) if item is not None and item.edited else None

    def scene_model_source(self, model_id: int) -> str | None:
        spec = self._spec_for_model(model_id)
        return _editable_spec_xml(spec) if spec is not None else None

    def set_scene_model_xml(self, model_id: int, xml: str) -> bool:
        spec = mujoco.MjSpec.from_string(str(xml))
        path = next(
            (item.path for item in self._attached_models if item.model_id == int(model_id)),
            self._root_path,
        )
        if path is not None:
            spec.modelfiledir = str(path.parent)
        return self._replace_model_spec(model_id, spec)

    def model_component_count(self, model_id: int, category: str) -> int:
        spec = self._spec_for_model(model_id)
        collections = {
            "contact": ("pairs", "excludes"),
            "actuator": ("actuators",),
            "sensor": ("sensors",),
            "tendon": ("tendons",),
            "equality": ("equalities",),
        }
        return (
            sum(len(getattr(spec, name)) for name in collections.get(category, ()))
            if spec is not None
            else 0
        )

    def model_components(self, model_id: int, category: str) -> tuple[ModelComponentInfo, ...]:
        if category not in _MODEL_COMPONENT_CATEGORIES:
            return ()
        spec = self._spec_for_model(model_id)
        if spec is None:
            return ()
        root, _xml = _component_xml(spec)
        section = _component_section(root, category)
        if section is None:
            self._component_entries.pop((int(model_id), str(category)), None)
            return ()
        entries = self._sync_model_component_ids(model_id, category, section)
        references = _ComponentReferences(root)
        presets = {}
        components = []
        for entry, element in zip(entries, section, strict=True):
            attributes = dict(element.attrib)
            path = tuple(
                ModelComponentPathItem(
                    child.tag,
                    _component_path_fields(references, category, element.tag, child),
                )
                for child in element
            )
            if element.tag not in presets:
                presets[element.tag] = _component_path_presets(references, category, element.tag)
            components.append(
                ModelComponentInfo(
                    entry.component_id,
                    int(model_id),
                    category,
                    element.tag,
                    attributes.get("name", f"{category}{entry.component_id}"),
                    _component_fields(references, category, element.tag, attributes),
                    path,
                    presets[element.tag],
                )
            )
        return tuple(components)

    def _sync_model_component_ids(
        self,
        model_id: int,
        category: str,
        section: ET.Element,
        forced: dict[int, int] | None = None,
    ) -> tuple[_ModelComponentEntry, ...]:
        """Reconcile stable runtime IDs without ever retargeting a stale ID."""

        key = (int(model_id), str(category))
        previous = self._component_entries.get(key, ())
        descriptors = tuple(
            (
                element.tag,
                str(element.attrib.get("name", "")).strip(),
                ET.tostring(element, encoding="unicode"),
            )
            for element in section
        )
        if len(previous) == len(descriptors) and all(
            (entry.subtype, entry.name, entry.signature) == descriptor
            for entry, descriptor in zip(previous, descriptors, strict=True)
        ):
            return previous

        forced = forced or {}
        used: set[int] = set()
        reconciled = []
        for index, (subtype, name, signature) in enumerate(descriptors):
            component_id = forced.get(index, -1)
            if component_id < 0 and name:
                match = next(
                    (
                        entry
                        for entry in previous
                        if entry.component_id not in used
                        and entry.name == name
                        and (category == "actuator" or entry.subtype == subtype)
                    ),
                    None,
                )
                component_id = match.component_id if match is not None else -1
            if component_id < 0 and not name:
                matches = tuple(
                    entry
                    for entry in previous
                    if entry.component_id not in used
                    and entry.subtype == subtype
                    and not entry.name
                    and entry.signature == signature
                )
                component_id = matches[0].component_id if len(matches) == 1 else -1
            if component_id < 0:
                component_id = self._next_component_id.get(key, 0)
                self._next_component_id[key] = component_id + 1
            used.add(component_id)
            reconciled.append(_ModelComponentEntry(component_id, subtype, name, signature))
        result = tuple(reconciled)
        if result:
            self._component_entries[key] = result
        else:
            self._component_entries.pop(key, None)
        return result

    def _model_component_index(
        self, model_id: int, category: str, section: ET.Element, component_id: int
    ) -> tuple[int, tuple[_ModelComponentEntry, ...]]:
        entries = self._sync_model_component_ids(model_id, category, section)
        index = next(
            (
                index
                for index, entry in enumerate(entries)
                if entry.component_id == int(component_id)
            ),
            -1,
        )
        return index, entries

    def model_component_presets(self, model_id: int, category: str) -> tuple[str, ...]:
        if category not in _MODEL_COMPONENT_CATEGORIES:
            return ()
        spec = self._spec_for_model(model_id)
        if spec is None:
            return ()
        root, _xml = _component_xml(spec)
        return _component_presets(_ComponentReferences(root), category)

    def add_model_component(self, model_id: int, category: str, subtype: str, name: str) -> int:
        spec = self._spec_for_model(model_id)
        value = str(name).strip()
        if spec is None or not value:
            return -1
        root, _xml = _component_xml(spec)
        references = _ComponentReferences(root)
        presets = _component_presets(references, category)
        if subtype not in presets:
            raise ValueError(f"Cannot add {category} subtype {subtype!r} to this model")
        section = _component_section(root, category, create=True)
        assert section is not None
        previous_entries = self._sync_model_component_ids(model_id, category, section)
        if any(
            item.attrib.get("name") == value and (category != "contact" or item.tag == subtype)
            for item in section
        ):
            raise ValueError(f"{category} {value!r} already exists")
        element = ET.SubElement(section, subtype, {"name": value})
        if category == "contact" and subtype == "pair":
            element.set("geom1", references.names("geom")[0])
            element.set("geom2", references.names("geom")[1])
        elif category == "contact":
            element.set("body1", references.names("body")[0])
            element.set("body2", references.names("body")[1])
        elif category == "actuator":
            if subtype == "adhesion":
                element.set("body", references.names("body")[0])
                element.set("ctrlrange", "0 1")
            elif subtype == "orientation":
                element.set("joint", references.ball_joints[0])
            else:
                element.set("joint", references.names("joint")[0])
                if subtype == "damper":
                    element.set("ctrlrange", "0 1")
                elif subtype == "muscle":
                    element.set("lengthrange", "0 1")
                elif subtype == "dcmotor":
                    element.set("motorconst", "1")
                    element.set("resistance", "1")
                    element.set("inductance", "1")
        elif category == "sensor":
            site_sensors = {
                "touch",
                "accelerometer",
                "velocimeter",
                "gyro",
                "force",
                "torque",
                "magnetometer",
                "rangefinder",
                "camprojection",
            }
            if subtype in site_sensors:
                element.set("site", references.names("site")[0])
                if subtype == "camprojection":
                    element.set("camera", references.names("camera")[0])
            elif subtype in {"ballquat", "ballangvel"}:
                element.set("joint", references.ball_joints[0])
            elif subtype.startswith("jointlimit"):
                element.set("joint", references.limited_joints[0])
            elif subtype.startswith("joint"):
                element.set("joint", references.names("joint")[0])
            elif subtype.startswith("tendonlimit"):
                element.set("tendon", references.limited_tendons[0])
            elif subtype.startswith("tendon"):
                element.set("tendon", references.tendons[0])
            elif subtype.startswith("actuator"):
                element.set("actuator", references.names("actuator")[0])
            elif subtype in {"subtreecom", "subtreelinvel", "subtreeangmom"}:
                element.set("body", references.names("body")[0])
            elif subtype == "insidesite":
                element.set("site", references.names("site")[0])
                element.set("objtype", "body")
                element.set("objname", references.names("body")[0])
            elif subtype in {"distance", "normal", "fromto", "contact"}:
                element.set("body1", references.names("body")[0])
                element.set("body2", references.names("body")[1])
            elif subtype == "tactile":
                element.set("mesh", references.names("mesh")[0])
            elif subtype == "user":
                element.set("dim", "1")
                element.set("needstage", "pos")
            elif subtype.startswith("frame"):
                element.set("objtype", "body")
                element.set("objname", references.names("body")[0])
        elif category == "tendon" and subtype == "fixed":
            ET.SubElement(element, "joint", {"joint": references.names("joint")[0], "coef": "1"})
        elif category == "tendon":
            ET.SubElement(element, "site", {"site": references.names("site")[0]})
            ET.SubElement(element, "site", {"site": references.names("site")[1]})
        elif category == "equality" and subtype == "joint":
            element.set("joint1", references.names("joint")[0])
        elif category == "equality" and subtype == "connect":
            if len(references.names("site")) >= 2:
                element.set("site1", references.names("site")[0])
                element.set("site2", references.names("site")[1])
            else:
                element.set("body1", references.names("body")[0])
                element.set("anchor", "0 0 0")
        elif category == "equality" and subtype == "weld":
            element.set("body1", references.names("body")[0])
        elif category == "equality" and subtype == "tendon":
            element.set("tendon1", references.tendons[0])
        elif category == "equality":
            element.set("flex", references.names("flex")[0])
            if subtype == "flexstrain":
                element.set("cell", "0")
        new_spec = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        if not self._replace_model_spec(model_id, new_spec):
            return -1
        entries = self._sync_model_component_ids(
            model_id,
            category,
            section,
            {index: entry.component_id for index, entry in enumerate(previous_entries)},
        )
        return entries[-1].component_id

    def update_model_component(
        self,
        model_id: int,
        category: str,
        component_id: int,
        name: str,
        fields: tuple[tuple[str, str], ...],
        path: tuple[tuple[str, tuple[tuple[str, str], ...]], ...],
    ) -> bool:
        spec = self._spec_for_model(model_id)
        value = str(name).strip()
        if spec is None or not value:
            return False
        root, _xml = _component_xml(spec)
        section = _component_section(root, category)
        if section is None:
            return False
        component_index, _entries = self._model_component_index(
            model_id, category, section, component_id
        )
        if component_index < 0:
            return False
        if any(
            index != component_index and item.attrib.get("name") == value
            for index, item in enumerate(section)
            if category != "contact" or item.tag == section[component_index].tag
        ):
            raise ValueError(f"{category} {value!r} already exists")
        element = section[component_index]
        next_attributes = {
            str(field_name).strip(): str(field_value).strip()
            for field_name, field_value in fields
            if str(field_name).strip()
            and str(field_name).strip() != "name"
            and str(field_value).strip()
        }
        next_attributes = {"name": value, **next_attributes}
        next_path = tuple(
            (
                str(item_kind).strip(),
                tuple(
                    (str(field_name).strip(), str(field_value).strip())
                    for field_name, field_value in item_fields
                    if str(field_name).strip() and str(field_value).strip()
                ),
            )
            for item_kind, item_fields in path
            if str(item_kind).strip()
        )
        current_path = tuple((child.tag, tuple(child.attrib.items())) for child in element)
        path_category = category == "tendon"
        if dict(element.attrib) == next_attributes and (
            not path_category or current_path == next_path
        ):
            return True
        element.attrib.clear()
        element.attrib.update(next_attributes)
        if path_category:
            element[:] = []
            for item_kind, item_fields in next_path:
                ET.SubElement(element, item_kind, dict(item_fields))
        new_spec = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        changed = self._replace_model_spec(model_id, new_spec)
        if changed:
            self._sync_model_component_ids(
                model_id, category, section, {component_index: int(component_id)}
            )
        return changed

    def remove_model_component(self, model_id: int, category: str, component_id: int) -> bool:
        spec = self._spec_for_model(model_id)
        if spec is None:
            return False
        root, _xml = _component_xml(spec)
        section = _component_section(root, category)
        if section is None:
            return False
        component_index, entries = self._model_component_index(
            model_id, category, section, component_id
        )
        if component_index < 0:
            return False
        section.remove(section[component_index])
        if not len(section):
            root.remove(section)
        new_spec = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        changed = self._replace_model_spec(model_id, new_spec)
        if changed:
            remaining = tuple(
                entry for index, entry in enumerate(entries) if index != component_index
            )
            self._sync_model_component_ids(
                model_id,
                category,
                section,
                {index: entry.component_id for index, entry in enumerate(remaining)},
            )
        return changed

    def model_assets(self, model_id: int) -> tuple[ModelAssetInfo, ...]:
        spec = self._spec_for_model(model_id)
        if spec is None:
            return ()
        root, _xml = _component_xml(spec)
        asset = root.find("asset")
        if asset is None:
            return ()
        material_indices: dict[str, int] = {}
        for material_id in range(self._m.nmat):
            compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_MATERIAL, material_id)
            owner, local_name = self._model_element_name(
                compiled_name or f"material{material_id}", mujoco.mjtObj.mjOBJ_MATERIAL
            )
            if owner == int(model_id):
                material_indices[local_name] = material_id
        height_field_previews: dict[
            str,
            tuple[
                tuple[int, int],
                tuple[int, int],
                tuple[float, ...],
                tuple[float, float],
            ],
        ] = {}
        for field_id in range(self._m.nhfield):
            compiled_name = mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_HFIELD, field_id)
            owner, local_name = self._model_element_name(
                compiled_name or f"hfield{field_id}", mujoco.mjtObj.mjOBJ_HFIELD
            )
            if owner != int(model_id):
                continue
            rows = int(self._m.hfield_nrow[field_id])
            columns = int(self._m.hfield_ncol[field_id])
            address = int(self._m.hfield_adr[field_id])
            data = np.asarray(
                self._m.hfield_data[address : address + rows * columns], np.float32
            ).reshape(rows, columns)
            # The inspector submits one ImGui rectangle per preview sample. A 24×24
            # grid already exceeds the visible resolution of the 180 px preview
            # while keeping the immediate-mode draw cost bounded.
            preview_rows = min(rows, 24)
            preview_columns = min(columns, 24)
            row_indices = np.rint(np.linspace(0, rows - 1, preview_rows)).astype(np.intp)
            column_indices = np.rint(np.linspace(0, columns - 1, preview_columns)).astype(np.intp)
            preview = data[np.ix_(row_indices, column_indices)]
            height_field_previews[local_name] = (
                (rows, columns),
                (preview_rows, preview_columns),
                tuple(float(value) for value in preview.reshape(-1)),
                (float(np.min(data)), float(np.max(data))),
            )
        items: list[ModelAssetInfo] = []
        for asset_type in _MODEL_ASSET_TYPES:
            for index, element in enumerate(asset.findall(asset_type)):
                attributes = dict(element.attrib)
                name = str(attributes.get("name", "")).strip()
                if not name:
                    continue
                fields = tuple(
                    ModelComponentField(field, value)
                    for field, value in attributes.items()
                    if field not in {"name", "file"}
                )
                preview = height_field_previews.get(name, ((0, 0), (0, 0), (), (0.0, 0.0)))
                texture_layers: tuple[tuple[str, str], ...] = ()
                if asset_type == "material":
                    layer_map = {
                        str(layer.attrib.get("role", "rgb")): str(layer.attrib.get("texture", ""))
                        for layer in element.findall("layer")
                        if str(layer.attrib.get("texture", ""))
                    }
                    legacy_texture = str(element.attrib.get("texture", ""))
                    if legacy_texture and "rgb" not in layer_map:
                        layer_map["rgb"] = legacy_texture
                    texture_layers = tuple(
                        (role, layer_map[role])
                        for role in MATERIAL_TEXTURE_ROLES
                        if role in layer_map
                    )
                items.append(
                    ModelAssetInfo(
                        model_id=int(model_id),
                        type=asset_type,
                        name=name,
                        index=index,
                        file=str(attributes.get("file", "")),
                        fields=fields,
                        references=_model_asset_references(root, asset_type, name, element),
                        data_shape=preview[0],
                        preview_shape=preview[1],
                        preview_values=preview[2],
                        preview_range=preview[3],
                        runtime_index=(
                            material_indices.get(name, -1) if asset_type == "material" else -1
                        ),
                        texture_layers=texture_layers,
                    )
                )
        return tuple(items)

    def import_model_asset(
        self,
        model_id: int,
        asset_type: str,
        path: Path,
        name: str,
        fields: tuple[tuple[str, str], ...] = (),
    ) -> bool:
        spec = self._spec_for_model(model_id)
        source = Path(path).expanduser().resolve()
        kind = str(asset_type).strip().lower()
        value = str(name).strip()
        if spec is None or not source.is_file() or kind not in ("mesh", "hfield") or not value:
            return False
        root, _xml = _component_xml(spec)
        asset, existing = _model_asset_element(root, kind, value)
        if existing is not None:
            return False
        if asset is None:
            asset = _ensure_model_asset_section(root)
        allowed = set(_MJCF_SCHEMA_ATTRIBUTES.get(("mujoco", "asset", kind), ()))
        attributes = {"name": value, "file": str(source)}
        for raw_name, raw_value in fields:
            field = str(raw_name).strip()
            field_value = str(raw_value).strip()
            if not field or field in {"name", "file"} or field not in allowed:
                raise ValueError(f"Unknown {kind} asset field {field!r}")
            if field_value:
                attributes[field] = field_value
        ET.SubElement(asset, kind, attributes)
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def set_height_field_size(
        self,
        model_id: int,
        name: str,
        size: tuple[float, float, float, float],
    ) -> bool:
        spec = self._spec_for_model(model_id)
        value = str(name).strip()
        try:
            dimensions = np.asarray(size, np.float64).reshape(4)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            spec is None
            or not value
            or not np.all(np.isfinite(dimensions))
            or np.any(dimensions[:3] <= 0.0)
            or dimensions[3] < 0.0
        ):
            return False
        root, _xml = _component_xml(spec)
        _asset, target = _model_asset_element(root, "hfield", value)
        if target is None:
            return False
        target.set("size", _format_mjcf_values(dimensions))
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def rename_model_asset(self, model_id: int, asset_type: str, name: str, new_name: str) -> bool:
        spec = self._spec_for_model(model_id)
        kind = str(asset_type).strip().lower()
        before_name = str(name).strip()
        after_name = str(new_name).strip()
        if spec is None or kind not in _MODEL_ASSET_TYPES or not before_name or not after_name:
            return False
        root, _xml = _component_xml(spec)
        _asset, target = _model_asset_element(root, kind, before_name)
        _asset, collision = _model_asset_element(root, kind, after_name)
        if target is None or collision is not None:
            return False
        target.set("name", after_name)
        for element in root.iter():
            if element is target:
                continue
            for field in _MODEL_ASSET_REFERENCE_FIELDS.get(kind, ()):
                if element.attrib.get(field) == before_name:
                    element.set(field, after_name)
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def duplicate_model_asset(
        self, model_id: int, asset_type: str, name: str, new_name: str
    ) -> bool:
        spec = self._spec_for_model(model_id)
        kind = str(asset_type).strip().lower()
        source_name = str(name).strip()
        duplicate_name = str(new_name).strip()
        if spec is None or kind not in _MODEL_ASSET_TYPES or not source_name or not duplicate_name:
            return False
        root, _xml = _component_xml(spec)
        asset, target = _model_asset_element(root, kind, source_name)
        _asset, collision = _model_asset_element(root, kind, duplicate_name)
        if asset is None or target is None or collision is not None:
            return False
        duplicate = deepcopy(target)
        duplicate.set("name", duplicate_name)
        children = tuple(asset)
        asset.insert(children.index(target) + 1, duplicate)
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def replace_model_asset_file(
        self, model_id: int, asset_type: str, name: str, path: Path
    ) -> bool:
        spec = self._spec_for_model(model_id)
        kind = str(asset_type).strip().lower()
        value = str(name).strip()
        source = Path(path).expanduser().resolve()
        if (
            spec is None
            or kind not in ("mesh", "hfield", "texture")
            or not value
            or not source.is_file()
        ):
            return False
        root, _xml = _component_xml(spec)
        _asset, target = _model_asset_element(root, kind, value)
        if target is None:
            return False
        target.set("file", str(source))
        target.attrib.pop("content_type", None)
        if kind == "hfield":
            target.attrib.pop("nrow", None)
            target.attrib.pop("ncol", None)
            target.attrib.pop("elevation", None)
        elif kind == "mesh":
            for field in ("vertex", "normal", "texcoord", "face", "builtin", "params"):
                target.attrib.pop(field, None)
        else:
            for field in (
                "builtin",
                "width",
                "height",
                "rgb1",
                "rgb2",
                "mark",
                "markrgb",
                "random",
            ):
                target.attrib.pop(field, None)
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)

    def remove_model_asset(self, model_id: int, asset_type: str, name: str) -> bool:
        spec = self._spec_for_model(model_id)
        kind = str(asset_type).strip().lower()
        value = str(name).strip()
        if spec is None or kind not in _MODEL_ASSET_TYPES or not value:
            return False
        root, _xml = _component_xml(spec)
        asset, target = _model_asset_element(root, kind, value)
        if asset is None or target is None:
            return False
        if _model_asset_references(root, kind, value, target):
            return False
        asset.remove(target)
        if not len(asset):
            root.remove(asset)
        edited = self._spec_from_component_xml(model_id, _serialize_component_xml(root))
        return self._replace_model_spec(model_id, edited)
