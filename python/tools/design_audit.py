"""Audit each design field through the real native editor and production painters."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
from imgui_bundle import imgui

from mojive import commands as cmd
from mojive.app.composition import build_workspace
from mojive.config import LayoutConfig, ViewerConfig
from mojive.design.consumer_support import UNAVAILABLE_FIELDS, UNAVAILABLE_PARTS
from mojive.design.editor import MISSING, SpecificationEditor, assign, path_value
from mojive.design.help import get_part_help
from mojive.design.native.workbench import NativeWorkbench
from mojive.design.specification import DEFAULT, component_catalog, resolve_part_value
from mojive.design.store import SpecificationStore, read_spec
from mojive.design.support import native_support
from mojive.tools.design_audit_draw import NativeDrawSamples
from mojive.tools.design_audit_evidence import field_effect_status, identity, instance_effects
from mojive.tools.design_audit_inputs import WorkbenchInput
from mojive.tools.design_audit_recipes import prepare_presentation
from mojive.tools.design_audit_scenes import (
    ProductionScenes,
    configure_joint_example,
    write_audit_model,
)
from mojive.tools.design_audit_viewport import (
    partial_keys,
    recover_instances,
    scroll_positions,
)

GEOMETRY_FIELDS = frozenset(
    (
        "hit",
        "frame",
        "content",
        "outer",
        "layoutRect",
        "spacingRect",
        "textBlockRect",
        "windowRect",
        "marginBox",
        "textRect",
        "titleRect",
        "closeRect",
        "iconRect",
        "fields",
        "numericRect",
        "labelRect",
        "trackRect",
        "thumbRect",
        "destination",
        "uv0",
        "uv1",
    )
)


def changed_value(rule, current):
    """Use a visible, bounded change rather than an imperceptible numeric epsilon."""
    if rule["type"] == "select":
        choices = rule["choices"]
        return choices[(choices.index(current) + 1) % len(choices)]
    if rule["type"] == "color":
        return "#e8b04fff" if current != "#e8b04fff" else "#68b5b7ff"
    step = rule.get("step", 1)
    delta = step if rule.get("max", 1) <= 10 else max(step, 4)
    value = current + delta if current + delta <= rule["max"] else current - delta
    value = max(rule["min"], min(rule["max"], value))
    return int(value) if step >= 1 else round(value, 4)


def stable_record(record):
    return {
        key: value
        for key, value in record.items()
        if key not in ("submittedFrame", "visibleSegments", "visibleHit")
    }


def rectangle_issues(records):
    """Report invalid geometry and content beyond its own component frame."""
    issues = []
    identities = Counter((record.get("windowId"), record["key"]) for record in records)
    issues.extend(
        {"key": key, "windowId": window, "kind": "duplicate-observation-key", "count": count}
        for (window, key), count in identities.items()
        if count > 1
    )
    for record in records:
        hit = record["hit"]
        if not all(np.isfinite(hit)) or min(hit[2:]) <= 0:
            issues.append({"key": record["key"], "kind": "invalid-frame", "rect": hit})
            continue
        x, y, w, h = hit
        for index, field in enumerate(record.get("fields", ())):
            fx, fy, fw, fh = field
            clip = record.get("clip")
            if clip is not None:
                right, bottom = min(fx + fw, clip[2]), min(fy + fh, clip[3])
                fx, fy = max(fx, clip[0]), max(fy, clip[1])
                fw, fh = right - fx, bottom - fy
                if fw <= 0 or fh <= 0:
                    continue
            if fx < x - 1 or fy < y - 1 or fx + fw > x + w + 1 or fy + fh > y + h + 1:
                issues.append(
                    {
                        "key": record["key"],
                        "kind": "child-outside-group",
                        "child": index,
                        "hit": hit,
                        "field": field,
                    }
                )
        content = record.get("content")
        if content is None:
            continue
        cx, cy, cw, ch = content
        if not all(np.isfinite(content)) or min(cw, ch) < 0:
            issues.append({"key": record["key"], "kind": "invalid-content", "rect": content})
        elif cx < x - 1 or cy < y - 1 or cx + cw > x + w + 1 or cy + ch > y + h + 1:
            issues.append(
                {
                    "key": record["key"],
                    "kind": "content-outside-frame",
                    "hit": hit,
                    "content": content,
                }
            )
    return issues


def declared_child(parent, child):
    """Use production semantic identities; spatial overlap alone does not establish ownership."""
    if child.get("parentKey") == parent["key"]:
        return True
    name = parent["key"]
    if name.startswith("panel-header-"):
        return child["key"].startswith("header-" + name.removeprefix("panel-header-") + "-")
    if name == "inspector-header-frame":
        return child["key"] in ("pin", "inspector-close")
    if name == "application-statusbar":
        return child["key"] == "application-status_hints"
    # Numeric producers publish the actual child rectangles in their group record.
    return any(child.get("hit") == field for field in parent.get("fields", ()))


def child_bounds_changes(before, after, *, spatial_candidates=False):
    """Follow children contained by a production parent before the field edit."""
    after_children = {
        (record.get("windowId"), record["key"]): record for record in after["allRecords"]
    }
    after_parents = {(record.get("windowId"), record["key"]): record for record in after["records"]}
    failures = []
    for parent in before["records"]:
        if parent.get("kind") != "container" or not parent.get("content"):
            continue
        key = parent.get("windowId"), parent["key"]
        owned = [
            child
            for child in before["allRecords"]
            if child.get("windowId") == key[0]
            and child["key"] != key[1]
            and (spatial_candidates or declared_child(parent, child))
        ]
        new_parent = after_parents.get(key)
        if new_parent is None or not new_parent.get("content"):
            if owned:
                failures.append({"kind": "parent-removed", "parent": key[1]})
            continue
        px, py, pw, ph = parent["content"]
        nx, ny, nw, nh = new_parent["content"]
        for child in owned:
            child_key = child.get("windowId"), child["key"]
            if child_key == key or child_key[0] != key[0] or child.get("kind") == "container":
                continue
            x, y, w, h = child["hit"]
            if not (px - 1 <= x and py - 1 <= y and x + w <= px + pw + 1 and y + h <= py + ph + 1):
                continue
            moved = after_children.get(child_key)
            if moved is None:
                failures.append(
                    {"kind": "child-not-visible", "parent": key[1], "child": child_key[1]}
                )
                continue
            x, y, w, h = moved["hit"]
            if not (nx - 1 <= x and ny - 1 <= y and x + w <= nx + nw + 1 and y + h <= ny + nh + 1):
                failures.append(
                    {
                        "kind": "child-outside-parent",
                        "parent": key[1],
                        "child": child_key[1],
                        "content": new_parent["content"],
                        "childHit": moved["hit"],
                    }
                )
    return failures


class ParameterAudit:
    def __init__(self, viewer, bench, output, *, screenshots="issues", prior=()):
        self.viewer, self.bench, self.output = viewer, bench, output
        self.runtime = bench.runtime
        self.input = WorkbenchInput(viewer, bench)
        bench.show_bounds = False
        self.screenshots = screenshots
        self.rows = list(prior)
        self.current_field = None
        self.journal = output.with_suffix(".jsonl")
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        if not prior:
            self.journal.write_text("")

    def snapshot(self):
        self.input.neutral()
        ensure_visible = getattr(self, "ensure_visible", None)
        if ensure_visible is not None and not getattr(self, "recovering_visible_instances", False):
            ensure_visible()
        records = [
            deepcopy(record)
            for record in self.bench.native_instances
            if not record["key"].startswith("reference-")
        ]
        role_exclusions = []
        if self.current_field is not None:
            part, field = self.current_field
            consumed = [
                record
                for record in records
                if record.get("consumedFields", {}).get(part) is None
                or field in record["consumedFields"][part]
            ]
            if consumed:
                role_exclusions = [
                    {
                        "instance": identity(record),
                        "consumedFields": record["consumedFields"][part],
                        "reason": "此实际生产角色不消费目标字段。",
                    }
                    for record in records
                    if record not in consumed
                ]
                records = consumed
        # imgui-bundle's OpenGL renderer scales scissors in place during submit.
        # Native bgfx stages its own packet and retains logical draw-data clips.
        clip_scale = (
            tuple(imgui.get_draw_data().framebuffer_scale)
            if self.viewer.backend.caps.name == "opengl"
            else (1.0, 1.0)
        )
        draw_samples = NativeDrawSamples(self.bench, clip_scale=clip_scale)
        instances = {}
        sources = {}
        for record in records:
            key = f"{record.get('windowId', '')}:{record['key']}"
            instances[key], sources[key] = draw_samples.sample(record)
        vertices = (
            np.concatenate([instances[key] for key in sorted(instances)])
            if instances
            else np.empty((0, 6))
        )
        raw_records = self.production_raw_records()
        app, context = self.viewer.app, imgui.get_current_context()
        input_state = app._state
        status_context = {
            "owner": app._status_panel,
            "activeId": context.active_id,
            "navActivateId": context.nav_activate_id,
            "popupOwned": app._popup_owned_frame,
            "gizmoHintVisible": app._gizmo_hint_hover.visible,
            "ctrl": input_state.ctrl,
            "hasSelection": input_state.has_selection,
            "gizmoAvailable": input_state.gizmo_available,
            "selectedNodeId": getattr(self.viewer.session.selected_node, "node_id", None),
            "keycaps": sorted(
                record["key"] for record in raw_records if record["key"].startswith("status-badge-")
            ),
        }
        return {
            "records": [stable_record(record) for record in records],
            "vertices": vertices,
            "instanceVertices": instances,
            "drawSources": sources,
            "issues": rectangle_issues(records),
            "allRecords": self.production_records(),
            "rawRecords": [record for record in raw_records if self.bench.part in record["parts"]],
            "productionScroll": scroll_positions(self.bench),
            "statusContext": status_context,
            "roleExclusions": role_exclusions,
        }

    def settled_snapshot(self):
        """Wait for native preparation and the actual delayed render resize to finish."""
        before = self.snapshot()
        deadline = time.monotonic() + 3.0
        attempt = 0
        settled_attempts = 0
        self.last_settle_trace = []
        while True:
            attempt += 1
            after = self.snapshot()
            difference = self.differences(before, after)
            latch = getattr(self.viewer.window, "latch", None)
            pending_resize = getattr(latch, "_pending", None) is not None
            settled_attempts = 0 if pending_resize else settled_attempts + 1
            self.last_settle_trace.append(
                {"difference": difference, "pendingResize": pending_resize}
            )
            if not pending_resize and not any(
                difference[key]
                for key in ("changedGeometryKeys", "removedKeys", "addedKeys", "drawChanged")
            ):
                return after, True, attempt
            before = after
            if time.monotonic() >= deadline or settled_attempts >= 8:
                return before, False, attempt

    def comparison_snapshot(self):
        """Capture clipped text in actual scroll views for both sides of a comparison."""
        state, stable, frames = self.settled_snapshot()
        recovery = recover_instances(
            self,
            state,
            state,
            {"removedKeys": [], "missingTextKeys": []},
            additional_targets=partial_keys(state),
        )
        state["baselineScrollRecovery"] = recovery
        stable &= all(item.get("visibleBeforeScroll") and item.get("stable") for item in recovery)
        return state, stable, frames

    def production_raw_records(self):
        ui = self.viewer.app.workspace.ui
        primitives = getattr(ui, "design_primitives", None)
        records = list(primitives.observed) if primitives else []
        records.extend(self.runtime.observed)
        return [
            deepcopy(record) for record in records if not record["key"].startswith("reference-")
        ]

    def production_records(self):
        return [
            deepcopy(record)
            for record in self.production_raw_records()
            if self.bench.visible_native_rects(
                record["hit"], record.get("clip", (0, 0, 100000, 100000)), self.bench._viewport_rect
            )
        ]

    @staticmethod
    def differences(before, after):
        def instances(snapshot):
            return {
                f"{record.get('windowId', '')}:{record['key']}": record
                for record in snapshot["records"]
            }

        a, b = instances(before), instances(after)
        common = a.keys() & b.keys()
        changed = [
            key
            for key in common
            if any(
                a[key][field] != b[key][field]
                for field in a[key].keys() & b[key].keys() & GEOMETRY_FIELDS
            )
        ]
        draw_changes = [
            key
            for key in before.get("instanceVertices", {}).keys()
            & after.get("instanceVertices", {}).keys()
            if not np.array_equal(before["instanceVertices"][key], after["instanceVertices"][key])
        ]
        missing_text = [
            key
            for key in before.get("instanceVertices", {}).keys()
            & after.get("instanceVertices", {}).keys()
            if before["instanceVertices"][key].shape[1] > 3
            and np.count_nonzero(before["instanceVertices"][key][:, 3])
            and not np.count_nonzero(after["instanceVertices"][key][:, 3])
            and a.get(key, {}).get("kind") in ("input_number", "axis_input", "input_text", "button")
        ]
        glyph_counts = {
            key: {
                "before": int(np.count_nonzero(before["instanceVertices"][key][:, 3])),
                "after": int(np.count_nonzero(after["instanceVertices"][key][:, 3])),
            }
            for key in before.get("instanceVertices", {}).keys()
            & after.get("instanceVertices", {}).keys()
            if before["instanceVertices"][key].shape[1] > 3
        }
        clipped_numeric_text = [
            key
            for key, counts in glyph_counts.items()
            if counts["after"] < counts["before"]
            and a.get(key, {}).get("kind") in ("input_number", "axis_input")
        ]
        return {
            "instancesBefore": len(a),
            "instancesAfter": len(b),
            "changedGeometryKeys": sorted(changed),
            "changedDrawKeys": sorted(draw_changes),
            "missingTextKeys": sorted(missing_text),
            "clippedNumericTextKeys": sorted(clipped_numeric_text),
            "glyphVertexCounts": glyph_counts,
            "removedKeys": sorted(a.keys() - b.keys()),
            "addedKeys": sorted(b.keys() - a.keys()),
            "drawChanged": not np.array_equal(before["vertices"], after["vertices"]),
            "vertexCountBefore": len(before["vertices"]),
            "vertexCountAfter": len(after["vertices"]),
        }

    def capture(self, part, field, phase, *, include_editor=False):
        from PIL import Image

        future = self.viewer.app.request_capture_async(surface="window", memory=True)
        self.input.frames()
        target = self.output.parent / "captures" / part
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{field}-{phase}.png"
        image = Image.fromarray(future.result(timeout=3)["image"])
        # Preserve the whole production workspace and remove the parameter
        # editor. The exported capture keeps the native screen scale.
        rect = self.bench._viewport_rect
        right = image.width
        if rect and not include_editor:
            right = max(0, min(image.width, round(self.viewer.window.points_to_pixels(rect[0]))))
            if right == 0:
                raise RuntimeError("The production workspace is outside the captured window")
        image.crop((0, 0, right, image.height)).save(path)
        return str(path)

    def write(self, *, checkpoint=False):
        completed = [row for row in self.rows if "status" in row]
        counts = Counter(row["status"] for row in completed)
        report = {
            "method": "Native ImGui input, production instance geometry and submitted draw vertices",
            "fixedTelemetry": {"frameMs": 1000 / 60, "fps": 60, "renderTimingMs": 1},
            "scrollbarExperiment": {
                "clockSeconds": 0,
                "state": "Registered scrollbar owners are held visible for parameter comparisons",
                "dynamicBehavior": "Validated separately through actual idle and panel input GPU tests",
            },
            "scale": self.viewer.window.style_scale,
            "asset": "joint_types with real sensor and spatial-tendon model components",
            "catalogueParts": len(self.runtime.catalog),
            "catalogueFields": sum(len(part["fields"]) for part in self.runtime.catalog.values()),
            "fieldsVisited": len(completed),
            "defaultValueMismatches": list(self.input.default_mismatches.values()),
            "counts": dict(counts),
            "fields": completed if not checkpoint else [],
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        target = (
            self.output.with_name(self.output.stem + "-progress.json")
            if checkpoint
            else self.output
        )
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def record_result(self, row):
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.write(checkpoint=True)

    def discover(self, part):
        self.input.reads.clear()
        self.input.frames(2)
        state = self.snapshot()
        fields = self.runtime.catalog[part]["fields"]
        observed_reads = sorted(field for owner, field in self.input.reads if owner == part)
        return {
            "part": part,
            "label": self.runtime.catalog[part]["label"],
            "instanceCount": len(state["records"]),
            "geometry": state["records"],
            "alignment": state["issues"],
            "readFields": observed_reads,
            "unreadFields": sorted(fields.keys() - set(observed_reads)),
        }

    def field(self, part, field, rule):
        runtime, bench = self.runtime, self.bench
        support = native_support()[part]["fields"][field]
        row = {
            "part": part,
            "field": field,
            "label": rule["label"],
            "area": get_part_help(part)["preview"]["area"],
            "bindingStatus": support["status"],
            "consumers": support["consumers"],
            "condition": rule.get("condition"),
            "nativeCondition": support.get("condition", ""),
            "productionState": getattr(self, "scene_state", {}),
        }
        self.rows = [
            record for record in self.rows if (record["part"], record["field"]) != (part, field)
        ]
        self.rows.append(row)
        self.current_field = part, field
        if part == "workspace_presets":
            row.update(
                status="unavailable",
                reason="当前主界面已移除旧布局预设，工作台禁用此部件的两项参数。",
            )
            self.record_result(row)
            return
        before_spec = deepcopy(bench.editor.spec)
        selection = self.viewer.session.selected_node
        fatal_error = None
        try:
            self.input.hover_part = (
                part
                if field == "radius"
                and part in ("menu", "menu_item", "menu_trigger", "link_button", "tab", "filter")
                else None
            )
            row["presentationContext"] = prepare_presentation(self, part, field)
            if self.input.hover_part is not None:
                row["presentationContext"]["hover"] = part
            condition = rule.get("condition")
            if condition:
                requirement = runtime.catalog[part]["fields"][condition["field"]]
                if runtime.resolved_part(part, condition["field"]) != condition["value"]:
                    self.input.set_field(part, condition["field"], requirement, condition["value"])
            self.input.show_field(field)
            original = runtime.resolved_part(part, field)
            path = ("parts", part, field)
            original_override = path_value(bench.editor.spec, path)
            before, stable, settle_frames = self.comparison_snapshot()
            row["baselineStable"] = stable
            if not stable:
                row["baselineSettleTrace"] = self.last_settle_trace
            row["baselineSettleFrames"] = settle_frames
            row["baselineScrollRecovery"] = before["baselineScrollRecovery"]
            row["originalOverridePresent"] = original_override is not MISSING
            row["baselineObservationValid"] = not any(
                issue["kind"] == "duplicate-observation-key" for issue in before["issues"]
            )
            if self.screenshots == "all":
                row["beforeCapture"] = self.capture(part, field, "before")
            row["baselineFieldRead"] = (part, field) in self.input.reads
            reset_spec = deepcopy(bench.editor.spec)
            assign(reset_spec, path, MISSING)
            expected_reset = resolve_part_value(reset_spec, part, field, catalog=runtime.catalog)
            if path_value(bench.editor.spec, path) is not MISSING:
                self.input.restore(part, field)
            if (
                path_value(bench.editor.spec, path) is not MISSING
                or runtime.resolved_part(part, field) != expected_reset
            ):
                row.update(
                    status="reset-failure",
                    reason="建立默认基准时恢复按钮未删除覆盖或未恢复继承值。",
                    resetVerified=False,
                    expectedResetValue=expected_reset,
                    resetButtonValue=runtime.resolved_part(part, field),
                    editorError=bench.editor.error,
                    editorActions=bench.actions[-6:],
                    remainingOverrides=bench.editor.spec.get("parts", {}).get(part, {}),
                )
                row["editorCapture"] = self.capture(
                    part, field, "reset-editor", include_editor=True
                )
                return
            reset_baseline, reset_stable, reset_settle_frames = self.comparison_snapshot()
            if original_override is not MISSING:
                self.input.set_field(part, field, rule, original)
            row["expectedResetValue"] = expected_reset
            row["resetBaselineStable"] = reset_stable
            if not reset_stable:
                row["resetBaselineSettleTrace"] = self.last_settle_trace
            row["resetBaselineSettleFrames"] = reset_settle_frames
            changed = changed_value(rule, original)
            if field == "fontFamily":
                effective = (
                    original
                    if original != "inherit"
                    else runtime.value(
                        "font_family",
                        before["records"][0].get("styleFamily") if before["records"] else None,
                    )
                )
                changed = "system" if effective == "portable" else "portable"
            if part == "color_input" and field == "width":
                minimum_text = max(
                    (
                        record.get("minimumTextWidth", 0) / self.viewer.window.style_scale
                        for record in before["records"]
                    ),
                    default=0,
                )
                changed = min(rule["max"], max(changed, 120, minimum_text + 4))
                row["presentationContext"]["widthPreservesHexText"] = changed
            if field == "lineHeight" and rule["type"] == "number":
                changed = min(rule["max"], max(changed, 40))
                row["presentationContext"]["lineHeightAboveMinimumControl"] = 40
            if field == "height" and rule["type"] == "number" and before["records"]:
                minimum_visible = max(
                    record["hit"][3] / self.viewer.window.style_scale
                    for record in before["records"]
                )
                candidate = minimum_visible + max(rule.get("step", 1), 4)
                if original < candidate <= rule["max"]:
                    changed = int(candidate) if rule.get("step", 1) >= 1 else round(candidate, 4)
                    row["presentationContext"]["heightAboveActualInstances"] = minimum_visible
            row.update({"beforeValue": original, "changedValue": changed})
            self.input.reads.clear()
            self.input.set_field(part, field, rule, changed)
            after, after_stable, after_settle_frames = self.comparison_snapshot()
            row["afterStable"] = after_stable
            row["afterSettleFrames"] = after_settle_frames
            trials = [{"value": changed, "effect": self.differences(before, after)}]
            if before["records"] and rule["type"] == "number":
                for extreme in (rule["min"], rule["max"]):
                    effect = trials[-1]["effect"]
                    recovery = recover_instances(
                        self, before, after, effect, additional_targets=partial_keys(after)
                    )
                    row.setdefault("trialScrollRecovery", []).append(
                        {"value": changed, "instances": recovery}
                    )
                    effect = self.differences(before, after)
                    trials[-1]["effect"] = effect
                    applicable = instance_effects(
                        before,
                        after,
                        part,
                        field,
                        bench.editor.spec,
                        self.viewer.window.style_scale,
                        GEOMETRY_FIELDS,
                        effect,
                    )
                    if field_effect_status(applicable)[0] in ("observed", "inactive"):
                        break
                    if extreme in (original, changed):
                        continue
                    self.input.set_field(part, field, rule, extreme)
                    changed = extreme
                    after, row["afterStable"], row["afterSettleFrames"] = self.comparison_snapshot()
                    trials.append({"value": extreme, "effect": self.differences(before, after)})
            row["trials"] = trials
            row["changedValue"] = changed
            row["changedFieldRead"] = (part, field) in self.input.reads
            row["editCommitted"] = runtime.resolved_part(part, field) == changed
            row["effect"] = self.differences(before, after)
            row["scrollRecovery"] = recover_instances(
                self, before, after, row["effect"], additional_targets=partial_keys(after)
            )
            row["effect"] = self.differences(before, after)
            row["instanceEffects"] = instance_effects(
                before,
                after,
                part,
                field,
                bench.editor.spec,
                self.viewer.window.style_scale,
                GEOMETRY_FIELDS,
                row["effect"],
            )
            row["drawSourceErrors"] = {
                phase: {
                    key: source["sourceError"]
                    for key, source in state["drawSources"].items()
                    if "sourceError" in source
                }
                for phase, state in (("before", before), ("after", after))
            }
            row["alignmentBefore"] = before["issues"]
            row["alignmentAfter"] = after["issues"]
            row["childAlignmentAfter"] = child_bounds_changes(before, after)
            row["spatialAlignmentCandidates"] = child_bounds_changes(
                before, after, spatial_candidates=True
            )
            row["geometryBefore"] = before["records"]
            row["geometryAfter"] = after["records"]
            row["statusContextBefore"] = before["statusContext"]
            row["statusContextAfter"] = after["statusContext"]
            row["roleExclusions"] = before["roleExclusions"]
            if not before["records"] or not after["records"]:
                row["geometrySubmittedBefore"] = before["rawRecords"]
                row["geometrySubmittedAfter"] = after["rawRecords"]
            provisional_status, _ = field_effect_status(row["instanceEffects"])
            alignment_changed = (
                after["issues"] != before["issues"]
                or row["childAlignmentAfter"]
                or row["effect"]["missingTextKeys"]
                or row["effect"]["clippedNumericTextKeys"]
            )
            if self.screenshots == "all" or (
                self.screenshots == "issues"
                and (provisional_status != "observed" or alignment_changed)
            ):
                row["afterCapture"] = self.capture(part, field, "after")
            self.input.restore(part, field)
            row["resetButtonValue"] = runtime.resolved_part(part, field)
            row["resetVerified"] = (
                path_value(bench.editor.spec, path) is MISSING
                and row["resetButtonValue"] == expected_reset
            )
            reset_state, row["resetStable"], row["resetSettleFrames"] = self.comparison_snapshot()
            row["resetRestoration"] = self.differences(reset_baseline, reset_state)
            row["geometryReset"] = reset_state["records"]
            row["statusContextReset"] = reset_state["statusContext"]
            row["resetVisualRestored"] = (
                row["resetVerified"]
                and not row["resetRestoration"]["changedGeometryKeys"]
                and not row["resetRestoration"]["removedKeys"]
                and not row["resetRestoration"]["addedKeys"]
                and row["resetBaselineStable"]
                and not row["resetRestoration"]["drawChanged"]
            )
            # A failed reset remains visible in the report; never overwrite it
            # with a later hand-entered original value.
            if (
                row["resetVerified"]
                and row["resetVisualRestored"]
                and original_override is not MISSING
            ):
                self.input.set_field(part, field, rule, original)
                restored, row["restoredStable"], row["restoredSettleFrames"] = (
                    self.comparison_snapshot()
                )
            else:
                restored = reset_state
            row["restoredValue"] = runtime.resolved_part(part, field)
            row["restoration"] = self.differences(before, restored)
            row["geometryRestored"] = restored["records"]
            row["visualRestored"] = (
                row["restoredValue"] == original
                and not row["restoration"]["changedGeometryKeys"]
                and not row["restoration"]["removedKeys"]
                and not row["restoration"]["addedKeys"]
                and row["baselineStable"]
                and not row["restoration"]["drawChanged"]
            )
            row["alignmentRestored"] = restored["issues"]
            unavailable = UNAVAILABLE_FIELDS.get((part, field)) or UNAVAILABLE_PARTS.get(part)
            if not before["records"]:
                row.update(
                    status="unavailable" if unavailable else "absent",
                    reason=unavailable or "定位后没有此类型的可见生产组件绘制记录。",
                )
            elif not row["baselineObservationValid"]:
                row.update(
                    status="observation-error",
                    reason="同窗口多个实例共用观测标识，无法逐个对应修改与恢复。",
                )
            elif any(row["drawSourceErrors"].values()):
                row.update(
                    status="observation-error", reason="无法把生产记录对应到其实际提交的绘制源。"
                )
            elif not row["resetVerified"] or not row["resetVisualRestored"]:
                row.update(status="reset-failure", reason="恢复按钮未恢复继承值及其实际默认绘制。")
            elif row["baselineStable"] and not row["visualRestored"]:
                row.update(
                    status="restore-failure", reason="恢复后生产组件的几何或字段值与修改前不一致。"
                )
            elif not row["baselineStable"]:
                row.update(
                    status="unstable-baseline",
                    reason="未修改参数的连续帧绘制不同；需要排除动态文字或动画。",
                )
            else:
                status, reason = field_effect_status(row["instanceEffects"])
                row.update(
                    status="unavailable" if unavailable else status, reason=unavailable or reason
                )
            if row["status"] not in ("reset-failure", "restore-failure", "observation-error") and (
                after["issues"] != before["issues"]
                or row["childAlignmentAfter"]
                or row["effect"]["missingTextKeys"]
                or row["effect"]["clippedNumericTextKeys"]
            ):
                row.update(status="alignment-failure", reason="调整引入组件内容越界或无效尺寸。")
            if self.screenshots == "all":
                row["restoredCapture"] = self.capture(part, field, "restored")
            elif self.screenshots == "issues" and row["status"] != "observed":
                row["capture"] = self.capture(part, field, "restored")
        except Exception as error:
            interrupted = imgui.get_current_context().within_frame_scope
            row.update(
                status="native-frame-error" if interrupted else "editor-error",
                reason=f"{type(error).__name__}: {error}",
                frameInterrupted=interrupted,
                traceback=traceback.format_exc(),
            )
            if interrupted:
                fatal_error = error
            else:
                self.input.key(imgui.Key.escape)
                if (
                    selection is not None
                    and self.viewer.session.node(selection.node_id) is not None
                ):
                    self.viewer.session.submit(cmd.SelectNode(selection.node_id))
                    self.input.frames(2)
        finally:
            # Recovery after a failed gesture belongs to the audit's isolated
            # specification. The report retains the failed native operation.
            if bench.editor.spec != before_spec:
                bench.editor.store.write(before_spec)
                bench.editor.poll(force=True)
                runtime.poll(force=True)
                if fatal_error is None:
                    self.input.frames(3)
            self.input.hover_part = None
            self.current_field = None
            self.record_result(row)
        if fatal_error is not None:
            raise RuntimeError(
                "Native frame failed; resume in a fresh process after repair"
            ) from fatal_error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", help="Comma-separated catalogue IDs; default visits all parts")
    parser.add_argument("--fields", help="Comma-separated field IDs")
    parser.add_argument("--cases", help="Comma-separated exact part.field pairs")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resume", action="store_true", help="Retain existing results and skip visited fields"
    )
    parser.add_argument(
        "--retry", help="Revisit only these comma-separated statuses in an existing report"
    )
    parser.add_argument("--backend", choices=("opengl", "bgfx"), default="opengl")
    parser.add_argument("--scale", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--screenshots", choices=("all", "issues", "none"), default="issues")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument(
        "--discover", action="store_true", help="List actual production instances without edits"
    )
    parser.add_argument(
        "--pick", action="store_true", help="Alt-click every visible production instance"
    )
    args = parser.parse_args(argv)
    output = args.output or Path(f"output/design-parameter-audit/{args.scale}x/audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MOJIVE_UI_SCALE"] = str(args.scale)
    os.environ["MOJIVE_SETTINGS"] = str(output.parent / "settings.json")
    store = SpecificationStore(output.parent / "current-spec.json")
    store.write(DEFAULT, save=True)
    os.environ["MOJIVE_DESIGN_SPEC"] = str(store.saved_path)
    catalog = component_catalog()
    editor = SpecificationEditor(store.saved_path, store.live_path, read_spec)
    requested_parts = set(args.parts.split(",")) if args.parts else set(catalog)
    requested_fields = set(args.fields.split(",")) if args.fields else None
    unknown = requested_parts - catalog.keys()
    if unknown:
        parser.error("Unknown catalogue parts: " + ", ".join(sorted(unknown)))
    requested_cases = (
        {tuple(case.rsplit(".", 1)) for case in args.cases.split(",")} if args.cases else None
    )
    available_cases = {
        (part, field) for part, definition in catalog.items() for field in definition["fields"]
    }
    if requested_cases and requested_cases - available_cases:
        parser.error("Unknown catalogue cases: " + repr(sorted(requested_cases - available_cases)))
    fields = [
        (part, field, rule)
        for part, definition in component_catalog().items()
        if part in requested_parts
        for field, rule in definition["fields"].items()
        if requested_fields is None or field in requested_fields
        if requested_cases is None or (part, field) in requested_cases
    ]
    fields = fields[args.start : None if args.limit is None else args.start + args.limit]
    prior = []
    if args.resume or args.retry:
        if output.exists():
            previous = json.loads(output.read_text())
            if previous["scale"] != args.scale:
                parser.error("The existing report uses a different UI scale")
            prior = previous["fields"]
        journal = output.with_suffix(".jsonl")
        if journal.exists():
            merged = {(row["part"], row["field"]): row for row in prior}
            for line in journal.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A terminated process may leave its final append incomplete.
                    continue
                merged[row["part"], row["field"]] = row
            prior = list(merged.values())
        prior = [row for row in prior if "status" in row]
        results = {(row["part"], row["field"]): row for row in prior}
        retry = set(args.retry.split(",")) if args.retry else set()
        fields = [
            item
            for item in fields
            if (item[0], item[1]) not in results or results[item[0], item[1]]["status"] in retry
        ]
    asset = write_audit_model(output.parent / "audit_model.xml")
    with build_workspace(
        asset,
        width=1920 if args.scale == 1 else 3200,
        height=1080 if args.scale == 1 else 1800,
        show_window=args.visible,
        vsync=False,
        renderer=args.backend,
        config=ViewerConfig(
            layout=LayoutConfig(persistence=False), threaded_physics=False, debug_server=False
        ),
    ) as viewer:
        runtime = viewer.app.design
        bench = runtime.workbench = NativeWorkbench(editor, runtime, runtime)
        viewer.app.set_language("zh_CN")
        configure_joint_example(viewer)
        audit = ParameterAudit(viewer, bench, output, screenshots=args.screenshots, prior=prior)
        scenes = ProductionScenes(audit, asset)
        audit.input.frames(8)
        current = None
        started = time.monotonic()
        discovery = []
        try:
            if args.discover or args.pick:
                from mojive.tools.design_audit_picking import audit_picking

                for part in catalog:
                    if part not in requested_parts:
                        continue
                    bench.part = part
                    scenes.navigate(part)
                    discovery.append(
                        audit_picking(audit, part) if args.pick else audit.discover(part)
                    )
                    print(
                        f"{part}: {discovery[-1]['instanceCount']} production instances",
                        flush=True,
                    )
                return int(any(row.get("status") == "pick-failure" for row in discovery))
            for index, (part, field, rule) in enumerate(fields):
                if current != part:
                    current = bench.part = part
                    scenes.navigate(part)
                scenes.prepare_field(part, field)
                audit.field(part, field, rule)
                row = audit.rows[-1]
                print(
                    f"{index + 1}/{len(fields)} {part}.{field}: {row['status']} ({time.monotonic() - started:.1f}s)",
                    flush=True,
                )
        finally:
            audit.input.close()
            if args.discover or args.pick:
                output.write_text(
                    json.dumps(
                        {"scale": args.scale, "parts": discovery}, ensure_ascii=False, indent=2
                    )
                    + "\n"
                )
            else:
                audit.write()
    return int(
        any(row["status"] not in ("observed", "inactive", "unavailable") for row in audit.rows)
    )


if __name__ == "__main__":
    raise SystemExit(main())
