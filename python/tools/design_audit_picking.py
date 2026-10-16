"""Exercise inspection clicks against visible production instances and their parents."""

from copy import deepcopy

from imgui_bundle import imgui


def interaction_state(viewer):
    app, workspace, session = viewer.app, viewer.app.workspace, viewer.session
    node = session.selected_node
    return {
        "playing": workspace.playing,
        "selected": node.node_id if node else None,
        "panels": dict(workspace.open),
        "inspectorTab": workspace.inspector_tab,
        "recording": session.state_take_recording,
        "language": str(app.localizer.language),
        "popups": len(imgui.get_current_context().open_popup_stack),
        "tool": app.tool,
        "space": workspace.space,
        "snap": workspace.snap,
        "camera": {
            key: getattr(app.camera, key)
            for key in ("yaw", "pitch", "distance", "fov_y_deg", "orthographic")
        },
    }


def audit_picking(audit, part):
    bench, inputs = audit.bench, audit.input
    audit.ensure_visible()
    inputs.frames(3)
    records = deepcopy(
        [record for record in bench.native_instances if not record["key"].startswith("reference-")]
    )
    results = []
    for record in records:
        x, y, width, height = record["visibleHit"]
        inputs.event("add_mouse_pos_event", x + width / 2, y + height / 2)
        inputs.frames(2)
        before = interaction_state(audit.viewer)
        inputs.event("add_key_event", imgui.Key.mod_alt, True)
        inputs.event("add_mouse_button_event", 0, True)
        selected = list(bench.selection_parts)
        inputs.event("add_key_event", imgui.Key.mod_alt, False)
        inputs.event("add_mouse_button_event", 0, False)
        inputs.frames(2)
        after = interaction_state(audit.viewer)
        row = {
            "key": record["key"],
            "kind": record["kind"],
            "hit": record["visibleHit"],
            "selectedParts": selected,
            "status": "observed" if part in selected and before == after else "pick-failure",
        }
        if before != after:
            row.update(before=before, after=after)
        if row["status"] != "observed":
            row["capture"] = audit.capture(part, str(len(results)), "pick")
        results.append(row)
    return {
        "part": part,
        "instanceCount": len(records),
        "status": "observed"
        if results and all(r["status"] == "observed" for r in results)
        else "pick-failure",
        "instances": results,
    }
