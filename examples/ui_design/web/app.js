import { createViewport } from "./scene.js";
import { icon } from "./icons.js";
import {
  History,
  initialDocument,
  clone,
  clamp,
  JOINTS,
  samplePose,
  numericValue,
  uniqueName,
  upsertKey,
  groupKeys,
} from "./model.mjs";
const $ = (q, root = document) => root.querySelector(q);
const esc = (v) =>
  String(v ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
let history = new History(initialDocument());
const panels = [
  ["Scene", "scene"],
  ["Pose", "pose"],
  ["Control", "control"],
  ["Assets", "assets"],
  ["Cameras", "camera"],
  ["Sensors", "chart"],
  ["Layers", "layers"],
  ["Output", "terminal"],
  ["Statistics", "chart"],
];
const state = {
  panel: "Scene",
  workspace: "Edit",
  tab: "Properties",
  selected: "hinge_body",
  pinned: null,
  tool: "rotate",
  frame: "World",
  snap: false,
  playing: false,
  time: 4,
  speed: 1,
  loop: true,
  previewPose: null,
  query: "",
  filter: "All",
  worldOpen: true,
  finder: true,
  inspector: true,
  drawer: null,
  bottom: "Timeline",
  collapsed: true,
  ortho: false,
  cameraLabel: "Perspective",
  grid: false,
  checker: true,
  shadows: true,
  outlines: true,
  gizmos: true,
  navOpacity: 0.55,
  lightIntensity: 3,
  shading: "Solid",
  layers: { link: true, geom: true, light: false, camera: false, floor: true },
  font: 12,
  selectedKey: null,
  sensor: "hinge",
  bookmarks: [],
  log: [
    {
      time: "00:00",
      level: "info",
      text: "Joint study opened. Local authoring is ready.",
    },
  ],
  material: "sage",
};
Object.defineProperties(state, {
  entities: { get: () => history.document.entities },
  pose: { get: () => state.previewPose || history.document.pose },
});
const materials = [
  { id: "sage", name: "Sage ceramic", color: "#9cbf8d", roughness: 0.65 },
  { id: "amber", name: "Amber satin", color: "#e3bc67", roughness: 0.45 },
  { id: "slate", name: "Slate matte", color: "#829cb7", roughness: 0.9 },
  { id: "coral", name: "Coral polymer", color: "#d58979", roughness: 0.35 },
];
const button = (action, label, ico = "", cls = "", attrs = "") =>
  `<button type="button" data-action="${esc(action)}" ${label ? `aria-label="${esc(label.replace(/<[^>]*>/g, ""))}"` : ""} class="button ${cls}" ${attrs}>${ico ? icon(ico) : ""}${label ? `<span>${label}</span>` : ""}</button>`;
const ib = (action, ico, label, active = false, attrs = "") =>
  button(
    action,
    "",
    ico,
    `icon-button ${active ? "active" : ""}`,
    `aria-label="${esc(label)}" title="${esc(label)}" ${active ? 'aria-pressed="true"' : ""} ${attrs}`,
  );
const disabled = (reason) => `disabled title="${esc(reason)}"`;
const notice = (text, kind = "info") =>
  `<div class="notice ${kind}">${icon(kind)}<p>${text}</p></div>`;
const empty = (title, text, action = "") =>
  `<div class="empty">${icon("box")}<strong>${title}</strong><p>${text}</p>${action}</div>`;
const section = (id, title, html, open = true) =>
  `<details data-section="${id}" ${open ? "open" : ""}><summary>${icon("right")}<span>${title}</span></summary><div class="section-body">${html}</div></details>`;
const field = (label, key, value, unit = "", opts = {}) =>
  `<label class="field-wrap"><span class="field-label">${label}</span><span class="input-shell"><input aria-label="${esc(opts.aria || label)}" data-bind="${key}" data-target="${opts.target || ""}" type="number" value="${Number(value.toFixed(3))}" step="${opts.step ?? 0.01}" min="${opts.min ?? -10000}" max="${opts.max ?? 10000}" ${opts.disabled ? "disabled" : ""}>${unit ? `<span class="unit">${unit}</span>` : ""}</span></label>`;
const select = (label, key, values, value) =>
  `<label class="field-wrap"><span class="field-label">${label}</span><select aria-label="${label}" data-choice="${key}">${values.map((v) => `<option ${v === value ? "selected" : ""}>${v}</option>`).join("")}</select></label>`;
const toggle = (key, label, value, hint = "", attrs = "") =>
  `<label class="switch-row"><span><span>${label}</span>${hint ? `<small>${hint}</small>` : ""}</span><input type="checkbox" role="switch" aria-label="${label}" data-toggle="${key}" ${value ? "checked" : ""} ${attrs}><span class="switch" aria-hidden="true"></span></label>`;
const entity = (id) => state.entities.find((e) => e.id === id);
const inspected = () => entity(state.pinned || state.selected);
const now = () => new Date().toLocaleTimeString("en-GB", { hour12: false });
let viewport,
  toastTimer,
  dialogOpener,
  popoverOpener,
  lastTick = 0,
  dragStart = null,
  renderPending = false;
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $("#toast").classList.remove("visible"), 3600);
}
function log(message, level = "info") {
  state.log.push({ time: now(), level, text: message });
  if (state.log.length > 100) state.log.shift();
}
function buttonState(action, active) {
  document.querySelectorAll(`[data-action="${action}"]`).forEach((b) => {
    b.classList.toggle("active", active);
    b.setAttribute("aria-pressed", String(active));
  });
}
function transact(label, fn, { rebuild = false } = {}) {
  const preview = state.previewPose;
  if (
    history.commit(label, (d) => {
      if (preview) d.pose = clone(preview);
      fn(d);
    })
  ) {
    state.previewPose = null;
    state.playing = false;
    log(label);
    repairSelection();
    if (rebuild) viewport.rebuild();
    else viewport.update();
    render();
  }
}
function repairSelection() {
  if (!entity(state.selected)) state.selected = null;
  if (!entity(state.pinned)) state.pinned = null;
  if (!history.document.keys.some((k) => k.id === state.selectedKey))
    state.selectedKey = null;
}
function historyAction(redo = false) {
  const label = redo ? history.redo() : history.undo();
  if (!label) return;
  state.previewPose = null;
  state.playing = false;
  repairSelection();
  viewport.rebuild();
  render();
  toast(`${redo ? "Redid" : "Undid"} ${label.toLowerCase()}`);
}
function replaceHTML(node, html) {
  const active = document.activeElement,
    inside = node.contains(active),
    key = inside
      ? active.dataset.bind
        ? `[data-bind="${active.dataset.bind}"][data-target="${active.dataset.target}"]`
        : active.dataset.query
          ? "[data-query]"
          : active.dataset.action
            ? `[data-action="${active.dataset.action}"]`
            : null
      : null;
  const caret =
    inside && ["text", "search"].includes(active.type)
      ? active.selectionStart
      : null;
  const open = new Map(
    [...node.querySelectorAll("details")].map((d) => [
      d.dataset.section,
      d.open,
    ]),
  );
  const scrolls = [...node.querySelectorAll("[data-scroll]")].map((e) => [
    e.dataset.scroll,
    e.scrollTop,
  ]);
  node.innerHTML = html;
  node.querySelectorAll("details").forEach((d) => {
    if (open.has(d.dataset.section)) d.open = open.get(d.dataset.section);
  });
  for (const [id, top] of scrolls) {
    const el = node.querySelector(`[data-scroll="${id}"]`);
    if (el) el.scrollTop = top;
  }
  if (key) {
    const next = node.querySelector(key);
    next?.focus({ preventScroll: true });
    if (caret !== null) next?.setSelectionRange(caret, caret);
  }
}
$("#root").innerHTML = `
 <header class="titlebar" id="titlebar"></header>
 <main class="workspace" id="workspace">
  <nav class="rail" aria-label="Workspace panels"></nav>
  <aside class="finder panel" id="finder" aria-label="Panel browser"></aside>
  <div class="splitter left-split" role="separator" aria-label="Resize panel browser" aria-orientation="vertical" tabindex="0" data-resize="left" aria-valuemin="240" aria-valuemax="380" aria-valuenow="272"></div>
  <section class="stage" aria-label="Scene workspace">
   <div class="viewport" id="viewport">
    <div class="viewport-toolbar" id="viewport-toolbar"></div>
    <section class="transport glass" id="transport" aria-label="Playback"></section>
    <div class="toolbox glass" role="toolbar" aria-label="Transform tools"></div>
    <div class="nav-widget" aria-label="View orientation"></div>
    <div class="view-caption" hidden><span class="view-name">Perspective</span><span class="view-context">Joint study</span></div>
    <div class="play-state glass" hidden></div><div class="viewport-message" hidden></div>
    <div class="viewport-footer" id="viewport-footer"></div>
   </div>
   <section class="bottom-panel" id="bottom-panel" aria-label="Timeline and signals"></section>
  </section>
  <div class="splitter right-split" role="separator" aria-label="Resize inspector" aria-orientation="vertical" tabindex="0" data-resize="right" aria-valuemin="290" aria-valuemax="420" aria-valuenow="320"></div>
  <aside class="inspector panel" id="inspector" aria-label="Inspector"></aside>
  <button class="drawer-scrim" aria-label="Close side panel" data-action="drawer-close" hidden></button>
 </main>
 <footer class="statusbar" id="statusbar"></footer>`;
function renderTitle() {
  replaceHTML(
    $("#titlebar"),
    `<a class="brand" href="./" aria-label="Instrument design study" title="Instrument design study"></a><nav class="menus" aria-label="Application menus">${["File", "Edit", "Add", "View", "Simulate", "Window", "Help", "More"].map((n) => button("menu:" + n, n, "", "quiet menu-trigger", 'aria-haspopup="menu" aria-expanded="false"')).join("")}</nav><div class="document-title">${icon("file")}<span>${esc(history.document.name)}</span>${history.dirty ? '<span class="dirty-dot unsaved" title="Unsaved changes"></span>' : ""}<small>· Scene preview</small></div><div class="workspace-presets" role="group" aria-label="Workspace layout">${["Edit", "Author", "Review"].map((n) => button("workspace:" + n, n, "", state.workspace === n ? "active" : "", `aria-pressed="${state.workspace === n}"`)).join("")}</div><div class="title-actions">${ib("toggle-inspector", "inspect", "Toggle inspector", (innerWidth >= (1200 * state.font) / 12 && state.inspector) || state.drawer === "inspector")}${ib("preferences", "settings", "Preferences")}</div>`,
  );
}
function panelHead(title, subtitle = "", actions = "") {
  return `<div class="panel-head"><div><strong>${title}</strong>${subtitle ? `<small>${subtitle}</small>` : ""}</div><span class="spacer"></span>${actions}${ib("close-panel", "close", "Close " + title, false)}</div>`;
}
function search(placeholder) {
  return `<div class="search-field">${icon("search")}<input type="search" data-query="true" aria-label="${placeholder}" placeholder="${placeholder}" value="${esc(state.query)}">${state.query ? ib("clear-search", "close", "Clear search") : ""}</div>`;
}
function scenePanel() {
  const result = state.entities.filter(
    (e) =>
      (state.filter === "All" || e.type === state.filter.toLowerCase()) &&
      e.name.toLowerCase().includes(state.query.toLowerCase()),
  );
  return (
    panelHead(
      "Scene",
      `${state.entities.length} entities`,
      ib("menu:Add", "plus", "Add entity"),
    ) +
    `<div class="panel-tools">${search("Search scene")}<div class="filter-tabs" role="group" aria-label="Entity filter">${["All", "Link", "Geom", "Light", "Camera"].map((n) => button("filter:" + n, n, "", state.filter === n ? "active" : "", `aria-pressed="${state.filter === n}"`)).join("")}</div></div><div class="panel-scroll tree" data-scroll="scene"><div class="tree-root">${ib("world-toggle", state.worldOpen ? "down" : "right", state.worldOpen ? "Collapse world" : "Expand world")}<span>${icon("world")}world</span><span class="count">${result.length}</span></div>${state.worldOpen ? result.map((e) => `<div class="tree-row ${state.selected === e.id ? "selected" : ""} ${e.hidden ? "muted" : ""}"><button class="entity-row type-${e.type}" data-action="select:${e.id}" aria-pressed="${state.selected === e.id}" title="${esc(e.name)}">${icon(e.type)}<span class="truncate">${esc(e.name)}</span><small>${e.type}</small></button>${ib("visible:" + e.id, e.hidden ? "eyeoff" : "eye", `${e.hidden ? "Show" : "Hide"} ${e.name}`)}</div>`).join("") || empty("No matching entities", "Try another name or clear the filter.", button("clear-filter", "Clear filters")) : ""}</div><div class="panel-foot">${state.selected ? "1 selected" : "No selection"}<span class="spacer"></span>${button("frame-all", "Frame all", "frame", "quiet")}</div>`
  );
}
function jointControl(j) {
  const available = state.entities.some((e) => e.id === j.name);
  return `<div class="joint-control ${!available ? "muted" : ""}"><div class="joint-title"><span>${j.label}</span><small>${j.unit}</small></div><div class="range-number"><input type="range" aria-label="${j.label} slider" data-pose-range="${j.key}" min="${j.min}" max="${j.max}" step="${j.step}" value="${state.pose[j.key]}" ${!available || state.playing ? "disabled" : ""}>${field(j.label, `pose.${j.key}`, state.pose[j.key], j.unit, { aria: j.label + " value", min: j.min, max: j.max, step: j.step, disabled: !available || state.playing })}</div><div class="range-bounds"><span>${j.min}</span><span>${j.max}</span></div></div>`;
}
function posePanel() {
  return (
    panelHead("Pose", "Joint coordinates") +
    `<div class="panel-scroll" data-scroll="pose">${notice(state.playing ? "Playback is active. Pause to edit the current pose." : "Edit joint coordinates, then capture a key at the playhead.")}<div class="padded">${JOINTS.map(jointControl).join("")}<div class="button-row">${button("pose-rest", "Rest pose", "reset", "", state.playing ? "disabled" : "")}${button("add-key", "Capture key", "addkey", "primary", state.playing ? "disabled" : "")}</div></div>${section("pose-list", "Pose keys", history.document.keys.map((k) => `<div class="list-row">${button("key:" + k.id, esc(k.name), "key", "quiet")}<span class="spacer"></span><span class="mono">${k.time.toFixed(2)} s</span></div>`).join("") || empty("No keys", "Capture a pose to begin."))}</div>`
  );
}
function controlPanel() {
  return (
    panelHead("Control", "Simulation adapter") +
    `<div class="panel-scroll">${notice("No physics adapter connected. Actuator commands require a live simulation.")}<div class="padded"><span class="eyebrow">ACTUATORS</span><fieldset disabled><legend class="sr-only">Unavailable actuator controls</legend>${["Shoulder motor", "Elbow motor", "Gripper motor"].map((n) => field(n, n, 0, "N·m", { disabled: true })).join("")}${button("none", "Reset controls", "reset", "", disabled("Connect a physics adapter to send commands."))}</fieldset></div>${section("capabilities", "Capabilities", `<div class="stat-row"><span>Scene authoring</span><span class="tag good">Available</span></div><div class="stat-row"><span>Joint pose preview</span><span class="tag good">Available</span></div><div class="stat-row"><span>Physics write-back</span><span class="tag">Disconnected</span></div>${button("panel:Pose", "Open pose editor", "pose", "wide")}`)}</div>`
  );
}
function assetsPanel() {
  const found = materials.filter((m) =>
      m.name.toLowerCase().includes(state.query.toLowerCase()),
    ),
    e = entity(state.selected),
    applicable = e && ["link", "geom"].includes(e.type);
  return (
    panelHead("Assets", "Material library") +
    `<div class="panel-tools">${search("Search materials")}</div><div class="panel-scroll" data-scroll="assets"><div class="material-grid">${found.map((m) => `<button class="material-card ${m.id === state.material ? "selected" : ""}" data-action="material:${m.id}" aria-pressed="${m.id === state.material}"><span class="material-preview"><span style="--material:${m.color}"></span></span><strong>${m.name}</strong><small>Roughness ${m.roughness.toFixed(2)}</small></button>`).join("")}</div>${!found.length ? empty("No materials found", "Try “sage” or clear your search.") : ""}</div><div class="panel-foot vertical"><span class="muted">${applicable ? "Apply to " + esc(e.name) : "Select a body or geometry to apply a material."}</span>${button("apply-material", "Apply material", "check", "primary wide", applicable ? "" : "disabled")}</div>`
  );
}
function camerasPanel() {
  return (
    panelHead("Cameras", "Viewport navigation") +
    `<div class="panel-scroll" data-scroll="cameras"><div class="padded">${select("Projection", "projection", ["Perspective", "Orthographic"], state.ortho ? "Orthographic" : "Perspective")}<div class="view-grid">${[
      ["X", "Right"],
      ["-X", "Left"],
      ["Y", "Back"],
      ["-Y", "Front"],
      ["Z", "Top"],
      ["-Z", "Bottom"],
    ]
      .map(([axis, label]) =>
        button("view:" + axis, label, "", "", `title="View from ${axis}"`),
      )
      .join(
        "",
      )}</div>${button("frame-all", "Frame all entities", "frame", "wide")}</div>${section("bookmarks", "Saved views", `<div class="button-row">${button("bookmark", "Save current view", "plus")}</div>${state.bookmarks.map((v, i) => `<div class="list-row">${button("bookmark:" + i, esc(v.name), "camera", "quiet")}<span class="spacer"></span>${ib("remove-bookmark:" + i, "close", "Remove " + v.name)}</div>`).join("") || '<p class="muted">Save a camera position to return to it later.</p>'}`)}</div>`
  );
}
function sensorsPanel() {
  return (
    panelHead("Sensors", "Pose preview signals") +
    `<div class="panel-scroll">${notice("These signals show the authored joint poses. Values use the timeline’s linear interpolation.")}<div class="padded">${JOINTS.map((j) => `<button class="sensor-row ${state.sensor === j.key ? "selected" : ""}" data-action="sensor:${j.key}">${icon("chart")}<span><strong>${j.name}</strong><small>Joint position · ${j.unit}</small></span><span class="mono" data-sensor-value="${j.key}">${state.pose[j.key].toFixed(2)}</span></button>`).join("")}</div></div><div class="panel-foot">${button("bottom:Signals", "Open signal plot", "chart", "wide")}</div>`
  );
}
function layersPanel() {
  return (
    panelHead("Layers", "Viewport visibility") +
    `<div class="panel-scroll" data-scroll="layers">${section("entitylayers", "Entity categories", ["link", "geom", "light", "camera"].map((k) => toggle("layer." + k, { link: "Bodies", geom: "Geometry", light: "Light helpers", camera: "Camera helpers" }[k], state.layers[k])).join(""))}${section("overlays", "Overlays", toggle("grid", "World grid", state.grid) + toggle("checker", "Checker floor", state.checker) + toggle("layer.floor", "Ground plane", state.layers.floor) + toggle("outlines", "Selection bounds", state.outlines) + toggle("gizmos", "Transform handles", state.gizmos))}${section("lighting", "Lighting", toggle("shadows", "Cast shadows", state.shadows) + field("Sun intensity", "lightIntensity", state.lightIntensity, "", { min: 0, max: 10, step: 0.1 }) + field("Navigation opacity", "navOpacity", state.navOpacity, "", { min: 0.15, max: 1, step: 0.05 }))}</div>`
  );
}
function outputRows() {
  return state.log
    .map(
      (row) =>
        `<div class="log-row ${row.level}">${icon(row.level)}<time>${row.time}</time><span>${esc(row.text)}</span></div>`,
    )
    .join("");
}
function outputPanel() {
  return (
    panelHead(
      "Output",
      `${state.log.length} events`,
      ib("clear-log", "delete", "Clear event log"),
    ) +
    `<div class="panel-scroll log-list" data-scroll="output">${outputRows() || empty("No events", "Editor events appear here.")}</div>`
  );
}
function statsPanel() {
  const metrics = viewport?.metrics() || {};
  return (
    panelHead("Statistics", "Current scene") +
    `<div class="panel-scroll padded"><div class="stat-row"><span>Scene entities</span><b>${state.entities.length}</b></div><div class="stat-row"><span>Visible entities</span><b>${state.entities.filter((e) => !e.hidden && state.layers[e.type]).length}</b></div><div class="stat-row"><span>Triangles</span><b data-metric="triangles">${metrics.triangles ?? 0}</b></div><div class="stat-row"><span>Draw calls</span><b data-metric="calls">${metrics.calls ?? 0}</b></div><div class="stat-row"><span>Renderer</span><b>WebGL</b></div><div class="stat-row"><span>World frame</span><b>Z up · meters</b></div>${notice("Counts describe this browser preview. Native backend performance is measured in Mojive.")}</div>`
  );
}
function renderFinder() {
  const f = {
    Scene: scenePanel,
    Pose: posePanel,
    Control: controlPanel,
    Assets: assetsPanel,
    Cameras: camerasPanel,
    Sensors: sensorsPanel,
    Layers: layersPanel,
    Output: outputPanel,
    Statistics: statsPanel,
  };
  replaceHTML($("#finder"), f[state.panel]());
}
function vector(e, key, label, unit, min = -10000, max = 10000) {
  return `<div class="vector-field"><div class="vector-heading"><span>${label}</span><small>${unit}</small></div><div class="vector-values">${e[key].map((v, i) => `<label class="axis-field axis-${i}"><span>${"XYZ"[i]}</span><input type="number" data-bind="entity.${key}.${i}" data-target="${e.id}" aria-label="${label} ${"XYZ"[i]}" value="${Number(v.toFixed(3))}" step="${key === "rotation" ? 1 : 0.01}" min="${min}" max="${max}" ${state.playing ? "disabled" : ""}></label>`).join("")}</div></div>`;
}
function renderInspector() {
  const e = inspected();
  let html = panelHead(
    "Inspector",
    state.pinned ? "Pinned to " + e?.name : "Follow selection",
    ib(
      "pin",
      "pin",
      state.pinned ? "Unpin inspector" : "Pin inspector",
      !!state.pinned,
      !e ? "disabled" : "",
    ),
  );
  if (!e) {
    replaceHTML(
      $("#inspector"),
      html +
        empty(
          "Nothing selected",
          "Choose an entity in the scene or viewport to inspect its properties.",
        ),
    );
    return;
  }
  const tabs = ["Properties", "Physics", "Material", "Joint"];
  if (!tabs.includes(state.tab)) state.tab = "Properties";
  const j = JOINTS.find((j) => j.name === e.id);
  html += `<div class="entity-heading"><div class="entity-title">${icon(e.type)}<strong title="${esc(e.name)}">${esc(e.name)}</strong><span class="type-pill type-${e.type}">${e.type}</span>${ib("rename:" + e.id, "more", "Rename " + e.name)}</div><div class="entity-path">world <span>›</span> <b>${esc(e.name)}</b></div></div>${state.playing ? `<div class="play-banner">${icon("lock")}<span>Playing · pause to edit</span>${button("play", "Pause", "pause", "quiet")}</div>` : ""}<div class="tab-strip" role="tablist" aria-label="Inspector section">${tabs.map((t) => button("tab:" + t, t, "", "", `role="tab" aria-selected="${state.tab === t}" tabindex="${state.tab === t ? 0 : -1}"`)).join("")}</div><div class="panel-scroll" data-scroll="inspector" role="tabpanel" aria-label="${state.tab}">`;
  if (state.tab === "Properties")
    html +=
      section(
        "transform",
        "Transform",
        `<div class="frame-note">World coordinates <span>Z up</span></div>` +
          vector(e, "position", "Position", "m") +
          vector(e, "rotation", "Rotation", "°", -3600, 3600) +
          vector(e, "scale", "Scale", "", 0.01, 100) +
          button(
            "reset-transform:" + e.id,
            "Restore transform",
            "reset",
            "quiet",
            state.playing ? "disabled" : "",
          ),
      ) +
      section(
        "display",
        "Display",
        toggle("visible:" + e.id, "Visible", !e.hidden) +
          (state.layers[e.type] === false
            ? notice(
                "This category is hidden. Enable it in Layers to see the entity.",
              )
            : "") +
          toggle("outlines", "Selection bounds", state.outlines),
      ) +
      section(
        "identity",
        "Identity",
        `<dl class="data-list"><dt>Object ID</dt><dd>${esc(e.id)}</dd><dt>Parent</dt><dd>world</dd><dt>Geometry</dt><dd>${e.shape}</dd></dl>`,
        false,
      );
  if (state.tab === "Physics")
    html +=
      notice("Connect a physics adapter to inspect motion, mass and inertia.") +
      section(
        "motion",
        "Motion",
        `<div class="unavailable-row"><span>Linear velocity</span><b>—</b><small>m/s</small></div><div class="unavailable-row"><span>Angular velocity</span><b>—</b><small>rad/s</small></div>`,
      ) +
      section(
        "inertial",
        "Inertial",
        `<div class="unavailable-row"><span>Mass</span><b>—</b><small>kg</small></div><div class="unavailable-row"><span>Inertia tensor</span><b>—</b></div>`,
      ) +
      section(
        "capability",
        "Simulation",
        button("panel:Control", "View adapter capabilities", "control", "wide"),
      );
  if (state.tab === "Material")
    html += ["link", "geom", "light"].includes(e.type)
      ? section(
          "surface",
          "Surface",
          `<label class="field-wrap"><span>Base color</span><span class="color-field"><input type="color" aria-label="Base color" data-color="${e.id}" value="${e.color}"><span class="mono">${e.color.toUpperCase()}</span></span></label>` +
            field("Roughness", "entity.roughness", e.roughness, "", {
              target: e.id,
              min: 0,
              max: 1,
              step: 0.05,
            }) +
            button("panel:Assets", "Browse materials", "assets", "wide"),
        )
      : empty(
          "No surface material",
          "Select a body or geometry to edit its surface.",
        );
  if (state.tab === "Joint")
    html += j
      ? section("joint", "Joint coordinate", jointControl(j)) +
        section(
          "limits",
          "Limits",
          `<dl class="data-list"><dt>Lower</dt><dd>${j.min} ${j.unit}</dd><dt>Upper</dt><dd>${j.max} ${j.unit}</dd><dt>Mode</dt><dd>Pose preview</dd></dl>`,
        )
      : empty(
          "No joint coordinate",
          "This entity has no editable joint in the preview scene.",
          button("panel:Pose", "Browse joints", "pose"),
        );

  html += `</div><div class="panel-foot">${button("duplicate:" + e.id, "Duplicate", "duplicate", "quiet")}${button("delete:" + e.id, "Delete", "delete", "quiet danger")}<span class="spacer"></span></div>`;
  replaceHTML($("#inspector"), html);
}
function renderViewport() {
  replaceHTML(
    $("#viewport-toolbar"),
    `${button("menu:Camera", state.ortho ? "Ortho" : "Persp", "camera", "glass", 'aria-haspopup="menu"')}${ib("menu:Shading", "shading", "Viewport shading", false, 'aria-haspopup="menu"')}${ib("panel:Layers", "layers", "Viewport layers")}`,
  );
  replaceHTML(
    $(".toolbox"),
    [
      ["select", "select", "Select", "V"],
      ["move", "move", "Move", "W"],
      ["rotate", "rotate", "Rotate", "E"],
      ["scale", "scale", "Scale", "R"],
    ]
      .map(
        ([a, i, label, key]) =>
          `<button class="button icon-button ${state.tool === a ? "active" : ""}" data-action="tool:${a}" title="${label} · ${key}" aria-label="${label} · ${key}" aria-pressed="${state.tool === a}" ${state.playing && a !== "select" ? "disabled" : ""}>${icon(i)}<kbd>${key}</kbd></button>`,
      )
      .join("") +
      `<div class="tool-separator"></div><button class="button icon-button" data-action="frame-mode" aria-label="${state.frame} frame · B" title="${state.frame} frame · B">${icon(state.frame === "World" ? "world" : "body")}<kbd>B</kbd></button><button class="button icon-button ${state.snap ? "active" : ""}" data-action="snap" aria-label="Snap · S" aria-pressed="${state.snap}" title="Snap · S">${icon("snap")}<kbd>S</kbd></button>`,
  );
  $(".view-name").textContent = state.cameraLabel;
  $(".view-context").textContent = history.document.name;
  const e = entity(state.selected);
  replaceHTML(
    $("#viewport-footer"),
    e
      ? `<div class="selection-caption glass">${icon(e.type)}<span class="crumb-parent">world</span><span class="crumb-separator">›</span><span class="truncate">${esc(e.name)}</span></div>`
      : "",
  );
  $("#viewport").classList.toggle("playing", state.playing);
  $(".play-state").hidden = !state.playing;
  $(".play-state").textContent = `Playing pose · ${state.speed.toFixed(2)}×`;
}
function renderTransport() {
  replaceHTML(
    $("#transport"),
    `<div class="transport-main">${ib("step:-1", "previous", "Previous frame", false, state.playing ? "disabled" : "")}${ib("play", state.playing ? "pause" : "play", state.playing ? "Pause playback · Space" : "Play timeline · Space", state.playing, history.document.keys.length ? "" : "disabled")}${ib("step:1", "next", "Next frame", false, state.playing ? "disabled" : "")}<span class="transport-divider"></span><label class="time-field"><input aria-label="Playhead time" type="number" data-bind="time" data-target="" min="0" max="10" step=".01" value="${state.time.toFixed(3)}" ${state.playing ? "disabled" : ""}><span>s</span></label><span class="transport-divider"></span>${ib("start", "reset", "Go to start")}${ib("record-unavailable", "record", "Record simulation · adapter required", false, "disabled")}${ib("menu:Playback", "down", "Playback options", false, 'aria-haspopup="menu"')}</div>`,
  );
  renderViewport();
  renderStatus();
}
function renderBottom() {
  const keys = history.document.keys;
  const stageWidth = $(".stage").clientWidth;
  const trackWidth = Math.max(
    1,
    stageWidth - (stageWidth > 35 * state.font ? 14 : 3) * state.font,
  );
  const groups = groupKeys(
    keys,
    trackWidth,
    history.document.duration,
    (36 * state.font) / 12,
  );
  replaceHTML(
    $("#bottom-panel"),
    `<div class="bottom-head">${ib("collapse", state.collapsed ? "right" : "down", state.collapsed ? "Expand bottom panel" : "Collapse bottom panel")}<div class="tab-strip" role="tablist" aria-label="Bottom panel">${["Timeline", "Signals", "Events"].map((t) => button("bottom:" + t, t, t === "Timeline" ? "key" : t === "Signals" ? "chart" : "terminal", "", `role="tab" aria-selected="${state.bottom === t}" tabindex="${state.bottom === t ? 0 : -1}"`)).join("")}</div><span class="spacer"></span>${state.bottom === "Timeline" ? `<span class="timeline-help">${state.selectedKey ? esc(keys.find((k) => k.id === state.selectedKey)?.name || "") : "Pose keys"}</span>${ib("delete-key", "delete", "Delete selected key", false, state.selectedKey ? "" : "disabled")}` : ""}${state.bottom === "Timeline" ? button("add-key", "Key", "addkey", "quiet", state.playing ? "disabled" : "") : ""}${ib("menu:Timeline", "more", "Timeline actions", false, 'aria-haspopup="menu"')}</div><div class="bottom-content" ${state.collapsed ? "hidden" : ""}>${
      state.bottom === "Timeline"
        ? `<div class="timeline-track"><div class="track-label"><b>Joint poses</b><small>Linear interpolation</small></div><div class="track-content"><div class="time-ruler">${Array.from({ length: 11 }, (_, i) => `<span style="--position:${i * 10}%">${i}</span>`).join("")}</div><div class="key-lane">${groups
            .map((group) => {
              const k = group[0],
                selected = group.some((k) => k.id === state.selectedKey);
              return `<button class="keyframe ${selected ? "selected" : ""}" style="--position:${k.time * 10}%" data-action="${group.length > 1 ? "keys:" + group.map((k) => k.id).join(",") : "key:" + k.id}" aria-label="${group.length > 1 ? group.length + " keys near " + k.time.toFixed(2) + " seconds" : "Key " + esc(k.name) + " at " + k.time.toFixed(2) + " seconds"}" aria-pressed="${selected}" title="${group.length > 1 ? "Choose one of " + group.length + " keys" : esc(k.name) + " · " + k.time.toFixed(2) + " s"}">${group.length > 1 ? `<b>${group.length}</b>` : icon("key")}<span>${group.length > 1 ? "Keys" : esc(k.name)}</span></button>`;
            })
            .join(
              "",
            )}</div><div class="playhead" style="left:${state.time * 10}%"></div><input class="timeline-seek" aria-label="Timeline playhead" type="range" data-seek min="0" max="10" step=".01" value="${state.time}"></div></div>`
        : state.bottom === "Signals"
          ? `<div class="signal-panel"><label><span>Joint position</span><select aria-label="Signal" data-choice="sensor">${JOINTS.map((j) => `<option value="${j.key}" ${state.sensor === j.key ? "selected" : ""}>${j.name} · ${j.unit}</option>`).join("")}</select></label><canvas id="signal-plot" aria-label="Authored joint position over ten seconds"></canvas></div>`
          : `<div class="event-feed" data-scroll="events">${outputRows()}</div>`
    }</div>`,
  );
  $(".stage").classList.toggle("bottom-collapsed", state.collapsed);
  drawPlot();
}
function drawPlot() {
  const canvas = $("#signal-plot");
  if (!canvas) return;
  const r = canvas.getBoundingClientRect(),
    d = Math.min(devicePixelRatio, 2);
  canvas.width = r.width * d;
  canvas.height = r.height * d;
  const c = canvas.getContext("2d");
  c.scale(d, d);
  const w = r.width,
    h = r.height,
    j = JOINTS.find((j) => j.key === state.sensor);
  c.strokeStyle = "#353b43";
  c.lineWidth = 1;
  for (let n = 0; n < 5; n++) {
    const y = 8 + ((h - 25) * n) / 4;
    c.beginPath();
    c.moveTo(35, y);
    c.lineTo(w - 12, y);
    c.stroke();
  }
  c.fillStyle = "#a8b0ba";
  c.font = "11px ui-monospace,monospace";
  c.fillText(j.max, 0, 12);
  c.fillText(j.min, 0, h - 14);
  c.fillText("0 s", 35, h - 1);
  c.fillText("10 s", w - 40, h - 1);
  c.strokeStyle = "#9cbf8d";
  c.lineWidth = 2;
  c.beginPath();
  for (let n = 0; n <= 200; n++) {
    const pose = samplePose(history.document.keys, n / 20) || state.pose,
      v = pose[j.key],
      x = 35 + ((w - 47) * n) / 200,
      y = 8 + (h - 25) * (1 - (v - j.min) / (j.max - j.min));
    n ? c.lineTo(x, y) : c.moveTo(x, y);
  }
  c.stroke();
  c.strokeStyle = "#e8b04f";
  c.beginPath();
  const x = 35 + ((w - 47) * state.time) / 10;
  c.moveTo(x, 4);
  c.lineTo(x, h - 16);
  c.stroke();
}
function renderStatus() {
  replaceHTML(
    $("#statusbar"),
    `<span class="status-indicator ${state.playing ? "running" : ""}"></span><span>${state.playing ? "Playing" : history.dirty ? "Unsaved changes" : "Paused"}</span><span class="status-hints"><kbd>W E R</kbd> Tools <kbd>B</kbd> Frame <kbd>T</kbd> Timeline</span><span class="spacer"></span>${button("panel:Output", String(state.log.length), "info", "quiet status-events")}<span class="status-extra">${state.entities.length} entities · Z up</span><button class="reference-link" data-action="components" title="Design system and prototype scope">Instrument / 08</button>`,
  );
}
function render() {
  renderTitle();
  replaceHTML(
    $(".rail"),
    panels
      .map(
        ([n, i]) =>
          (n === "Output" ? '<span class="rail-spacer"></span>' : "") +
          ib(
            "panel:" + n,
            i,
            n,
            state.panel === n,
            `aria-pressed="${state.panel === n}"`,
          ),
      )
      .join(""),
  );
  renderFinder();
  renderInspector();
  renderViewport();
  renderTransport();
  layout();
  renderBottom();
  renderStatus();
}
let widths = { left: 272, right: 320 };
function layout() {
  const unit = state.font / 12,
    w = innerWidth;
  const compactFinder = w < 960 * unit,
    compactInspector = w < 1200 * unit;
  const workspace = $("#workspace");
  workspace.classList.toggle("compact-finder", compactFinder);
  workspace.classList.toggle("compact-inspector", compactInspector);
  workspace.style.setProperty(
    "--left",
    state.finder && !compactFinder ? widths.left * unit + "px" : "0px",
  );
  workspace.style.setProperty(
    "--right",
    state.inspector && !compactInspector ? widths.right * unit + "px" : "0px",
  );
  workspace.classList.toggle("finder-off", !state.finder);
  workspace.classList.toggle("inspector-off", !state.inspector);
  if (state.drawer === "finder" && !compactFinder) state.drawer = null;
  if (state.drawer === "inspector" && !compactInspector) state.drawer = null;
  workspace.dataset.drawer = state.drawer || "";
  $(".drawer-scrim").hidden = !state.drawer;
  $(".stage").inert = !!state.drawer;
  $("#transport").inert = !!state.drawer;
  $("#bottom-panel").inert = !!state.drawer;
  $("#finder").inert =
    state.drawer === "inspector" ||
    (compactFinder ? state.drawer !== "finder" : !state.finder);
  $("#inspector").inert =
    state.drawer === "finder" ||
    (compactInspector ? state.drawer !== "inspector" : !state.inspector);
  document.querySelectorAll(".rail [data-action]").forEach((item) => {
    const active =
      item.dataset.action === "panel:" + state.panel && !$("#finder").inert;
    item.classList.toggle("active", active);
    item.setAttribute("aria-pressed", String(active));
  });
  $(".left-split").setAttribute("aria-valuenow", widths.left);
  $(".right-split").setAttribute("aria-valuenow", widths.right);
  buttonState(
    "toggle-inspector",
    compactInspector ? state.drawer === "inspector" : state.inspector,
  );
}
function selectEntity(id) {
  if (id === undefined) {
    renderViewport();
    return;
  }
  state.selected = id;
  state.selectedKey = null;
  if (!state.pinned) state.tab = "Properties";
  viewport.update();
  render();
}
function openPanel(name) {
  state.panel = name;
  state.query = "";
  state.finder = true;
  if (innerWidth < (960 * state.font) / 12) state.drawer = "finder";
  render();
}
function seek(time, selectKey = false) {
  state.time = clamp(time, 0, history.document.duration);
  state.playing = false;
  state.previewPose = samplePose(history.document.keys, state.time);
  if (!selectKey) state.selectedKey = null;
  viewport.update();
  render();
}
function captureKey() {
  const id = crypto.randomUUID(),
    pose = clone(state.pose);
  let keyId;
  transact("Capture pose key", (doc) => {
    doc.pose = pose;
    keyId = upsertKey(doc, state.time, id);
  });
  state.selectedKey = keyId;
  renderBottom();
  toast("Pose captured at " + state.time.toFixed(2) + " s");
}
function addEntity(shape) {
  let id = crypto.randomUUID(),
    index = 1,
    name = shape;
  while (state.entities.some((e) => e.name === name))
    name = shape + "_" + index++;
  transact(
    "Add " + shape,
    (doc) =>
      doc.entities.push({
        id,
        name,
        type: "geom",
        shape,
        position: [0, 0, 0.6],
        rotation: [0, 0, 0],
        scale: [1, 1, 1],
        color: "#9cbf8d",
        roughness: 0.6,
        hidden: false,
      }),
    { rebuild: true },
  );
  state.selected = id;
  state.tab = "Properties";
  state.inspector = true;
  viewport.update();
  render();
  toast(name + " added");
}
function download(data, name, type = "application/json") {
  const url = URL.createObjectURL(new Blob([data], { type })),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}
function saveLocal() {
  try {
    localStorage.setItem(
      "mojive-instrument-v03",
      JSON.stringify(history.document),
    );
    history.markSaved();
    renderTitle();
    renderStatus();
    toast("Saved locally in this browser");
  } catch {
    toast("Local storage is unavailable. Use File → Export scene JSON.");
  }
}
function loadLocal() {
  try {
    const raw = localStorage.getItem("mojive-instrument-v03");
    if (!raw) {
      toast("No saved scene in this browser");
      return;
    }
    const next = JSON.parse(raw);
    if (!next.entities || !next.keys) throw Error();
    confirmReplace("Open saved scene", () => {
      history = new History(next);
      resetSceneState();
    });
  } catch {
    toast(
      "The saved scene could not be read. Your current scene is unchanged.",
    );
  }
}
function resetSceneState() {
  state.selected = entity("hinge_body")?.id || state.entities[0]?.id || null;
  state.pinned = null;
  state.previewPose = null;
  state.selectedKey = null;
  state.playing = false;
  state.time =
    history.document.keys.find((k) =>
      JOINTS.every(
        (j) => Math.abs(k.pose[j.key] - history.document.pose[j.key]) < 0.00001,
      ),
    )?.time || 0;
  viewport.rebuild();
  render();
  viewport.frame(true);
}
function confirmReplace(title, callback) {
  if (!history.dirty) {
    callback();
    return;
  }
  showDialog(
    title,
    `<p>Your scene contains unsaved changes.</p><p class="muted">Save them locally before replacing this scene, or discard them.</p>`,
    button("dialog-close", "Cancel") +
      button("discard-document", "Discard changes", "", "danger") +
      button("save-replace", "Save and continue", "save", "primary"),
  );
  dialogHandlers["discard-document"] = () => {
    closeDialog();
    callback();
  };
  dialogHandlers["save-replace"] = () => {
    saveLocal();
    if (!history.dirty) {
      closeDialog();
      callback();
    }
  };
}
const dialogHandlers = {};
function showDialog(
  title,
  body,
  footer = button("dialog-close", "Done", "", "primary"),
  wide = false,
) {
  closePopover();
  const modal = $("#modal");
  if (!modal.open) dialogOpener = document.activeElement;
  modal.className = wide ? "wide-dialog" : "";
  modal.innerHTML = `<header class="dialog-head"><div><span class="eyebrow">MOJIVE / INSTRUMENT</span><h1 id="dialog-title">${title}</h1></div>${ib("dialog-close", "close", "Close dialog")}</header><div class="dialog-body">${body}</div><footer class="dialog-foot">${footer}</footer>`;
  modal.setAttribute("aria-labelledby", "dialog-title");
  if (!modal.open) modal.showModal();
  requestAnimationFrame(() => modal.querySelector("[autofocus]")?.focus());
}
function closeDialog() {
  $("#modal").close();
  dialogOpener?.isConnected && dialogOpener.focus();
}
function preferences() {
  showDialog(
    "Preferences",
    `<p class="muted">Display settings apply immediately to this reference.</p>${select("Interface text size", "font", ["12 px", "13 px", "14 px", "16 px"], state.font + " px")}${field("Navigation opacity", "navOpacity", state.navOpacity, "", { min: 0.15, max: 1, step: 0.05 })}${toggle("grid", "World grid", state.grid)}${toggle("shadows", "Cast shadows", state.shadows)}${toggle("gizmos", "Transform handles", state.gizmos)}<p class="muted">Panel widths and control heights scale with the text size. Narrow windows use side drawers.</p>`,
  );
}
function guide() {
  showDialog(
    "Interaction guide",
    `<div class="guide-grid">${[
      ["Orbit", "Drag on empty viewport"],
      ["Pan", "Right-drag or Shift + drag"],
      ["Zoom", "Scroll"],
      ["Select / Move / Rotate / Scale", "V / W / E / R"],
      ["Frame selected", "F"],
      ["World / body frame", "B"],
      ["Toggle snap", "S"],
      ["Play / pause", "Space"],
      ["Expand / collapse timeline", "T"],
      ["Capture pose key", "K"],
      ["Undo / redo", "⌘Z / ⇧⌘Z"],
      ["Save locally", "⌘S"],
      ["Rename entity", "F2"],
      ["Delete entity", "Delete"],
      ["Cancel edit / close overlay", "Escape"],
    ]
      .map(([a, b]) => `<div><span>${a}</span><kbd>${b}</kbd></div>`)
      .join(
        "",
      )}</div>${notice("Transform handles affect the selected entity. The inspector can be pinned to a different entity; its edits always target the pinned entity.")}<p class="muted">Timeline keys store joint poses. Capture overwrites a key only at the same time (within 0.025 s). Delete acts only on a selected key.</p>`,
    button("dialog-close", "Start exploring", "", "primary"),
    true,
  );
}
function components() {
  showDialog(
    "Design system",
    `<div class="design-intro"><h2>Quiet surfaces. Precise controls.</h2><p>One selection surface. Shared corner radii. Native docking constraints.</p></div><div class="spec-grid"><article><h3>Typography</h3><div class="type-sample"><strong>Entity properties</strong><p>Interface text · ${state.font} px</p><span class="mono">0.125 m &nbsp; −35.000°</span></div>${select("Interface text size", "font", ["12 px", "13 px", "14 px", "16 px"], state.font + " px")}<p class="muted">Body 1 rem · metadata 0.846 rem · title 1.25 rem. Numbers use tabular figures.</p></article><article><h3>Surface and semantics</h3><div class="swatches">${[
      ["#131518", "Canvas"],
      ["#1e2125", "Panel"],
      ["#2b2f34", "Input"],
      ["#9cbf8d", "Action"],
      ["#e8b04f", "Selected"],
      ["#e38b80", "Error"],
    ]
      .map(([c, n]) => `<span><i style="background:${c}"></i>${n}</span>`)
      .join(
        "",
      )}</div><p class="muted">Selection uses amber; available actions use sage. State is also conveyed by a label, outline or symbol.</p></article><article><h3>Buttons and states</h3><div class="button-row">${button("sample", "Default")}${button("sample", "Primary", "check", "primary")}${button("sample", "Disabled", "", "", "disabled")}</div><div class="button-row">${ib("sample", "move", "Move specimen")}${ib("sample", "rotate", "Selected rotate specimen", true)}${ib("sample", "scale", "Disabled scale specimen", false, "disabled")}</div><p class="muted">Controls: 2.333 rem high. Icons: 1.333 rem. Keyboard focus: 2 px sage ring.</p></article><article><h3>Validation</h3><label class="field-wrap"><span>Scale</span><span class="input-shell invalid"><input type="number" aria-label="Validation example" aria-invalid="true" aria-describedby="spec-error" value="0" min=".01" max="100" data-spec-input><span class="unit">×</span></span><small id="spec-error" class="field-error">Use a value from 0.01 to 100.</small></label><p class="muted">Invalid input remains visible. Enter commits; Escape restores the previous value. Try editing this specimen.</p></article></div><article class="docking-note"><h3>Shape and docking contract</h3><p>Docked panels have square shared edges. Controls use a 4.8 px radius; floating containers use 8 px. A selected rail item has one tinted surface. A focus outline appears only for keyboard navigation.</p><p>Viewport overlays use an alpha fill without background blur. Native dock tabs own window dragging, grouping and floating; the web reference does not emulate native docking.</p><p>Open the companion native study with <code>make ui-design-native ARGS=--interactive</code>.</p></article><article class="icon-spec"><h3>Canonical icons · 24-unit grid</h3><p class="muted">The icon masters come directly from the original design draft: 24-unit grid, 1.75-unit strokes, round caps and joins. Supplemental icons follow the same geometry rules.</p><div class="icon-grid">${["move", "rotate", "scale", "world", "body", "snap", "play", "pause", "previous", "next", "reset", "record", "stop", "key", "addkey", "search", "eye", "eyeoff", "camera", "light", "info", "warning", "error", "frame"].map((n) => `<div>${[14, 18, 24].map((size) => `<span style="--spec-size:${size}px">${icon(n)}</span>`).join("")}<small>${n}</small></div>`).join("")}</div></article>`,
    button("dialog-close", "Done", "", "primary"),
    true,
  );
}
function rename(id) {
  const e = entity(id);
  if (!e) return;
  showDialog(
    "Rename entity",
    `<label class="field-wrap"><span>Name</span><input type="text" aria-label="Entity name" id="entity-name" maxlength="80" value="${esc(e.name)}" autofocus><small class="field-error" id="rename-error"></small></label>`,
    button("dialog-close", "Cancel") +
      button("rename-commit", "Rename", "", "primary"),
  );
  dialogHandlers["rename-commit"] = () => {
    try {
      const name = uniqueName(state.entities, $("#entity-name").value, id);
      transact(
        "Rename entity",
        (doc) => (doc.entities.find((e) => e.id === id).name = name),
      );
      closeDialog();
    } catch (err) {
      $("#rename-error").textContent = err.message;
      $("#entity-name").setAttribute("aria-invalid", "true");
    }
  };
}
function menuItem(action, label, ico = "", shortcut = "", off = false) {
  return button(
    action,
    esc(label) + (shortcut ? `<kbd>${shortcut}</kbd>` : ""),
    ico,
    "menu-item",
    `role="menuitem" ${off ? "disabled" : ""}`,
  );
}
function openMenu(name, anchor) {
  if (anchor.closest("#popover")) anchor = popoverOpener;
  const items = {
    Camera: () =>
      menuItem(
        "projection",
        state.ortho ? "Switch to perspective" : "Switch to orthographic",
        "camera",
      ) +
      menuItem("frame", "Frame selected", "frame", "F", !state.selected) +
      menuItem("frame-all", "Frame all", "frame") +
      "<hr>" +
      menuItem("panel:Cameras", "Cameras and saved views", "camera"),
    Shading: () =>
      ["Solid", "Wireframe"]
        .map((n) =>
          menuItem(
            "shading:" + n,
            n,
            state.shading === n ? "check" : "shading",
          ),
        )
        .join(""),
    Playback: () =>
      menuItem("step:-1", "Previous frame", "previous", "", state.playing) +
      menuItem("step:1", "Next frame", "next", "", state.playing) +
      "<hr>" +
      [0.25, 0.5, 1, 2]
        .map((n) =>
          menuItem(
            "speed:" + n,
            n + "× speed",
            state.speed === n ? "check" : "",
          ),
        )
        .join("") +
      "<hr>" +
      menuItem("loop", state.loop ? "Disable loop" : "Enable loop", "loop"),
    Simulate: () =>
      menuItem(
        "play",
        state.playing ? "Pause preview" : "Play pose preview",
        state.playing ? "pause" : "play",
        "Space",
        !history.document.keys.length,
      ) +
      menuItem("start", "Reset playhead", "reset") +
      "<hr>" +
      menuItem("panel:Control", "Physics adapter", "control"),
    Window: () =>
      menuItem("toggle-finder", "Toggle panel browser", "scene") +
      menuItem("toggle-inspector", "Toggle inspector", "inspect") +
      menuItem(
        "collapse",
        state.collapsed ? "Expand timeline" : "Collapse timeline",
        "key",
        "T",
      ) +
      "<hr>" +
      ["Edit", "Author", "Review"]
        .map((n) => menuItem("workspace:" + n, n + " workspace"))
        .join(""),
    Help: () =>
      menuItem("guide", "Interaction guide", "help") +
      menuItem("components", "Design system", "assets") +
      menuItem("preferences", "Preferences", "settings"),
    More: () =>
      ["Edit", "View", "Simulate", "Window", "Help"]
        .map((n) => menuItem("menu:" + n, n))
        .join(""),
    File: () =>
      menuItem("save", "Save locally", "save", "⌘S") +
      menuItem("load", "Open saved scene", "assets") +
      menuItem("export", "Export scene JSON", "export") +
      "<hr>" +
      menuItem("demo", "Reset to joint study", "reset") +
      menuItem("new", "New empty scene", "plus"),
    Edit: () =>
      menuItem("undo", "Undo", "undo", "⌘Z", !history.past.length) +
      menuItem("redo", "Redo", "redo", "⇧⌘Z", !history.future.length) +
      "<hr>" +
      menuItem(
        "duplicate:" + state.selected,
        "Duplicate selected",
        "duplicate",
        "⇧D",
        !state.selected,
      ) +
      menuItem(
        "delete:" + state.selected,
        "Delete selected",
        "delete",
        "⌫",
        !state.selected,
      ),
    Add: () =>
      ["Box", "Sphere", "Cylinder", "Capsule"]
        .map((n) => menuItem("create:" + n.toLowerCase(), n, "box"))
        .join(""),
    View: () =>
      menuItem("toggle-finder", "Toggle panel browser", "scene") +
      menuItem("toggle-inspector", "Toggle inspector", "inspect") +
      menuItem("frame-all", "Frame all", "frame") +
      menuItem("preferences", "Preferences", "settings") +
      menuItem("components", "Design system", "assets") +
      menuItem("guide", "Interaction guide", "help"),
    Timeline: () =>
      menuItem("add-key", "Capture current pose", "addkey", "", state.playing) +
      menuItem(
        "delete-key",
        "Delete selected key",
        "delete",
        "",
        !state.selectedKey,
      ) +
      menuItem("export", "Export scene JSON", "export") +
      menuItem("start", "Go to start", "reset") +
      "<hr>" +
      menuItem(
        "record-disabled",
        "Record simulation",
        "record",
        "Adapter required",
        true,
      ),
  };
  if (!items[name]) return;
  closePopover();
  popoverOpener = anchor;
  const pop = $("#popover");
  pop.innerHTML = `<div class="menu" role="menu" aria-label="${name}">${items[name]()}</div>`;
  pop.hidden = false;
  anchor?.setAttribute("aria-expanded", "true");
  const r = anchor.getBoundingClientRect(),
    w = pop.offsetWidth,
    h = pop.offsetHeight;
  pop.style.left = clamp(r.left, 8, innerWidth - w - 8) + "px";
  pop.style.top =
    (r.bottom + h + 8 < innerHeight
      ? r.bottom + 4
      : Math.max(8, r.top - h - 4)) + "px";
  pop.querySelector("button:not(:disabled)")?.focus();
}
function openKeyChooser(ids, anchor) {
  popoverOpener = anchor;
  const pop = $("#popover");
  pop.innerHTML = `<div class="menu" role="menu" aria-label="Choose a key">${ids
    .map((id) => {
      const k = history.document.keys.find((k) => k.id === id);
      return menuItem("key:" + id, k.name, "key", k.time.toFixed(2) + " s");
    })
    .join("")}</div>`;
  pop.hidden = false;
  const r = anchor.getBoundingClientRect();
  pop.style.left = clamp(r.left, 8, innerWidth - pop.offsetWidth - 8) + "px";
  pop.style.top = Math.max(8, r.top - pop.offsetHeight - 6) + "px";
  pop.querySelector("button")?.focus();
}
function closePopover(restore = false) {
  $("#popover").hidden = true;
  popoverOpener?.setAttribute("aria-expanded", "false");
  if (restore && popoverOpener?.isConnected) popoverOpener.focus();
}
function action(a, el) {
  const [cmd, ...rest] = a.split(":"),
    arg = rest.join(":");
  if (dialogHandlers[a] && $("#modal").open) {
    dialogHandlers[a]();
    return;
  }
  if (cmd === "menu") {
    openMenu(arg, el);
    return;
  }
  closePopover(!!el.closest("#popover"));
  switch (cmd) {
    case "workspace":
      state.workspace = arg;
      state.drawer = null;
      state.inspector = true;
      state.finder = arg !== "Review";
      state.panel = arg === "Author" ? "Assets" : "Scene";
      state.collapsed = arg !== "Review";
      state.bottom = "Timeline";
      state.query = "";
      widths = { left: 272, right: arg === "Author" ? 360 : 320 };
      render();
      break;
    case "shading":
      state.shading = arg;
      viewport.update();
      renderViewport();
      break;
    case "speed":
      state.speed = Number(arg);
      renderTransport();
      break;
    case "panel":
      openPanel(arg);
      break;
    case "select":
      selectEntity(arg);
      break;
    case "filter":
      state.filter = arg;
      renderFinder();
      break;
    case "clear-filter":
      state.filter = "All";
      state.query = "";
      renderFinder();
      break;
    case "clear-search":
      state.query = "";
      renderFinder();
      $("[data-query]")?.focus();
      break;
    case "world-toggle":
      state.worldOpen = !state.worldOpen;
      renderFinder();
      break;
    case "visible":
      transact("Toggle entity visibility", (d) => {
        const e = d.entities.find((e) => e.id === arg);
        e.hidden = !e.hidden;
      });
      break;
    case "pin":
      state.pinned = state.pinned ? null : state.selected;
      renderInspector();
      break;
    case "tab":
      state.tab = arg;
      renderInspector();
      break;
    case "tool":
      state.tool = arg;
      viewport.update();
      renderViewport();
      break;
    case "frame-mode":
      state.frame = state.frame === "World" ? "Body" : "World";
      viewport.update();
      renderViewport();
      break;
    case "snap":
      state.snap = !state.snap;
      viewport.update();
      renderViewport();
      break;
    case "frame":
      viewport.frame();
      break;
    case "frame-all":
      viewport.frame(true);
      break;
    case "projection":
      state.ortho = !state.ortho;
      viewport.projectMode(state.ortho);
      renderViewport();
      if (state.panel === "Cameras") renderFinder();
      break;
    case "view":
      viewport.orient(arg);
      state.cameraLabel = {
        X: "Right",
        "-X": "Left",
        Y: "Back",
        "-Y": "Front",
        Z: "Top",
        "-Z": "Bottom",
      }[arg];
      renderViewport();
      break;
    case "bookmark":
      if (arg) {
        viewport.restoreView(state.bookmarks[+arg]);
        state.cameraLabel = state.bookmarks[+arg].name;
        render();
      } else {
        state.bookmarks.push({
          ...viewport.saveView(),
          name: "View " + (state.bookmarks.length + 1),
        });
        renderFinder();
        toast("Camera view saved");
      }
      break;
    case "remove-bookmark":
      state.bookmarks.splice(+arg, 1);
      renderFinder();
      break;
    case "play":
      if (!history.document.keys.length) return;
      if (!state.playing && state.time >= history.document.duration)
        state.time = 0;
      state.playing = !state.playing;
      viewport.update();
      renderTransport();
      renderFinder();
      renderInspector();
      renderBottom();
      break;
    case "loop":
      state.loop = !state.loop;
      renderTransport();
      break;
    case "step":
      seek(state.time + +arg / 60);
      break;
    case "start":
      seek(0);
      break;
    case "keys":
      openKeyChooser(arg.split(","), el);
      break;
    case "key": {
      const k = history.document.keys.find((k) => k.id === arg);
      if (k) {
        state.selectedKey = arg;
        seek(k.time, true);
      }
      break;
    }
    case "add-key":
      if (!state.playing) captureKey();
      break;
    case "delete-key":
      if (state.selectedKey)
        transact(
          "Delete pose key",
          (d) => (d.keys = d.keys.filter((k) => k.id !== state.selectedKey)),
        );
      break;
    case "pose-rest":
      transact(
        "Restore rest pose",
        (d) => (d.pose = { hinge: 0, ball: 0, slide: 0 }),
      );
      break;
    case "bottom":
      state.bottom = arg;
      state.collapsed = false;
      renderBottom();
      break;
    case "collapse":
      state.collapsed = !state.collapsed;
      renderBottom();
      break;
    case "sensor":
      state.sensor = arg;
      state.bottom = "Signals";
      state.collapsed = false;
      renderFinder();
      renderBottom();
      break;
    case "material":
      state.material = arg;
      renderFinder();
      break;
    case "apply-material": {
      const m = materials.find((m) => m.id === state.material),
        id = state.selected;
      transact("Apply " + m.name, (d) =>
        Object.assign(
          d.entities.find((e) => e.id === id),
          { color: m.color, roughness: m.roughness },
        ),
      );
      break;
    }
    case "reset-transform":
      transact("Reset transform", (d) => {
        const e = d.entities.find((e) => e.id === arg),
          original = initialDocument().entities.find((o) => o.id === arg);
        Object.assign(
          e,
          original
            ? {
                position: original.position,
                rotation: original.rotation,
                scale: original.scale,
              }
            : { position: [0, 0, 0.6], rotation: [0, 0, 0], scale: [1, 1, 1] },
        );
      });
      break;
    case "rename":
      rename(arg);
      break;
    case "duplicate": {
      const source = entity(arg);
      if (!source) break;
      let index = 1,
        name = source.name + "_copy";
      while (state.entities.some((e) => e.name === name))
        name = source.name + "_copy" + index++;
      const copy = { ...clone(source), id: crypto.randomUUID(), name };
      copy.position[0] += 0.5;
      transact("Duplicate entity", (d) => d.entities.push(copy), {
        rebuild: true,
      });
      state.selected = copy.id;
      viewport.update();
      render();
      break;
    }
    case "delete":
      if (entity(arg)) {
        transact(
          "Delete entity",
          (d) => (d.entities = d.entities.filter((e) => e.id !== arg)),
          { rebuild: true },
        );
        toast("Entity deleted · Undo with ⌘Z");
      }
      break;
    case "create":
      addEntity(arg);
      break;
    case "undo":
      historyAction();
      break;
    case "redo":
      historyAction(true);
      break;
    case "save":
      saveLocal();
      break;
    case "load":
      loadLocal();
      break;
    case "export":
      download(
        JSON.stringify(
          {
            format: "mojive-design-reference",
            version: 3,
            document: history.document,
          },
          null,
          2,
        ),
        "mojive-scene.json",
      );
      toast("Scene JSON prepared for download");
      break;
    case "demo":
      confirmReplace("Reset to joint study", () => {
        history = new History(initialDocument());
        resetSceneState();
      });
      break;
    case "new":
      confirmReplace("New scene", () => {
        const d = initialDocument();
        d.name = "Untitled scene";
        d.entities = [];
        d.keys = [];
        history = new History(d);
        resetSceneState();
      });
      break;
    case "toggle-finder":
      if (innerWidth < (960 * state.font) / 12)
        state.drawer = state.drawer === "finder" ? null : "finder";
      else state.finder = !state.finder;
      render();
      break;
    case "toggle-inspector":
      if (innerWidth < (1200 * state.font) / 12)
        state.drawer = state.drawer === "inspector" ? null : "inspector";
      else state.inspector = !state.inspector;
      render();
      break;
    case "close-panel":
      if (state.drawer) state.drawer = null;
      else if (el.closest("#finder")) state.finder = false;
      else state.inspector = false;
      render();
      break;
    case "drawer-close":
      state.drawer = null;
      layout();
      break;
    case "preferences":
      preferences();
      break;
    case "guide":
      guide();
      break;
    case "components":
      components();
      break;
    case "dialog-close":
      closeDialog();
      break;
    case "sample":
      toast("Button activated");
      break;
    case "clear-log":
      state.log = [];
      renderFinder();
      renderBottom();
      break;
  }
}
function setError(input, message) {
  const root =
    input.closest(".field-wrap,.vector-field,.time-field") ||
    input.parentElement;
  input.setAttribute("aria-invalid", "true");
  input.closest(".input-shell,.axis-field")?.classList.add("invalid");
  let error = root.querySelector(".field-error");
  if (!error) {
    error = document.createElement("small");
    error.className = "field-error";
    error.id = "error-" + crypto.randomUUID();
    root.append(error);
  }
  error.textContent = message;
  input.setAttribute("aria-describedby", error.id);
}
function clearError(input) {
  input.removeAttribute("aria-invalid");
  input.removeAttribute("aria-describedby");
  input.closest(".input-shell,.axis-field")?.classList.remove("invalid");
  const root = input.closest(".field-wrap,.vector-field,.time-field");
  root?.querySelector(".field-error")?.remove();
}
function setNumeric(input) {
  const key = input.dataset.bind;
  if (!key) return;
  try {
    const value = numericValue(input.value, {
      min: Number(input.min),
      max: Number(input.max),
    });
    clearError(input);
    if (key === "time") {
      seek(value);
      return;
    }
    if (key.startsWith("entity.")) {
      const [_, prop, index] = key.split("."),
        id = input.dataset.target;
      transact("Edit " + prop, (d) => {
        const e = d.entities.find((e) => e.id === id);
        if (index !== undefined) e[prop][+index] = value;
        else e[prop] = value;
      });
    } else if (key.startsWith("pose.")) {
      const pose = clone(state.pose);
      pose[key.split(".")[1]] = value;
      transact("Edit joint pose", (d) => (d.pose = pose));
    } else {
      state[key] = value;
      viewport.update();
      if (!$("#modal").open) render();
    }
  } catch (err) {
    setError(input, err.message);
  }
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-action]");
  if (b && !b.disabled) {
    action(b.dataset.action, b);
  } else if (!e.target.closest("#popover")) closePopover();
});
document.addEventListener("input", (e) => {
  const el = e.target;
  if (el.dataset.query) {
    state.query = el.value;
    renderFinder();
  }
  if (el.hasAttribute("data-seek")) {
    state.time = Number(el.value);
    state.playing = false;
    state.previewPose = samplePose(history.document.keys, state.time);
    state.selectedKey = null;
    viewport.update();
    updateTime();
  }
  if (el.dataset.poseRange) {
    if (!dragStart) dragStart = clone(state.pose);
    state.previewPose = {
      ...state.pose,
      [el.dataset.poseRange]: Number(el.value),
    };
    viewport.update();
    const field = $(
      `[data-bind="pose.${el.dataset.poseRange}"]`,
      el.closest(".joint-control"),
    );
    if (field) field.value = el.value;
  }
  if (el.hasAttribute("data-spec-input")) {
    try {
      numericValue(el.value, { min: 0.01, max: 100 });
      clearError(el);
    } catch (error) {
      setError(el, error.message);
    }
  }
});
document.addEventListener("change", (e) => {
  const el = e.target;
  if (el.dataset.bind) setNumeric(el);
  if (el.dataset.poseRange) {
    const pose = clone(state.pose);
    dragStart = null;
    transact("Edit joint pose", (d) => (d.pose = pose));
  }
  if (el.hasAttribute("data-seek")) {
    seek(Number(el.value));
  }
  if (el.dataset.color) {
    transact(
      "Edit base color",
      (d) =>
        (d.entities.find((e) => e.id === el.dataset.color).color = el.value),
    );
  }
  if (el.dataset.toggle) {
    const key = el.dataset.toggle;
    if (key.startsWith("visible:"))
      transact(
        "Toggle entity visibility",
        (d) =>
          (d.entities.find((e) => e.id === key.slice(8)).hidden = !el.checked),
      );
    else {
      if (key.startsWith("layer.")) state.layers[key.slice(6)] = el.checked;
      else state[key] = el.checked;
      viewport.update();
      if (!$("#modal").open) render();
    }
  }
  if (el.dataset.choice) {
    const key = el.dataset.choice;
    if (key === "font") {
      state.font = parseInt(el.value);
      document.documentElement.style.fontSize = state.font + "px";
      document
        .querySelectorAll('[data-choice="font"]')
        .forEach((s) => (s.value = state.font + " px"));
      state.drawer = null;
      render();
      if (
        $("#modal").open &&
        $("#dialog-title").textContent === "Design system"
      )
        components();
    } else if (key === "projection") {
      state.ortho = el.value === "Orthographic";
      viewport.projectMode(state.ortho);
      renderViewport();
    } else {
      state[key] = key === "speed" ? Number(el.value) : el.value;
      viewport.update();
      if (key === "sensor") drawPlot();
    }
  }
});
document.addEventListener("keydown", (e) => {
  const editing = e.target.matches("input,select,textarea"),
    modal = $("#modal").open,
    pop = !$("#popover").hidden;
  if (e.key === "Escape") {
    if (modal) {
      e.preventDefault();
      closeDialog();
    } else if (pop) {
      e.preventDefault();
      closePopover(true);
    } else if (editing) {
      e.preventDefault();
      if (e.target.dataset.poseRange && dragStart) {
        state.previewPose = dragStart;
        dragStart = null;
        viewport.update();
      }
      render();
    } else {
      viewport.cancelTransform();
      state.drawer = null;
      layout();
    }
    return;
  }
  if (pop && e.key === "Tab") {
    closePopover(true);
    return;
  }
  if (pop && ["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) {
    e.preventDefault();
    const items = [...$("#popover").querySelectorAll("button:not(:disabled)")],
      i = items.indexOf(document.activeElement);
    items[
      e.key === "Home"
        ? 0
        : e.key === "End"
          ? items.length - 1
          : (i + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length
    ]?.focus();
    return;
  }
  if (
    e.target.getAttribute("role") === "tab" &&
    ["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)
  ) {
    e.preventDefault();
    const tabs = [
        ...e.target
          .closest('[role="tablist"]')
          .querySelectorAll('[role="tab"]'),
      ],
      i = tabs.indexOf(e.target),
      next =
        tabs[
          e.key === "Home"
            ? 0
            : e.key === "End"
              ? tabs.length - 1
              : (i + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) %
                tabs.length
        ],
      a = next.dataset.action;
    action(a, next);
    document.querySelector(`[data-action="${a}"]`)?.focus();
    return;
  }
  if (e.target.dataset.resize && ["ArrowLeft", "ArrowRight"].includes(e.key)) {
    e.preventDefault();
    resizePanel(
      e.target.dataset.resize,
      (e.key === "ArrowRight" ? 10 : -10) *
        (e.target.dataset.resize === "right" ? -1 : 1),
    );
    return;
  }
  if (editing) {
    if (e.key === "Enter" && e.target.dataset.bind) {
      e.preventDefault();
      setNumeric(e.target);
    }
    if (e.key === "Enter" && e.target.id === "entity-name") {
      e.preventDefault();
      dialogHandlers["rename-commit"]();
    }
    return;
  }
  if (modal || pop) return;
  const mod = e.metaKey || e.ctrlKey;
  if (mod && e.key.toLowerCase() === "z") {
    e.preventDefault();
    historyAction(e.shiftKey);
    return;
  }
  if (mod && e.key.toLowerCase() === "s") {
    e.preventDefault();
    saveLocal();
    return;
  }
  if (
    e.target.matches('button,[role="button"]') &&
    [" ", "Enter"].includes(e.key)
  )
    return;
  const shortcuts = {
    t: "collapse",
    k: "add-key",
    v: "tool:select",
    w: "tool:move",
    e: "tool:rotate",
    r: "tool:scale",
    f: "frame",
    b: "frame-mode",
    s: "snap",
    " ": "play",
    F2: "rename:" + state.selected,
    Delete: "delete:" + state.selected,
    Backspace: "delete:" + state.selected,
  };
  if (e.shiftKey && e.key.toLowerCase() === "d") {
    e.preventDefault();
    action("duplicate:" + state.selected, e.target);
    return;
  }
  if (!mod && shortcuts[e.key]) {
    e.preventDefault();
    action(shortcuts[e.key], e.target);
  }
});
function resizePanel(side, delta) {
  widths[side] = clamp(
    widths[side] + delta,
    side === "left" ? 240 : 290,
    side === "left" ? 380 : 420,
  );
  layout();
  renderBottom();
}
let resizing = null;
document.addEventListener("pointerdown", (e) => {
  const split = e.target.closest("[data-resize]");
  if (!split) return;
  resizing = {
    side: split.dataset.resize,
    x: e.clientX,
    value: widths[split.dataset.resize],
    el: split,
  };
  split.setPointerCapture(e.pointerId);
  document.body.classList.add("resizing");
});
document.addEventListener("pointermove", (e) => {
  if (!resizing) return;
  const delta =
    ((e.clientX - resizing.x) * (resizing.side === "right" ? -1 : 1)) /
    (state.font / 12);
  widths[resizing.side] = resizing.value;
  resizePanel(resizing.side, delta);
});
for (const event of ["pointerup", "pointercancel"])
  document.addEventListener(event, () => {
    resizing = null;
    document.body.classList.remove("resizing");
  });
$("#modal").addEventListener("click", (e) => {
  if (e.target === $("#modal")) {
    const r = e.target.getBoundingClientRect();
    if (
      e.clientX < r.left ||
      e.clientX > r.right ||
      e.clientY < r.top ||
      e.clientY > r.bottom
    )
      closeDialog();
  }
});
$("#modal").addEventListener("cancel", () => {
  dialogOpener?.isConnected && dialogOpener.focus();
});
window.addEventListener("resize", () => {
  closePopover();
  layout();
  renderBottom();
});
function updateTime() {
  const t = $('[data-bind="time"]');
  if (t && t !== document.activeElement) t.value = state.time.toFixed(3);
  const seekInput = $("[data-seek]");
  if (seekInput && seekInput !== document.activeElement)
    seekInput.value = state.time;
  const head = $(".playhead");
  if (head) head.style.left = state.time * 10 + "%";
  for (const j of JOINTS) {
    document
      .querySelectorAll(`[data-sensor-value="${j.key}"]`)
      .forEach((e) => (e.textContent = state.pose[j.key].toFixed(2)));
    if (state.playing)
      document
        .querySelectorAll(
          `[data-bind="pose.${j.key}"],[data-pose-range="${j.key}"]`,
        )
        .forEach((e) => (e.value = Number(state.pose[j.key].toFixed(2))));
  }
  drawPlot();
}
viewport = createViewport(
  $("#viewport"),
  state,
  selectEntity,
  (dt) => {
    if (state.playing) {
      let next = state.time + dt * state.speed,
        stopped = false;
      if (next >= history.document.duration) {
        if (state.loop) next %= history.document.duration;
        else {
          next = history.document.duration;
          state.playing = false;
          stopped = true;
        }
      }
      state.time = next;
      state.previewPose = samplePose(history.document.keys, next);
      viewport?.update();
      if (stopped) {
        renderTransport();
        renderFinder();
        renderInspector();
      }
    }
    lastTick += dt;
    if (lastTick > 0.08) {
      lastTick = 0;
      updateTime();
      if (state.panel === "Statistics") {
        const m = viewport?.metrics();
        for (const key of ["calls", "triangles"]) {
          const el = $(`[data-metric="${key}"]`);
          if (el && m) el.textContent = m[key];
        }
      }
    }
  },
  (phase, value) => {
    if (phase === "start") {
      state.playing = false;
      return;
    }
    if (phase === "preview" && value) {
      for (const key of ["position", "rotation", "scale"])
        value[key].forEach((v, i) => {
          const el = $(
            `[data-bind="entity.${key}.${i}"][data-target="${value.id}"]`,
          );
          if (el) el.value = Number(v.toFixed(3));
        });
    }
    if (phase === "commit" && value)
      transact("Transform entity", (d) =>
        Object.assign(
          d.entities.find((e) => e.id === value.id),
          {
            position: value.position,
            rotation: value.rotation,
            scale: value.scale,
          },
        ),
      );
  },
);
$("#popover").hidden = true;
render();
