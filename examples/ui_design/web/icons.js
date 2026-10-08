import { conceptIcons } from "./concept-icons.js";
const paths = {
  right: '<path d="m9 6 6 6-6 6"/>',
  scene:
    '<path d="M5 4v15h5M5 8h5"/><rect x="11" y="5" width="8" height="6" rx="1.5"/><rect x="11" y="15" width="8" height="6" rx="1.5"/>',
  pose: '<circle cx="12" cy="12" r="3"/><path d="M12 3v6m0 6v6M4 7l5 3m6 4 5 3"/>',
  control:
    '<path d="M4 7h9m5 0h2M4 17h3m5 0h8"/><circle cx="15.5" cy="7" r="2.5"/><circle cx="9.5" cy="17" r="2.5"/>',
  assets: '<path d="M3 8h7l2-3h8v15H3z"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5zM3 13l9 5 9-5M3 18l9 5 9-5"/>',
  chart: '<path d="M4 4v16h16M7 15l4-5 4 3 5-7"/>',
  terminal:
    '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="m6 8 4 4-4 4m7 0h5"/>',
  box: '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9zM4 7.5l8 4.5 8-4.5M12 12v9"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  check: '<path d="m5 12 4.5 4.5L19 7"/>',
  select: '<path d="m6 3 13 10-7 1-4 7z"/>',
  settings:
    '<path d="M4 7h16M4 17h16"/><rect x="7" y="4" width="4" height="6" rx="1" fill="var(--surface)"/><rect x="14" y="14" width="4" height="6" rx="1" fill="var(--surface)"/>',
  undo: '<path d="m8 5-5 5 5 5M3 10h11a5 5 0 0 1 0 10h-3"/>',
  redo: '<path d="m16 5 5 5-5 5m5-5H10a5 5 0 0 0 0 10h3"/>',
  pin: '<path d="M8 3h8l-1 6 3 4H6l3-4zM12 13v8"/>',
  duplicate:
    '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V4H4v12h4"/>',
  delete: '<path d="M4 6h16M9 3h6M6 6l1 15h10l1-15M10 10v7m4-7v7"/>',
  save: '<path d="M4 3h13l4 4v14H4zM8 3v6h9V3M8 21v-8h9v8"/>',
  help: '<circle cx="12" cy="12" r="9"/><path d="M9 8a3 3 0 0 1 6 0c0 2-3 2-3 5"/><circle cx="12" cy="17" r=".8" fill="currentColor" stroke="none"/>',
  inspect:
    '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M14 4v16M17 8h1m-1 4h1"/>',
  loop: '<path d="M5 7h12l3 3m-3-6 3 6h-6M19 17H7l-3-3m3 6-3-6h6"/>',
  export: '<path d="M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5"/>',
  circle: '<circle cx="12" cy="12" r="8"/>',
  more: '<circle cx="5" cy="12" r="1.5" fill="currentColor"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/><circle cx="19" cy="12" r="1.5" fill="currentColor"/>',
};
const aliases = {
  scene: "tree",
  pose: "joint",
  control: "sliders",
  assets: "folder",
  settings: "gear",
  body: "cube",
  snap: "magnet",
  previous: "stepb",
  next: "stepf",
  record: "rec",
  key: "diamond",
  addkey: "diamond",
  close: "x",
  right: "right",
  down: "chev",
  perspective: "camera",
  orthographic: "cube",
  warning: "warn",
  error: "err",
  frame: "fit",
  duplicate: "copy",
  export: "upload",
  "transport-first": "reset",
};
export function icon(name) {
  const glyph = conceptIcons[aliases[name] || name] || paths[name] || paths.box;
  return `<svg class="icon" viewBox="0 0 24 24" preserveAspectRatio="xMidYMid meet" aria-hidden="true" focusable="false" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">${glyph}</svg>`;
}
export const iconNames = [...Object.keys(conceptIcons), ...Object.keys(paths)];
