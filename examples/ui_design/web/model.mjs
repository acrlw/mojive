// Document operations are atomic; transient UI state is deliberately separate.
export const clone = (value) => structuredClone(value);
export const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
export const JOINTS = [
  {
    key: "hinge",
    name: "hinge_body",
    label: "Hinge",
    min: -90,
    max: 90,
    step: 0.1,
    unit: "°",
  },
  {
    key: "ball",
    name: "ball_body",
    label: "Ball · Y axis",
    min: -60,
    max: 60,
    step: 0.1,
    unit: "°",
  },
  {
    key: "slide",
    name: "slide_body",
    label: "Slide",
    min: -0.6,
    max: 0.6,
    step: 0.01,
    unit: "m",
  },
];
export function initialDocument() {
  const entities = [
    ["free_body", "link", "box", [-2.1, 0, 0.8], "#d89556"],
    ["ball_body", "link", "ball", [-1.15, 0.25, 1.1], "#68b5b7"],
    ["slide_body", "link", "slide", [-0.2, 0, 0.55], "#8dc785"],
    ["hinge_body", "link", "hinge", [0.9, 0, 0.9], "#e3ce62"],
    ["hinge_unlimited", "link", "hinge", [0.9, 1, 0.9], "#d6b954"],
    ["chain_root", "link", "chain", [2.1, 0.35, 0.35], "#b299d9"],
    ["sun", "light", "light", [-3, -4, 8], "#ffeedb"],
    ["inspection", "camera", "camera", [2.5, -1.7, 1.7], "#9eafc5"],
  ].map(([id, type, shape, position, color]) => ({
    id,
    name: id,
    type,
    shape,
    position,
    rotation: [0, 0, 0],
    scale: [1, 1, 1],
    color,
    roughness: 0.5,
    hidden: false,
  }));
  return {
    name: "Joint study",
    entities,
    pose: { hinge: -35, ball: 20, slide: 0.35 },
    keys: [
      {
        id: "key-0",
        name: "Rest",
        time: 0,
        pose: { hinge: 0, ball: 0, slide: 0 },
      },
      {
        id: "key-1",
        name: "Reach",
        time: 4,
        pose: { hinge: -35, ball: 20, slide: 0.35 },
      },
      {
        id: "key-2",
        name: "Return",
        time: 8,
        pose: { hinge: 0, ball: 0, slide: 0 },
      },
    ],
    duration: 10,
  };
}
export function samplePose(keys, time) {
  if (!keys.length) return null;
  const sorted = [...keys].sort((a, b) => a.time - b.time);
  if (time <= sorted[0].time) return clone(sorted[0].pose);
  for (let i = 1; i < sorted.length; i++)
    if (time <= sorted[i].time) {
      const a = sorted[i - 1],
        b = sorted[i],
        f = (time - a.time) / (b.time - a.time);
      return Object.fromEntries(
        JOINTS.map((j) => [
          j.key,
          a.pose[j.key] + (b.pose[j.key] - a.pose[j.key]) * f,
        ]),
      );
    }
  return clone(sorted.at(-1).pose);
}
export function numericValue(
  raw,
  { min = -10000, max = 10000, integer = false } = {},
) {
  if (typeof raw === "string" && !raw.trim()) throw new Error("Enter a value.");
  const n = Number(raw);
  if (!Number.isFinite(n)) throw new Error("Enter a finite number.");
  if (n < min || n > max) throw new Error(`Use a value from ${min} to ${max}.`);
  if (integer && !Number.isInteger(n)) throw new Error("Use a whole number.");
  return n;
}
export function uniqueName(entities, base, exclude) {
  let name = base.trim();
  if (!name) throw new Error("Enter a name.");
  if (name.length > 80) throw new Error("Use 80 characters or fewer.");
  if (
    entities.some(
      (e) => e.id !== exclude && e.name.toLowerCase() === name.toLowerCase(),
    )
  )
    throw new Error("This name is already in use.");
  return name;
}
export class History {
  constructor(doc) {
    this.document = clone(doc);
    this.past = [];
    this.future = [];
    this.saved = JSON.stringify(doc);
  }
  commit(label, change) {
    const before = clone(this.document),
      next = clone(before);
    change(next);
    if (JSON.stringify(before) === JSON.stringify(next)) return false;
    this.past.push({ label, document: before });
    if (this.past.length > 60) this.past.shift();
    this.document = next;
    this.future = [];
    return true;
  }
  undo() {
    if (!this.past.length) return false;
    const p = this.past.pop();
    this.future.push({ label: p.label, document: clone(this.document) });
    this.document = p.document;
    return p.label;
  }
  redo() {
    if (!this.future.length) return false;
    const p = this.future.pop();
    this.past.push({ label: p.label, document: clone(this.document) });
    this.document = p.document;
    return p.label;
  }
  get dirty() {
    return JSON.stringify(this.document) !== this.saved;
  }
  markSaved() {
    this.saved = JSON.stringify(this.document);
  }
}
export function upsertKey(doc, time, id) {
  const existing = doc.keys.find((k) => Math.abs(k.time - time) < 0.025);
  if (existing) existing.pose = clone(doc.pose);
  else
    doc.keys.push({
      id,
      name: `Pose ${doc.keys.length + 1}`,
      time: Number(time.toFixed(2)),
      pose: clone(doc.pose),
    });
  doc.keys.sort((a, b) => a.time - b.time);
  return (existing || doc.keys.find((k) => k.id === id)).id;
}
// Dense keys share a chooser rather than overlapping their hit targets.
export function groupKeys(keys, trackWidth, duration, minimumGap = 34) {
  const groups = [];
  for (const key of [...keys].sort((a, b) => a.time - b.time)) {
    const previous = groups.at(-1);
    if (
      previous &&
      ((key.time - previous.at(-1).time) / duration) * trackWidth < minimumGap
    )
      previous.push(key);
    else groups.push([key]);
  }
  return groups;
}
