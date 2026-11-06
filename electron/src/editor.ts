import * as THREE from 'three';
import type { SceneDocument } from './contracts';
import type { PhysicsSnapshot } from './physics/contracts';
import { setJoint, setJointConfiguration } from './scene';
export interface Pose {
  transforms: Record<string, { p: number[]; q: number[]; s: number[] }>;
  joints: Record<string, number>;
  jointOrigins?: Record<string, number[]>;
  jointConfigurations?: Record<string, { position: number[]; rotation: number[] }>;
}
export interface RecordedSceneState {
  pose: Pose;
  simulation?: PhysicsSnapshot;
}
export interface Keyframe extends RecordedSceneState {
  id: string;
  name?: string;
  time: number;
}
export interface SceneSnapshot extends RecordedSceneState {
  id: string;
  name: string;
  time: number;
}
export interface SceneTake {
  id: string;
  name: string;
  frames: ({ time: number } & RecordedSceneState)[];
}
export interface WorkspaceTimeline {
  snapshots: SceneSnapshot[];
  takes: SceneTake[];
  activeTakeId: string | null;
}
/** Return a nearest recorded frame; an exact midpoint belongs to its earlier frame. */
export function sampleTake(
  doc: SceneDocument,
  take: SceneTake,
  time: number,
  mode: 'nearest' | 'floor' = 'nearest',
): number {
  const index = takeFrameIndex(take, time, mode);
  if (index < 0) return index;
  assertPoseCompatible(doc, take.frames[index].pose);
  applyPose(doc, take.frames[index].pose);
  return index;
}
export function takeFrameIndex(
  take: SceneTake, time: number, mode: 'nearest' | 'floor' = 'nearest',
): number {
  if (!Number.isFinite(time)) throw Error('Invalid take sample time');
  if (mode !== 'nearest' && mode !== 'floor') throw Error('Invalid take sampling mode');
  const frames = take.frames;
  if (!frames.length) return -1;
  let low = 0,
    high = frames.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (frames[middle].time < time) low = middle + 1;
    else high = middle;
  }
  const index =
    mode === 'floor'
      ? Math.max(0, low < frames.length && frames[low].time === time ? low : low - 1)
      : low === 0
        ? 0
        : low === frames.length
          ? frames.length - 1
          : time - frames[low - 1].time <= frames[low].time - time
            ? low - 1
            : low;
  return index;
}
const own = <T>(values: Record<string, T> | undefined, id: string): T | undefined =>
  values && Object.hasOwn(values, id) ? values[id] : undefined;
/** Reject a partial or structurally stale full-scene restore before mutating any object. */
export function assertPoseCompatible(doc: SceneDocument, state: Pose): void {
  pose(state);
  const matching = (
    expected: string[],
    actual: Record<string, unknown> | undefined,
    label: string,
  ) => {
    const ids = Object.keys(actual ?? {}),
      known = new Set(expected);
    if (ids.length !== known.size || ids.some((id) => !known.has(id)))
      throw Error(`This pose is incompatible with the current scene (${label}).`);
  };
  matching(
    doc.nodes.map((node) => node.id),
    state.transforms,
    'objects',
  );
  matching(
    doc.joints.map((joint) => joint.id),
    state.joints,
    'joints',
  );
  matching(
    doc.joints.map((joint) => joint.id),
    state.jointOrigins,
    'joint origins',
  );
  matching(
    doc.joints
      .filter((joint) => joint.type === 'ball' || joint.type === 'free')
      .map((joint) => joint.id),
    state.jointConfigurations,
    'joint configurations',
  );
  for (const joint of doc.joints) {
    if (
      joint.type === 'ball' &&
      Math.hypot(...state.jointConfigurations![joint.id].position) > 1e-12
    )
      throw Error('A ball joint pose cannot translate');
  }
}
export function capturePose(doc: SceneDocument): Pose {
  return {
    transforms: Object.fromEntries(
      doc.nodes.map((n) => [
        n.id,
        {
          p: n.object.position.toArray(),
          q: n.object.quaternion.toArray(),
          s: n.object.scale.toArray(),
        },
      ]),
    ),
    joints: Object.fromEntries(doc.joints.map((j) => [j.id, j.value])),
    jointOrigins: Object.fromEntries(doc.joints.map((j) => [j.id, j.origin.toArray()])),
    jointConfigurations: Object.fromEntries(
      doc.joints
        .filter((j) => j.type === 'ball' || j.type === 'free')
        .map((j) => [j.id, { position: j.position!.toArray(), rotation: j.rotation!.toArray() }]),
    ),
  };
}
export function applyPose(doc: SceneDocument, pose: Pose, localOnly = false) {
  for (const joint of doc.joints) {
    if (localOnly && joint.object.userData.physicsOwned) continue;
    const origin = own(pose.jointOrigins, joint.id);
    if (origin) joint.origin.fromArray(origin);
  }
  for (const node of doc.nodes) {
    if (localOnly && node.object.userData.physicsOwned) continue;
    const t = own(pose.transforms, node.id);
    if (t) {
      node.object.position.fromArray(t.p);
      node.object.quaternion.fromArray(t.q);
      node.object.scale.fromArray(t.s);
    }
  }
  for (const joint of doc.joints) {
    if (localOnly && joint.object.userData.physicsOwned) continue;
    // Detached replay/export scenes retain published transforms; joint proxies are not a kinematic chain.
    if (joint.object.userData.physicsOwned) {
      joint.value = pose.joints[joint.id];
      const configuration = own(pose.jointConfigurations, joint.id);
      if (configuration) {
        joint.position?.fromArray(configuration.position);
        joint.rotation?.fromArray(configuration.rotation);
      }
      continue;
    }
    if (joint.type === 'ball' || joint.type === 'free') {
      const config = own(pose.jointConfigurations, joint.id);
      if (config) setJointConfiguration(doc, joint.id, config);
      continue;
    }
    if (own(pose.joints, joint.id) !== undefined && !joint.mimic)
      setJoint(doc, joint.id, pose.joints[joint.id]);
  }
  doc.root.updateMatrixWorld(true);
}
export function samplePose(doc: SceneDocument, keys: Keyframe[], time: number) {
  if (!keys.length) return;
  const ordered = [...keys].sort((a, b) => a.time - b.time);
  let a = ordered[0],
    b = ordered[ordered.length - 1];
  if (time <= a.time) {
    applyPose(doc, a.pose);
    return;
  }
  if (time >= b.time) {
    applyPose(doc, b.pose);
    return;
  }
  for (let i = 1; i < ordered.length; i++) {
    if (ordered[i].time >= time) {
      a = ordered[i - 1];
      b = ordered[i];
      break;
    }
  }
  const t = (time - a.time) / (b.time - a.time);
  const pose: Pose = {
    transforms: Object.create(null),
    joints: Object.create(null),
    jointOrigins: Object.create(null),
    jointConfigurations: Object.create(null),
  };
  for (const id in a.pose.transforms) {
    const x = a.pose.transforms[id],
      y = own(b.pose.transforms, id) ?? x;
    pose.transforms[id] = {
      p: x.p.map((v, i) => THREE.MathUtils.lerp(v, y.p[i], t)),
      s: x.s.map((v, i) => THREE.MathUtils.lerp(v, y.s[i], t)),
      q: new THREE.Quaternion()
        .fromArray(x.q)
        .slerp(new THREE.Quaternion().fromArray(y.q), t)
        .toArray(),
    };
  }
  for (const id in a.pose.joints)
    pose.joints[id] = THREE.MathUtils.lerp(
      a.pose.joints[id],
      own(b.pose.joints, id) ?? a.pose.joints[id],
      t,
    );
  const origins = new Set([
    ...Object.keys(a.pose.jointOrigins ?? {}),
    ...Object.keys(b.pose.jointOrigins ?? {}),
  ]);
  for (const id of origins) {
    const x = own(a.pose.jointOrigins, id) ?? own(b.pose.jointOrigins, id)!,
      y = own(b.pose.jointOrigins, id) ?? x;
    const ap = new THREE.Vector3(),
      aq = new THREE.Quaternion(),
      as = new THREE.Vector3();
    const bp = new THREE.Vector3(),
      bq = new THREE.Quaternion(),
      bs = new THREE.Vector3();
    new THREE.Matrix4().fromArray(x).decompose(ap, aq, as);
    new THREE.Matrix4().fromArray(y).decompose(bp, bq, bs);
    pose.jointOrigins![id] = new THREE.Matrix4()
      .compose(ap.lerp(bp, t), aq.slerp(bq, t), as.lerp(bs, t))
      .toArray();
  }
  for (const id of new Set([
    ...Object.keys(a.pose.jointConfigurations ?? {}),
    ...Object.keys(b.pose.jointConfigurations ?? {}),
  ])) {
    const x = own(a.pose.jointConfigurations, id) ?? own(b.pose.jointConfigurations, id)!,
      y = own(b.pose.jointConfigurations, id) ?? x;
    pose.jointConfigurations![id] = {
      position: x.position.map((v, i) => THREE.MathUtils.lerp(v, y.position[i], t)),
      rotation: new THREE.Quaternion()
        .fromArray(x.rotation)
        .slerp(new THREE.Quaternion().fromArray(y.rotation), t)
        .toArray(),
    };
  }
  applyPose(doc, pose);
}
export function encodeDocument(doc: SceneDocument) {
  // toJSON reads cached local matrices; a commit can occur before the next render.
  doc.root.updateMatrixWorld(true);
  // Three's toJSON retains userData references. History snapshots must not alias
  // live metadata (or node properties/files) that a later editor command mutates.
  return structuredClone({
    name: doc.name,
    source: doc.source,
    files: doc.files,
    warnings: doc.warnings,
    root: doc.root.toJSON(),
    nodes: doc.nodes.map(({ object, ...n }) => ({ ...n, uuid: object.uuid })),
    joints: doc.joints.map(({ object, axis, origin, position, rotation, ...j }) => ({
      ...j,
      lower: j.lower === -Infinity ? null : j.lower,
      upper: j.upper === Infinity ? null : j.upper,
      uuid: object.uuid,
      axis: axis.toArray(),
      origin: origin.toArray(),
      position: position?.toArray(),
      rotation: rotation?.toArray(),
    })),
  });
}
export function decodeDocument(data: ReturnType<typeof encodeDocument>): SceneDocument {
  const snapshot = structuredClone(data);
  const root = new THREE.ObjectLoader().parse(snapshot.root) as THREE.Group;
  const find = (uuid: string) => {
    const o = root.getObjectByProperty('uuid', uuid);
    if (!o) throw Error(`Scene object ${uuid} is missing`);
    return o;
  };
  return {
    name: snapshot.name,
    source: snapshot.source,
    files: snapshot.files,
    warnings: snapshot.warnings,
    root,
    nodes: snapshot.nodes.map(({ uuid, ...n }) => ({ ...n, object: find(uuid) })),
    joints: snapshot.joints.map(({ uuid, axis, origin, position, rotation, ...j }) => ({
      ...j,
      lower: j.lower ?? -Infinity,
      upper: j.upper ?? Infinity,
      object: find(uuid),
      axis: new THREE.Vector3().fromArray(axis),
      origin: new THREE.Matrix4().fromArray(origin),
      position: position ? new THREE.Vector3().fromArray(position) : undefined,
      rotation: rotation ? new THREE.Quaternion().fromArray(rotation) : undefined,
    })),
  };
}
export function saveWorkspace(
  doc: SceneDocument,
  keys: Keyframe[],
  duration: number,
  timeline?: WorkspaceTimeline,
) {
  const document = encodeDocument(doc);
  if (timeline !== undefined) {
    finiteJson(timeline, 'timeline');
    validateTimeline(timeline, document, duration);
  }
  return JSON.stringify(
    {
      format: 'mojive-electron',
      version: 1,
      document,
      keys,
      duration,
      ...(timeline ? { timeline } : {}),
    },
    null,
    2,
  );
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value))
    throw Error(`Invalid ${label}`);
  return value as Record<string, unknown>;
}
function text(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.length) throw Error(`Invalid ${label}`);
  return value;
}
function number(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw Error(`Invalid ${label}`);
  return value;
}
function vector(value: unknown, length: number, label: string): number[] {
  if (!Array.isArray(value) || value.length !== length) throw Error(`Invalid ${label}`);
  return value.map((v) => number(v, label));
}
function transform(value: unknown, label: string): number[] {
  const matrix = vector(value, 16, label);
  if (
    Math.abs(matrix[3]) + Math.abs(matrix[7]) + Math.abs(matrix[11]) + Math.abs(matrix[15] - 1) >
      1e-8 ||
    new THREE.Matrix4().fromArray(matrix).determinant() === 0
  )
    throw Error(`Invalid ${label}`);
  return matrix;
}
function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw Error(`Invalid ${label}`);
  return value;
}
function finiteJson(value: unknown, label: string, depth = 0): void {
  if (depth > 128) throw Error(`${label} is nested too deeply`);
  if (typeof value === 'number') number(value, label);
  else if (Array.isArray(value)) value.forEach((v) => finiteJson(v, label, depth + 1));
  else if (value && typeof value === 'object')
    Object.values(value).forEach((v) => finiteJson(v, label, depth + 1));
}
function files(value: unknown): void {
  for (const item of array(value, 'project files')) {
    const f = record(item, 'project file');
    text(f.name, 'project file name');
    text(f.path, 'project file path');
    if (f.kind !== 'file' && f.kind !== 'directory') throw Error('Invalid project file kind');
    if (f.children !== undefined) files(f.children);
  }
}
function pose(value: unknown): void {
  const p = record(value, 'keyframe pose');
  for (const [id, value] of Object.entries(record(p.transforms, 'pose transforms'))) {
    text(id, 'pose object ID');
    const transform = record(value, 'pose transform');
    vector(transform.p, 3, 'pose position');
    const q = vector(transform.q, 4, 'pose rotation');
    if (Math.abs(Math.hypot(...q) - 1) > 1e-6) throw Error('Pose rotation must be normalized');
    const s = vector(transform.s, 3, 'pose scale');
    if (s.some((v) => v === 0)) throw Error('Invalid pose scale');
  }
  for (const [id, value] of Object.entries(record(p.joints, 'pose joints'))) {
    text(id, 'pose joint ID');
    number(value, 'joint position');
  }
  if (p.jointConfigurations !== undefined)
    for (const [id, value] of Object.entries(
      record(p.jointConfigurations, 'pose joint configurations'),
    )) {
      text(id, 'pose joint ID');
      const config = record(value, 'joint configuration');
      vector(config.position, 3, 'joint position');
      const q = vector(config.rotation, 4, 'joint rotation');
      if (Math.abs(Math.hypot(...q) - 1) > 1e-6) throw Error('Joint rotation must be normalized');
    }
  if (p.jointOrigins !== undefined)
    for (const [id, value] of Object.entries(record(p.jointOrigins, 'pose joint origins'))) {
      text(id, 'pose joint ID');
      transform(value, 'joint origin');
    }
}
function validateDocument(value: unknown): void {
  const doc = record(value, 'scene document');
  text(doc.name, 'scene name');
  if (doc.source !== undefined && typeof doc.source !== 'string')
    throw Error('Invalid URDF source');
  files(doc.files);
  array(doc.warnings, 'scene warnings').forEach((v) => {
    if (typeof v !== 'string') throw Error('Invalid scene warning');
  });
  const root = record(doc.root, 'scene graph');
  const rootObject = record(root.object, 'scene root');
  if (rootObject.type !== 'Group') throw Error('Scene root must be a collection group');
  const objects = new Map<string, { data: Record<string, unknown>; parent?: string }>();
  const resources = (name: string) => {
    const ids = new Set<string>();
    for (const item of root[name] === undefined ? [] : array(root[name], name)) {
      const id = text(record(item, name).uuid, `${name} UUID`);
      if (ids.has(id)) throw Error(`Duplicate ${name} UUID`);
      ids.add(id);
    }
    return ids;
  };
  const geometryIds = resources('geometries'),
    materialIds = resources('materials');
  resources('textures');
  resources('images');
  const visit = (value: unknown, parent?: string) => {
    const object = record(value, 'scene object'),
      id = text(object.uuid, 'object UUID');
    text(object.type, 'object type');
    if (objects.has(id)) throw Error('Duplicate object UUID');
    objects.set(id, { data: object, parent });
    if (object.matrix !== undefined) transform(object.matrix, 'object transform');
    if (object.visible !== undefined && typeof object.visible !== 'boolean')
      throw Error('Invalid object visibility');
    if (
      object.geometry !== undefined &&
      !geometryIds.has(text(object.geometry, 'geometry reference'))
    )
      throw Error('Unknown object geometry');
    if (object.material !== undefined)
      for (const material of Array.isArray(object.material) ? object.material : [object.material])
        if (!materialIds.has(text(material, 'material reference')))
          throw Error('Unknown object material');
    if (object.children !== undefined)
      array(object.children, 'object children').forEach((child) => visit(child, id));
  };
  visit(rootObject);
  const nodes = new Map<string, Record<string, unknown>>(),
    links = new Map<string, Record<string, unknown>>(),
    nodeObjects = new Set<string>();
  const kinds = new Set([
    'robot',
    'world',
    'geom',
    'link',
    'joint',
    'box',
    'sphere',
    'ellipsoid',
    'cylinder',
    'cone',
    'capsule',
    'ground',
    'light',
    'camera',
    'environment',
  ]);
  for (const value of array(doc.nodes, 'scene nodes')) {
    const node = record(value, 'scene node'),
      id = text(node.id, 'object ID'),
      name = text(node.name, 'object name'),
      uuid = text(node.uuid, 'node UUID');
    if (nodes.has(id) || nodeObjects.has(uuid)) throw Error('Duplicate scene node identity');
    nodes.set(id, node);
    nodeObjects.add(uuid);
    if (!kinds.has(text(node.kind, 'object kind')) || typeof node.visible !== 'boolean')
      throw Error('Invalid scene node');
    const object = objects.get(uuid)?.data;
    if (!object || object === rootObject)
      throw Error(`Scene object ${uuid} is missing or is the collection root`);
    const metadata = record(object.userData, 'object metadata');
    const physicsOwned = metadata.physicsOwned === true;
    if (physicsOwned && !record(rootObject.userData, 'scene metadata').physicsSourceId)
      throw Error('Physics entities require a source identity');
    const expectedType = ['robot', 'world', 'link', 'joint', 'environment'].includes(node.kind as string) ||
      (node.kind === 'geom' && !physicsOwned)
      ? 'Group'
      : node.kind === 'light'
        ? 'PointLight'
        : node.kind === 'camera'
          ? 'PerspectiveCamera'
          : 'Mesh';
    if (
      node.kind === 'light'
        ? ![
            'PointLight',
            'DirectionalLight',
            'HemisphereLight',
            'SpotLight',
            'RectAreaLight',
          ].includes(object.type as string)
        : node.kind === 'camera'
          ? !['PerspectiveCamera', 'OrthographicCamera'].includes(object.type as string)
          : object.type !== expectedType
    )
      throw Error('Scene node kind does not match its object type');
    if (node.kind === 'light') {
      const intensity = number(object.intensity, 'light intensity');
      if (intensity < 0 || intensity > 1e6) throw Error('Invalid light intensity');
      if (['PointLight', 'SpotLight'].includes(object.type as string)) {
        if (number(object.distance, 'light range') < 0 || number(object.decay, 'light decay') < 0)
          throw Error('Invalid light attenuation');
      }
      if (
        object.type === 'SpotLight' &&
        (number(object.angle, 'spot angle') <= 0 ||
          number(object.angle, 'spot angle') > Math.PI / 2 ||
          number(object.penumbra, 'spot penumbra') < 0 ||
          number(object.penumbra, 'spot penumbra') > 1)
      )
        throw Error('Invalid spot cone');
      if (
        object.type === 'RectAreaLight' &&
        (number(object.width, 'area width') <= 0 || number(object.height, 'area height') <= 0)
      )
        throw Error('Invalid area emitter');
      if (object.target !== undefined && !objects.has(text(object.target, 'light target')))
        throw Error('Light target is missing');
    }
    if (node.kind === 'camera') {
      const near = number(object.near, 'camera near'),
        far = number(object.far, 'camera far');
      if (near < 1e-8 || far <= near || far > 1e9) throw Error('Invalid camera clipping range');
      if (object.type === 'PerspectiveCamera') {
        const fov = number(object.fov, 'camera FOV'),
          aspect = number(object.aspect, 'camera aspect');
        if (fov < 1 || fov > 179 || aspect <= 0) throw Error('Invalid camera projection');
      } else {
        const left = number(object.left, 'camera left'),
          right = number(object.right, 'camera right'),
          top = number(object.top, 'camera top'),
          bottom = number(object.bottom, 'camera bottom');
        if (right <= left || top <= bottom) throw Error('Invalid orthographic bounds');
      }
      if (object.zoom !== undefined && number(object.zoom, 'camera zoom') <= 0)
        throw Error('Invalid camera zoom');
      const metadata = record(object.userData, 'camera metadata').camera;
      if (metadata !== undefined) {
        const camera = record(metadata, 'camera settings'),
          resolution = vector(camera.resolution, 2, 'camera resolution');
        if (
          resolution.some((v) => !Number.isInteger(v) || v < 1 || v > 8192) ||
          resolution[0] * resolution[1] > 16777216
        )
          throw Error('Invalid camera resolution');
        for (const key of ['targetDistance', 'sensorWidth', 'orthoHeight'])
          if (number(camera[key], `camera ${key}`) <= 0) throw Error('Invalid camera optics');
        const fov = number(camera.fov, 'camera FOV');
        if (fov < 1 || fov > 179) throw Error('Invalid camera FOV');
      }
    }
    if (record(object.userData, 'object metadata').nodeId !== id)
      throw Error('Object ID does not match scene metadata');
    if (node.visible !== (object.visible !== false))
      throw Error('Scene visibility does not match object visibility');
    if (node.parentId !== undefined) text(node.parentId, 'parent object ID');
    if (node.properties !== undefined)
      for (const value of Object.values(record(node.properties, 'object properties')))
        if (typeof value !== 'string') number(value, 'object property');
    if (['link', 'world', 'robot'].includes(node.kind as string)) {
      if (links.has(name)) throw Error('Duplicate link name');
      links.set(name, node);
    }
  }
  for (const node of nodes.values()) {
    const entry = objects.get(node.uuid as string)!;
    const metadata = record(entry.data.userData, 'object metadata');
    // Engine-derived joint/camera/light poses are world-space proxies. Their
    // logical owner remains a body, without inventing a transform hierarchy.
    if (metadata.physicsWorldFrame === true) {
      if (metadata.physicsOwned !== true || !['joint', 'camera', 'light'].includes(node.kind as string) ||
        !nodes.has(node.parentId as string) || entry.parent !== rootObject.uuid)
        throw Error('Invalid physics world-frame proxy');
      continue;
    }
    const parent =
      node.parentId === undefined ? rootObject.uuid : nodes.get(node.parentId as string)?.uuid;
    if (!parent || objects.get(node.uuid as string)?.parent !== parent)
      throw Error('Scene hierarchy does not match object parents');
  }
  const joints = new Map<string, Record<string, unknown>>(),
    jointIds = new Set<string>();
  for (const value of array(doc.joints, 'scene joints')) {
    const joint = record(value, 'joint'),
      id = text(joint.id, 'joint ID'),
      name = text(joint.name, 'joint name'),
      node = nodes.get(id);
    if (joints.has(name) || jointIds.has(id)) throw Error('Duplicate joint identity');
    joints.set(name, joint);
    jointIds.add(id);
    if (!node || node.kind !== 'joint' || joint.uuid !== node.uuid || name !== node.name)
      throw Error('Joint does not match its scene node');
    if (
      !['fixed', 'continuous', 'revolute', 'prismatic', 'ball', 'free'].includes(
        text(joint.type, 'joint type'),
      )
    )
      throw Error('Invalid joint type');
    const axis = vector(joint.axis, 3, 'joint axis');
    if (Math.abs(Math.hypot(...axis) - 1) > 1e-6) throw Error('Joint axis must be normalized');
    transform(joint.origin, 'joint origin');
    if (joint.type === 'ball' || joint.type === 'free') {
      const position = vector(joint.position, 3, 'joint position');
      if (joint.type === 'ball' && Math.hypot(...position) > 1e-12)
        throw Error('A ball joint cannot translate');
      const q = vector(joint.rotation, 4, 'joint rotation');
      if (Math.abs(Math.hypot(...q) - 1) > 1e-6) throw Error('Joint rotation must be normalized');
      if (joint.mimic) throw Error('Multi-DOF joints cannot mimic scalar joints');
    }
    if (joint.angularLimit !== undefined && number(joint.angularLimit, 'joint angular limit') < 0)
      throw Error('Invalid angular limit');
    const physicsOwned = record(objects.get(node.uuid as string)!.data.userData, 'joint metadata').physicsOwned === true;
    const lower = physicsOwned && joint.lower === null ? -Infinity : number(joint.lower, 'joint lower limit'),
      upper = physicsOwned && joint.upper === null ? Infinity : number(joint.upper, 'joint upper limit');
    number(joint.value, 'joint position');
    if (lower > upper) throw Error('Reversed joint limits');
    const parent = links.get(text(joint.parent, 'joint parent')),
      child = links.get(text(joint.child, 'joint child'));
    if (!parent || !child) throw Error('Joint references a missing or disconnected link');
    if (physicsOwned) {
      if (joint.bodyId !== child.id || node.parentId !== child.id || child.parentId !== parent.id)
        throw Error('Physics joint owner does not match its body hierarchy');
    } else if (joint.bodyId !== undefined) {
      if (joint.bodyId !== child.id) throw Error('Joint body identity does not match its child');
      let cursor: Record<string, unknown> | undefined = child;
      const chain = new Set<string>();
      while (cursor && cursor.id !== parent.id) {
        if (chain.has(cursor.id as string)) throw Error('Joint chain contains a cycle');
        chain.add(cursor.id as string);
        cursor = nodes.get(cursor.parentId as string);
      }
      if (!cursor || !chain.has(id)) throw Error('Joint is disconnected from its body chain');
    } else if (node.parentId !== parent.id || child.parentId !== id)
      throw Error('Joint references a missing or disconnected link');
    if (joint.mimic !== undefined) {
      const mimic = record(joint.mimic, 'mimic joint');
      text(mimic.joint, 'mimic driver');
      number(mimic.multiplier, 'mimic multiplier');
      number(mimic.offset, 'mimic offset');
    }
  }
  for (const node of nodes.values())
    if (node.kind === 'joint' && !jointIds.has(node.id as string))
      throw Error('Joint node has no joint definition');
  for (const joint of joints.values()) {
    const seen = new Set<string>();
    let current: Record<string, unknown> | undefined = joint;
    while (current?.mimic) {
      const name = current.name as string;
      if (seen.has(name)) throw Error('Mimic joint cycle');
      seen.add(name);
      current = joints.get((current.mimic as Record<string, unknown>).joint as string);
      if (!current) throw Error('Unknown mimic driver');
    }
  }
  const metadata =
    rootObject.userData === undefined ? {} : record(rootObject.userData, 'scene metadata');
  if (metadata.actuators !== undefined)
    for (const value of array(metadata.actuators, 'actuators')) {
      const actuator = record(value, 'actuator');
      text(actuator.name, 'actuator name');
      if (!joints.has(text(actuator.joint, 'actuator joint')))
        throw Error('Actuator references a missing joint');
      if (actuator.mechanicalReduction !== undefined)
        number(actuator.mechanicalReduction, 'mechanical reduction');
    }
}
function validatePoseForDocument(value: unknown, document: ReturnType<typeof encodeDocument>) {
  pose(value);
  const state = value as Pose;
  for (const joint of document.joints) {
    if (joint.type !== 'ball' && joint.type !== 'free') continue;
    const config = own(state.jointConfigurations, joint.id);
    if (own(state.transforms, joint.id) && !config)
      throw Error('Multi-DOF pose is missing its joint configuration');
    if (joint.type === 'ball' && config && Math.hypot(...config.position) > 1e-12)
      throw Error('A ball joint pose cannot translate');
  }
}
function timelineTime(value: unknown, duration: number, label: string): number {
  const time = number(value, label);
  if (time < 0 || time > duration + 1e-9) throw Error(`Invalid ${label}`);
  return time;
}
function validateTimeline(
  value: unknown,
  document: ReturnType<typeof encodeDocument>,
  duration: number,
): WorkspaceTimeline {
  if (value === undefined) return { snapshots: [], takes: [], activeTakeId: null };
  const timeline = record(value, 'timeline');
  const snapshots = array(timeline.snapshots, 'snapshots'),
    takes = array(timeline.takes, 'takes');
  const ids = new Set<string>();
  let previous = -Infinity;
  for (const value of snapshots) {
    const snapshot = record(value, 'snapshot'),
      id = text(snapshot.id, 'snapshot ID');
    if (ids.has(id)) throw Error('Duplicate snapshot ID');
    ids.add(id);
    if (!text(snapshot.name, 'snapshot name').trim()) throw Error('Invalid snapshot name');
    const time = timelineTime(snapshot.time, duration, 'snapshot time');
    if (time < previous) throw Error('Snapshots must be sorted by time');
    previous = time;
    validatePoseForDocument(snapshot.pose, document);
  }
  const takeIds = new Set<string>();
  for (const value of takes) {
    const take = record(value, 'take'),
      id = text(take.id, 'take ID');
    if (takeIds.has(id)) throw Error('Duplicate take ID');
    takeIds.add(id);
    if (!text(take.name, 'take name').trim()) throw Error('Invalid take name');
    const frames = array(take.frames, 'take frames');
    let previous = -Infinity;
    for (const value of frames) {
      const frame = record(value, 'take frame'),
        time = timelineTime(frame.time, duration, 'take frame time');
      if (time <= previous) throw Error('Take frames must be strictly chronological');
      previous = time;
      validatePoseForDocument(frame.pose, document);
    }
  }
  if (
    timeline.activeTakeId !== null &&
    (typeof timeline.activeTakeId !== 'string' || !takeIds.has(timeline.activeTakeId))
  )
    throw Error('Invalid active take ID');
  return timeline as unknown as WorkspaceTimeline;
}
export function loadWorkspace(content: string) {
  const data = record(JSON.parse(content), 'workspace');
  if (data.format !== 'mojive-electron' || data.version !== 1)
    throw Error('Unsupported Mojive project format');
  finiteJson(data, 'workspace');
  validateDocument(data.document);
  const document = data.document as ReturnType<typeof encodeDocument>,
    duration = number(data.duration, 'timeline duration');
  if (duration <= 0) throw Error('Invalid timeline duration');
  const keys = array(data.keys, 'keyframes'),
    ids = new Set<string>();
  let previous = -Infinity;
  for (const value of keys) {
    const key = record(value, 'keyframe'),
      id = text(key.id, 'keyframe ID');
    if (ids.has(id)) throw Error('Duplicate keyframe ID');
    ids.add(id);
    if (key.name !== undefined && (typeof key.name !== 'string' || !key.name.trim()))
      throw Error('Invalid keyframe name');
    const time = timelineTime(key.time, duration, 'keyframe time');
    if (time < previous) throw Error('Keyframes must be sorted by time');
    previous = time;
    validatePoseForDocument(key.pose, document);
  }
  const timeline = validateTimeline(data.timeline, document, duration);
  return { doc: decodeDocument(document), keys: keys as unknown as Keyframe[], duration, timeline };
}
