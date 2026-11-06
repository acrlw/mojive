import * as THREE from 'three';
import { sceneCommands } from './scene-commands';
import { getSceneRendering } from './scene-rendering';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';
import type { EntityKind, Joint, ProjectFile, SceneDocument, SceneNode } from './contracts';

const direct = (element: Element, name: string) =>
  Array.from(element.children).filter((child) => child.tagName === name);
const child = (element: Element, name: string) => direct(element, name)[0];
function numbers(text: string | null | undefined, fallback: number[], label: string): number[] {
  if (!text?.trim()) return [...fallback];
  const values = text.trim().split(/\s+/).map(Number);
  if (values.length !== fallback.length || values.some((value) => !Number.isFinite(value)))
    throw new Error(`Invalid ${label}: ${text}`);
  return values;
}
function scalar(element: Element | undefined, attribute: string, fallback: number): number {
  const raw = element?.getAttribute(attribute);
  if (raw === null || raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) throw new Error(`Invalid ${attribute}: ${raw}`);
  return value;
}
function nameOf(element: Element): string {
  const name = element.getAttribute('name');
  if (!name) throw new Error(`A ${element.tagName} is missing its name.`);
  return name;
}
function origin(element: Element, object: THREE.Object3D): void {
  const value = child(element, 'origin');
  object.position.fromArray(numbers(value?.getAttribute('xyz'), [0, 0, 0], 'origin position'));
  const [r, p, y] = numbers(value?.getAttribute('rpy'), [0, 0, 0], 'origin rotation');
  // URDF fixed-axis roll/pitch/yaw composes Rz * Ry * Rx.
  object.rotation.set(r, p, y, 'ZYX');
  object.updateMatrix();
}
function material(
  element?: Element,
  definitions = new Map<string, Element>(),
): THREE.MeshStandardMaterial {
  const definition =
    element && child(element, 'color')
      ? element
      : definitions.get(element?.getAttribute('name') || '') || element;
  const rgba = numbers(
    definition && child(definition, 'color')?.getAttribute('rgba'),
    [0.68, 0.73, 0.76, 1],
    'material color',
  );
  if (rgba.some((value) => value < 0 || value > 1))
    throw new Error('URDF material color must be in the range 0–1.');
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace),
    opacity: rgba[3],
    transparent: rgba[3] < 1,
    roughness: 0.48,
    metalness: 0.15,
  });
}
export function resourceUrl(filename: string, resourceBase: string): string {
  if (/^package:\/\//i.test(filename)) {
    const packagePath = filename.slice('package://'.length);
    if (resourceBase.startsWith('mojive-resource:')) {
      const base = new URL(resourceBase);
      const token = base.pathname.split('/')[1];
      return `mojive-resource://project/${token}/@package/${packagePath}`;
    }
    // Browser projects provide an explicit package root as their base.
    return new URL(packagePath, resourceBase || window.location.href).href;
  }
  if (/^file:/i.test(filename))
    throw new Error(
      'Absolute file:// mesh paths are not supported. Use a project-relative or package:// path.',
    );
  return new URL(filename, resourceBase || window.location.href).href;
}
function zUpGeometry(source: THREE.BufferGeometry): THREE.BufferGeometry {
  source.rotateX(Math.PI / 2);
  // Parametric Three geometries serialize their original parameters, dropping baked
  // vertex transforms. A neutral buffer preserves positions/normals in Undo and files.
  const geometry = new THREE.BufferGeometry().copy(source);
  source.dispose();
  return geometry;
}
async function geometryObject(
  element: Element,
  mat: THREE.MeshStandardMaterial,
  resourceBase: string,
  manager: THREE.LoadingManager,
  overrideMaterial: boolean,
): Promise<THREE.Object3D> {
  const shape = element.firstElementChild;
  if (!shape) throw new Error('A URDF geometry element is empty.');
  let geometry: THREE.BufferGeometry;
  if (shape.tagName === 'box') {
    const size = numbers(shape.getAttribute('size'), [1, 1, 1], 'box size');
    if (size.some((v) => v <= 0)) throw new Error('Box dimensions must be positive.');
    geometry = new THREE.BoxGeometry(...(size as [number, number, number]));
  } else if (shape.tagName === 'sphere') {
    const radius = scalar(shape, 'radius', 0.5);
    if (radius <= 0) throw new Error('Sphere radius must be positive.');
    geometry = new THREE.SphereGeometry(radius, 40, 24);
  } else if (shape.tagName === 'cylinder' || shape.tagName === 'capsule') {
    const radius = scalar(shape, 'radius', 0.5),
      length = scalar(shape, 'length', 1);
    if (radius <= 0 || length <= 0) throw new Error('Cylinder dimensions must be positive.');
    geometry = zUpGeometry(
      shape.tagName === 'capsule'
        ? new THREE.CapsuleGeometry(radius, length, 12, 32)
        : new THREE.CylinderGeometry(radius, radius, length, 40),
    );
  } else if (shape.tagName === 'mesh') {
    const filename = shape.getAttribute('filename');
    if (!filename) throw new Error('A URDF mesh is missing its filename.');
    const url = resourceUrl(filename, resourceBase);
    const extension = filename.split(/[?#]/)[0].split('.').pop()?.toLowerCase();
    let loaded: THREE.Object3D;
    if (extension === 'stl')
      loaded = new THREE.Mesh(await new STLLoader(manager).loadAsync(url), mat);
    else if (extension === 'obj') loaded = await new OBJLoader(manager).loadAsync(url);
    else if (extension === 'dae') {
      loaded = (await new ColladaLoader(manager).loadAsync(url)).scene;
      // ColladaLoader changes Z-up assets to Y-up; URDF places mesh coordinates directly.
      // Remove only its generated scene-axis correction, preserving asset node transforms.
      loaded.quaternion.identity();
    } else throw new Error(`Unsupported mesh format in ${filename}. Use STL, OBJ or DAE.`);
    const scale = numbers(shape.getAttribute('scale'), [1, 1, 1], 'mesh scale');
    if (scale.some((v) => v === 0)) throw new Error('Mesh scale cannot be zero.');
    loaded.scale.multiply(new THREE.Vector3(...(scale as [number, number, number])));
    loaded.traverse((object) => {
      if (object instanceof THREE.Mesh && (overrideMaterial || extension !== 'dae')) {
        const old = Array.isArray(object.material) ? object.material : [object.material];
        old.filter((value) => value !== mat).forEach((value) => value.dispose());
        object.material = mat;
      }
    });
    if (extension === 'dae' && !overrideMaterial) mat.dispose();
    return loaded;
  } else throw new Error(`Unsupported URDF geometry: ${shape.tagName}`);
  return new THREE.Mesh(geometry, mat);
}

/** Parse atomically: a failed mesh or invalid hierarchy never replaces the current document. */
export async function loadUrdf(
  content: string,
  resourceBase = '',
  files: ProjectFile[] = [],
  resolveResource?: (url: string) => string,
): Promise<SceneDocument> {
  const xml = new DOMParser().parseFromString(content, 'application/xml');
  if (xml.querySelector('parsererror'))
    throw new Error(
      `Invalid URDF XML: ${xml.querySelector('parsererror')!.textContent?.slice(0, 180)}`,
    );
  const robot = xml.documentElement;
  if (robot.tagName !== 'robot') throw new Error('Expected a URDF <robot> document.');
  const robotName = nameOf(robot),
    root = new THREE.Group();
  root.name = robotName;
  root.userData = { resourceBase, kind: 'scene' };
  const robotObject = new THREE.Group();
  robotObject.name = robotName;
  robotObject.userData = { nodeId: 'robot', kind: 'robot' };
  root.add(robotObject);
  const doc: SceneDocument = {
    name: robotName,
    root,
    nodes: [{ id: 'robot', name: robotName, kind: 'robot', object: robotObject, visible: true }],
    joints: [],
    files,
    source: content,
    warnings: [],
  };
  const links = new Map<string, THREE.Group>();
  const nodeByName = new Map<string, SceneNode>();
  const materials = new Map(direct(robot, 'material').map((element) => [nameOf(element), element]));
  const manager = new THREE.LoadingManager();
  if (resolveResource) manager.setURLModifier(resolveResource);
  const failedResources = new Set<string>();
  manager.onError = (url) => failedResources.add(url);
  let activeResources = 0;
  let finishResources: (() => void) | undefined;
  const start = manager.itemStart.bind(manager),
    end = manager.itemEnd.bind(manager);
  manager.itemStart = (url) => {
    activeResources++;
    start(url);
  };
  manager.itemEnd = (url) => {
    end(url);
    if (--activeResources === 0) finishResources?.();
  };
  try {
    for (const element of direct(robot, 'link')) {
      const name = nameOf(element);
      if (links.has(name)) throw new Error(`Duplicate link: ${name}`);
      const group = new THREE.Group();
      const id = `link:${name}`;
      group.name = name;
      group.userData = { nodeId: id, kind: 'link' };
      links.set(name, group);
      robotObject.add(group);
      const node: SceneNode = {
        id,
        name,
        kind: 'link',
        object: group,
        parentId: 'robot',
        visible: true,
        properties: {
          Visuals: direct(element, 'visual').length,
          Collisions: direct(element, 'collision').length,
        },
      };
      doc.nodes.push(node);
      nodeByName.set(name, node);
      for (const role of ['visual', 'collision'] as const) {
        for (const [geometryIndex, visual] of direct(element, role).entries()) {
          const shape = child(visual, 'geometry');
          if (!shape) throw new Error(`${name}: ${role} is missing geometry.`);
          const materialElement = child(visual, 'material');
          const mat = material(materialElement, materials);
          const definition =
            materialElement && child(materialElement, 'texture')
              ? materialElement
              : materials.get(materialElement?.getAttribute('name') || '');
          const textureFilename =
            definition && child(definition, 'texture')?.getAttribute('filename');
          let mesh: THREE.Object3D;
          try {
            if (textureFilename) {
              mat.map = await new THREE.TextureLoader(manager).loadAsync(
                resourceUrl(textureFilename, resourceBase),
              );
              mat.map.colorSpace = THREE.SRGBColorSpace;
            }
            mesh = await geometryObject(
              shape,
              mat,
              resourceBase,
              manager,
              Boolean(materialElement),
            );
          } catch (error) {
            mat.map?.dispose();
            mat.dispose();
            throw error;
          }
          const placement = new THREE.Group();
          origin(visual, placement);
          const geometryId = `geom:${name}:${role}:${geometryIndex}`;
          const roleGroup = new THREE.Group();
          roleGroup.userData = { nodeId: geometryId, role, roleGroup: true };
          roleGroup.visible = role === 'visual';
          roleGroup.add(mesh);
          placement.add(roleGroup);
          placement.name = visual.getAttribute('name') || `${name}_${role}_${geometryIndex + 1}`;
          placement.userData = { nodeId: geometryId, kind: 'geom' };
          mesh.traverse((object) => {
            object.userData = { ...object.userData, nodeId: geometryId, role };
            if (object instanceof THREE.Mesh) {
              object.castShadow = role === 'visual';
              object.receiveShadow = role === 'visual';
              const sourceMaterial = Array.isArray(object.material)
                ? object.material[0]
                : object.material;
              object.userData.baseColor =
                'color' in sourceMaterial
                  ? (sourceMaterial.color as THREE.Color).getHex()
                  : mat.color.getHex();
              object.userData.baseOpacity = sourceMaterial.opacity;
            }
          });
          group.add(placement);
          const geometry = shape.firstElementChild!;
          doc.nodes.push({
            id: geometryId,
            name: placement.name,
            kind: 'geom',
            object: placement,
            parentId: id,
            visible: true,
            properties: {
              Role: role,
              Shape: geometry.tagName,
              ...Object.fromEntries(
                Array.from(geometry.attributes).map((attribute) => [
                  attribute.name,
                  attribute.value,
                ]),
              ),
            },
          });
        }
      }
    }
    if (!links.size) throw new Error('The URDF contains no links.');
    const jointNames = new Set<string>(),
      childLinks = new Set<string>();
    for (const element of direct(robot, 'joint')) {
      const name = nameOf(element),
        id = `joint:${name}`,
        type = element.getAttribute('type') || 'fixed';
      if (jointNames.has(name)) throw new Error(`Duplicate joint: ${name}`);
      jointNames.add(name);
      if (!['fixed', 'continuous', 'revolute', 'prismatic'].includes(type))
        throw new Error(`Joint ${name} has unsupported type ${type}.`);
      const parent = child(element, 'parent')?.getAttribute('link') || '',
        linked = child(element, 'child')?.getAttribute('link') || '';
      if (!links.has(parent) || !links.has(linked))
        throw new Error(`Joint ${name} references a missing parent or child link.`);
      if (childLinks.has(linked)) throw new Error(`Link ${linked} has more than one parent joint.`);
      if (parent === linked) throw new Error(`Joint ${name} cannot connect a link to itself.`);
      childLinks.add(linked);
      const object = new THREE.Group();
      object.name = name;
      object.userData = { nodeId: id, kind: 'joint' };
      origin(element, object);
      const axis = new THREE.Vector3(
        ...(numbers(child(element, 'axis')?.getAttribute('xyz'), [1, 0, 0], 'joint axis') as [
          number,
          number,
          number,
        ]),
      );
      if (axis.lengthSq() === 0) throw new Error(`Joint ${name} has a zero axis.`);
      axis.normalize();
      const limit = child(element, 'limit');
      const lower =
        type === 'continuous'
          ? -Math.PI
          : scalar(limit, 'lower', type === 'prismatic' ? 0 : -Math.PI);
      const upper =
        type === 'continuous'
          ? Math.PI
          : scalar(limit, 'upper', type === 'prismatic' ? 1 : Math.PI);
      if (lower > upper) throw new Error(`Joint ${name} has reversed limits.`);
      const mimic = child(element, 'mimic');
      const joint: Joint = {
        id,
        name,
        type,
        axis,
        lower,
        upper,
        value: 0,
        object,
        origin: object.matrix.clone(),
        parent,
        child: linked,
      };
      if (mimic)
        joint.mimic = {
          joint: mimic.getAttribute('joint') || '',
          multiplier: scalar(mimic, 'multiplier', 1),
          offset: scalar(mimic, 'offset', 0),
        };
      doc.joints.push(joint);
      doc.nodes.push({
        id,
        name,
        kind: 'joint',
        object,
        parentId: `link:${parent}`,
        visible: true,
        properties: {
          Type: type,
          Axis: axis.toArray().join(', '),
          Parent: parent,
          Child: linked,
          ...(limit?.hasAttribute('effort') ? { Effort: scalar(limit, 'effort', 0) } : {}),
          ...(limit?.hasAttribute('velocity') ? { Velocity: scalar(limit, 'velocity', 0) } : {}),
        },
      });
      nodeByName.get(linked)!.parentId = id;
    }
    const parents = new Map(doc.joints.map((joint) => [joint.child, joint.parent]));
    for (const name of links.keys()) {
      const seen = new Set<string>();
      let ancestor: string | undefined = name;
      while (ancestor) {
        if (seen.has(ancestor)) throw new Error(`URDF hierarchy contains a cycle at ${ancestor}.`);
        seen.add(ancestor);
        ancestor = parents.get(ancestor);
      }
    }
    for (const joint of doc.joints) {
      links.get(joint.parent)!.add(joint.object);
      joint.object.add(links.get(joint.child)!);
    }
    for (const joint of doc.joints) {
      if (joint.mimic && !jointNames.has(joint.mimic.joint))
        throw new Error(`Joint ${joint.name} mimics an unknown joint ${joint.mimic.joint}.`);
      const seen = new Set<string>();
      let dependency: Joint | undefined = joint;
      while (dependency?.mimic) {
        if (seen.has(dependency.id)) throw new Error(`Mimic joint cycle at ${dependency.name}.`);
        seen.add(dependency.id);
        dependency = doc.joints.find((item) => item.name === dependency!.mimic!.joint);
      }
    }
    for (const joint of doc.joints.filter((item) => !item.mimic)) setJoint(doc, joint.id, 0);
    root.userData.actuators = direct(robot, 'transmission').flatMap((transmission) =>
      direct(transmission, 'actuator').map((actuator) => {
        const joint = child(transmission, 'joint')?.getAttribute('name') || '';
        const mechanicalReduction = Number(
          child(actuator, 'mechanicalReduction')?.textContent || '1',
        );
        if (!Number.isFinite(mechanicalReduction))
          throw new Error('Invalid actuator mechanical reduction.');
        const result = {
          name: nameOf(actuator),
          joint,
          mechanicalReduction,
          type: child(transmission, 'type')?.textContent || 'Unknown',
          transmission: nameOf(transmission),
        };
        const node = doc.nodes.find((item) => item.id === `joint:${joint}`);
        if (node)
          Object.assign(node.properties!, {
            Actuator: result.name,
            Transmission: result.transmission,
            'Mechanical reduction': mechanicalReduction,
          });
        return result;
      }),
    );
    if (activeResources)
      await new Promise<void>((resolve) => {
        finishResources = resolve;
      });
    if (failedResources.size)
      throw new Error(`Could not load model resources: ${Array.from(failedResources).join(', ')}`);
    root.updateMatrixWorld(true);
    return doc;
  } catch (error) {
    disposeDocument(doc);
    throw error;
  }
}

function jointMotion(joint: Joint): THREE.Matrix4 {
  if (joint.type === 'ball' || joint.type === 'free')
    return new THREE.Matrix4().compose(
      joint.position ?? new THREE.Vector3(),
      joint.rotation ?? new THREE.Quaternion(),
      new THREE.Vector3(1, 1, 1),
    );
  return joint.type === 'prismatic'
    ? new THREE.Matrix4().makeTranslation(joint.axis.clone().multiplyScalar(joint.value))
    : new THREE.Matrix4().makeRotationAxis(joint.axis, joint.value);
}

/** Set local free/ball coordinates without changing the authored body frame. */
export function setJointConfiguration(
  doc: SceneDocument,
  id: string,
  configuration: { position?: number[]; rotation?: number[] },
): void {
  const joint = doc.joints.find((value) => value.id === id || value.name === id);
  if (!joint || !['ball', 'free'].includes(joint.type))
    throw new Error(`Joint ${id} is not a ball or free joint.`);
  const position = configuration.position ?? joint.position?.toArray() ?? [0, 0, 0];
  const rotation = configuration.rotation ?? joint.rotation?.toArray() ?? [0, 0, 0, 1];
  if (
    position.length !== 3 ||
    rotation.length !== 4 ||
    [...position, ...rotation].some((value) => !Number.isFinite(value))
  )
    throw new Error('Joint configuration must contain finite position and quaternion values.');
  if (joint.type === 'ball' && Math.hypot(...position) > 1e-12)
    throw new Error('A ball joint does not permit translation.');
  const q = new THREE.Quaternion().fromArray(rotation);
  if (q.lengthSq() < 1e-20) throw new Error('Joint rotation quaternion cannot be zero.');
  q.normalize();
  if (joint.angularLimit !== undefined) {
    if (q.w < 0) q.set(-q.x, -q.y, -q.z, -q.w);
    const angle = 2 * Math.acos(THREE.MathUtils.clamp(q.w, -1, 1));
    if (angle > joint.angularLimit) q.slerp(new THREE.Quaternion(), 1 - joint.angularLimit / angle);
  }
  const commands = sceneCommands(doc);
  if (commands) {
    commands.setJointPosition(joint.id, joint.type === 'free' ? [...position, ...q.toArray()] : q.toArray());
    return;
  }
  joint.position ??= new THREE.Vector3();
  joint.rotation ??= new THREE.Quaternion();
  joint.position.fromArray(position);
  joint.rotation.copy(q);
  new THREE.Matrix4()
    .multiplyMatrices(joint.origin, jointMotion(joint))
    .decompose(joint.object.position, joint.object.quaternion, joint.object.scale);
  joint.object.updateMatrix();
  doc.root.updateMatrixWorld(true);
}

/** Commit a generic object transform as the joint's authored frame, keeping its value. */
export function authorJointTransform(doc: SceneDocument, id: string): void {
  if (sceneCommands(doc)) throw new Error('Model structure is owned by the physics source.');
  const joint = doc.joints.find((item) => item.id === id);
  if (!joint) throw new Error(`Unknown joint: ${id}`);
  joint.object.updateMatrix();
  const motion = jointMotion(joint);
  joint.origin.copy(joint.object.matrix).multiply(motion.invert());
  doc.root.updateMatrixWorld(true);
}

/** Change the joint in its authored origin frame and update dependent mimic joints. */
export function setJoint(doc: SceneDocument, id: string, value: number): void {
  if (!Number.isFinite(value)) throw new Error('Joint position must be finite.');
  const joint = doc.joints.find((item) => item.id === id || item.name === id);
  if (!joint) throw new Error(`Unknown joint: ${id}`);
  if (joint.type === 'ball' || joint.type === 'free') {
    if (value !== 0)
      throw new Error('Use a position/quaternion configuration for a multi-DOF joint.');
    setJointConfiguration(doc, id, { position: [0, 0, 0], rotation: [0, 0, 0, 1] });
    return;
  }
  const commands = sceneCommands(doc);
  if (commands) {
    commands.setJointPosition(joint.id, [THREE.MathUtils.clamp(value, joint.lower, joint.upper)]);
    return;
  }
  const update = (current: Joint, next: number, path = new Set<string>()) => {
    if (path.has(current.id)) throw new Error(`Mimic joint cycle at ${current.name}.`);
    const children = doc.joints.filter((item) => item.mimic?.joint === current.name);
    const nextPath = new Set(path).add(current.id);
    // Validate the entire chain before writing any transforms.
    for (const dependent of children) validateMimic(dependent, nextPath);
    current.value =
      current.type === 'fixed'
        ? 0
        : current.type === 'continuous'
          ? next
          : THREE.MathUtils.clamp(next, current.lower, current.upper);
    const motion = jointMotion(current);
    new THREE.Matrix4()
      .multiplyMatrices(current.origin, motion)
      .decompose(current.object.position, current.object.quaternion, current.object.scale);
    current.object.updateMatrix();
    for (const dependent of children)
      update(
        dependent,
        current.value * dependent.mimic!.multiplier + dependent.mimic!.offset,
        nextPath,
      );
  };
  const validateMimic = (current: Joint, path: Set<string>) => {
    if (path.has(current.id)) throw new Error(`Mimic joint cycle at ${current.name}.`);
    const nextPath = new Set(path).add(current.id);
    doc.joints
      .filter((item) => item.mimic?.joint === current.name)
      .forEach((item) => validateMimic(item, nextPath));
  };
  update(joint, value);
  doc.root.updateMatrixWorld(true);
}

let nextPrimitive = 1;
export function createPrimitive(kind: EntityKind, name?: string): SceneNode {
  const id = `${kind}:${crypto.randomUUID()}`;
  const label = name || `${kind.charAt(0).toUpperCase()}${kind.slice(1)} ${nextPrimitive++}`;
  let object: THREE.Object3D;
  const mat = new THREE.MeshStandardMaterial({
    color: '#9CBF8D',
    roughness: 0.52,
    metalness: 0.12,
  });
  const properties: Record<string, string | number> = {};
  switch (kind) {
    case 'box':
      object = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.5, 0.5), mat);
      object.position.z = 0.25;
      break;
    case 'sphere':
      object = new THREE.Mesh(new THREE.SphereGeometry(0.3, 48, 32), mat);
      object.position.z = 0.3;
      break;
    case 'cylinder': {
      const geo = zUpGeometry(new THREE.CylinderGeometry(0.25, 0.25, 0.6, 48));
      object = new THREE.Mesh(geo, mat);
      object.position.z = 0.3;
      break;
    }
    case 'capsule': {
      const geo = zUpGeometry(new THREE.CapsuleGeometry(0.2, 0.45, 12, 32));
      object = new THREE.Mesh(geo, mat);
      object.position.z = 0.425;
      break;
    }
    case 'ground':
      object = new THREE.Mesh(
        new THREE.PlaneGeometry(20, 20),
        new THREE.MeshStandardMaterial({ color: '#40464e', roughness: 0.95, metalness: 0 }),
      );
      properties.Width = 20;
      properties.Length = 20;
      break;
    case 'light':
      object = new THREE.PointLight('#fff5df', 14, 20, 2);
      object.position.set(2, -2, 3);
      properties.Type = 'Point';
      break;
    case 'camera':
      object = new THREE.PerspectiveCamera(45, 16 / 9, 0.01, 1000);
      object.up.set(0, 0, 1);
      object.position.set(3, -3, 2);
      object.lookAt(0, 0, 0.8);
      properties.Projection = 'Perspective';
      break;
    default:
      throw new Error(`Cannot create primitive type ${kind}.`);
  }
  if (!(object instanceof THREE.Mesh) || object.material !== mat) mat.dispose();
  object.name = label;
  object.userData = {
    nodeId: id,
    kind,
    role: 'visual',
    baseColor: mat.color.getHex(),
    baseOpacity: 1,
  };
  if (kind === 'ground') object.userData.viewportPickable = false;
  object.castShadow = kind !== 'ground';
  object.receiveShadow = true;
  return { id, name: label, kind, object, visible: true, properties };
}
/** Delete at a joint boundary and include dependent mimic branches atomically. */
export function removeNode(doc: SceneDocument, id: string): string[] {
  const nodes = new Map(doc.nodes.map((node) => [node.id, node]));
  const selected = nodes.get(id);
  if (!selected) throw new Error(`Unknown object: ${id}`);
  const links = new Map(
    doc.nodes
      .filter((node) => ['link', 'world', 'robot'].includes(node.kind))
      .map((node) => [node.name, node]),
  );
  const joints = new Map(doc.joints.map((joint) => [joint.name, joint]));
  for (const joint of doc.joints) {
    if (!nodes.has(joint.id) || !links.has(joint.parent) || !links.has(joint.child))
      throw new Error(`Joint ${joint.name} has a missing link or object.`);
    if (joint.mimic && !joints.has(joint.mimic.joint))
      throw new Error(`Joint ${joint.name} has a missing mimic driver.`);
  }
  const incoming =
    selected.kind === 'link' && selected.parentId ? nodes.get(selected.parentId) : undefined;
  const boundary = incoming?.kind === 'joint' ? incoming : selected;
  const removed = new Set<string>([boundary.id]);
  let changed = true;
  while (changed) {
    changed = false;
    const add = (value: string) => {
      if (!removed.has(value)) {
        removed.add(value);
        changed = true;
      }
    };
    for (const node of doc.nodes) if (node.parentId && removed.has(node.parentId)) add(node.id);
    for (const joint of doc.joints) {
      if (
        removed.has(links.get(joint.parent)!.id) ||
        removed.has(links.get(joint.child)!.id) ||
        (joint.mimic && removed.has(joints.get(joint.mimic.joint)!.id))
      )
        add(joint.id);
    }
  }
  if (doc.nodes.some((node) => removed.has(node.id) && node.object.userData.physicsOwned))
    throw new Error('The simulation backend owns this model structure.');
  const detached = doc.nodes
    .filter((node) => removed.has(node.id) && (!node.parentId || !removed.has(node.parentId)))
    .map((node) => node.object);
  const nextNodes = doc.nodes.filter((node) => !removed.has(node.id)),
    nextJoints = doc.joints.filter((joint) => !removed.has(joint.id));
  const nextJointNames = new Set(nextJoints.map((joint) => joint.name));
  const actuators = doc.root.userData.actuators;
  if (actuators !== undefined && !Array.isArray(actuators))
    throw new Error('Invalid actuator metadata.');
  const nextActuators = actuators?.filter((actuator: { joint: string }) =>
    nextJointNames.has(actuator.joint),
  );
  for (const object of detached) object.removeFromParent();
  doc.nodes = nextNodes;
  doc.joints = nextJoints;
  if (nextActuators) doc.root.userData.actuators = nextActuators;
  // Dispose only detached resources: user-authored objects can share live resources.
  disposeResources(collectResources(detached), collectResources([doc.root]));
  doc.root.updateMatrixWorld(true);
  return [...removed];
}

function collectResources(roots: THREE.Object3D[]) {
  const geometries = new Set<THREE.BufferGeometry>(),
    materials = new Set<THREE.Material>(),
    textures = new Set<THREE.Texture>(),
    shadows = new Set<THREE.LightShadow>();
  for (const root of roots) {
    const skybox = getSceneRendering(root)?.skybox;
    if (skybox) textures.add(skybox);
    root.traverse((object) => {
      if (
        object instanceof THREE.Mesh ||
        object instanceof THREE.Line ||
        object instanceof THREE.Points
      ) {
        geometries.add(object.geometry);
        for (const material of Array.isArray(object.material) ? object.material : [object.material])
          materials.add(material);
      } else if (object instanceof THREE.Sprite) materials.add(object.material);
      if (object instanceof THREE.Mesh) {
        if (object.customDepthMaterial) materials.add(object.customDepthMaterial);
        if (object.customDistanceMaterial) materials.add(object.customDistanceMaterial);
      }
      if (
        object instanceof THREE.DirectionalLight ||
        object instanceof THREE.SpotLight ||
        object instanceof THREE.PointLight
      )
        shadows.add(object.shadow);
    });
  }
  materials.forEach((mat) => {
    for (const value of Object.values(mat)) if (value instanceof THREE.Texture) textures.add(value);
  });
  return { geometries, materials, textures, shadows };
}

function disposeResources(
  resources: ReturnType<typeof collectResources>,
  live?: ReturnType<typeof collectResources>,
) {
  resources.geometries.forEach((value) => {
    if (!live?.geometries.has(value)) value.dispose();
  });
  resources.materials.forEach((value) => {
    if (!live?.materials.has(value)) value.dispose();
  });
  resources.textures.forEach((value) => {
    if (!live?.textures.has(value)) value.dispose();
  });
  resources.shadows.forEach((value) => {
    if (!live?.shadows.has(value)) value.dispose();
  });
}

export function disposeDocument(doc: SceneDocument): void {
  // Shadow materials, shadow targets and scene backgrounds are document-owned too.
  disposeResources(collectResources([doc.root]));
}
export async function createDemo(): Promise<SceneDocument> {
  const url = new URL('demo/atelier_arm.urdf', document.baseURI);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load the demo (${response.status}).`);
  const doc = await loadUrdf(await response.text(), new URL('./', url).href, [
    { name: 'atelier_arm.urdf', path: url.href, kind: 'file' },
  ]);
  setJoint(doc, 'shoulder', -0.35);
  setJoint(doc, 'elbow', 1.05);
  setJoint(doc, 'wrist', 0.65);
  return doc;
}
