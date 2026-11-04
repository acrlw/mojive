import * as THREE from 'three';
import type { Joint, SceneDocument, SceneNode } from './contracts';
import { createDemo, createPrimitive, loadUrdf, setJoint, setJointConfiguration } from './scene';

export const SCENE_CATALOG = [
  {
    id: 'joint_gizmo',
    name: 'Joint gizmos',
    nameZh: '关节操作示例',
    description: 'Revolute, prismatic, ball, free and compound joints, with precision limits.',
    descriptionZh: '旋转、滑动、球形、自由与复合关节，以及精密限位。',
    source: 'assets/joint_gizmo.xml',
  },
  {
    id: 'atelier_arm',
    name: 'Atelier arm',
    nameZh: 'Atelier 机械臂',
    description: 'A six-axis URDF arm with a coupled gripper and drive metadata.',
    descriptionZh: '六轴 URDF 机械臂、联动夹爪与驱动器信息。',
    source: 'demo/atelier_arm.urdf',
  },
  {
    id: 'geometry_lab',
    name: 'Geometry comparison',
    nameZh: '几何显示对比',
    description: 'Visual meshes and collision proxies with overlapping surfaces.',
    descriptionZh: '可视几何、碰撞代理及重叠表面对比。',
    source: 'Programmatic URDF',
  },
  {
    id: 'primitive_gallery',
    name: 'Scene objects',
    nameZh: '场景对象',
    description: 'Primitives, lights and cameras in an editable studio scene.',
    descriptionZh: '可编辑的基本体、灯光与相机场景。',
    source: 'Programmatic scene',
  },
] as const;
export type CatalogSceneId = (typeof SCENE_CATALOG)[number]['id'];
const children = (element: Element, name: string) =>
  Array.from(element.children).filter((child) => child.tagName === name);
const values = (text: string | null, fallback: number[]) =>
  text ? text.trim().split(/\s+/).map(Number) : [...fallback];
const degrees = THREE.MathUtils.degToRad;
function buffer(geometry: THREE.BufferGeometry) {
  const copy = new THREE.BufferGeometry().copy(geometry);
  geometry.dispose();
  return copy;
}
function pose(element: Element, object: THREE.Object3D) {
  object.position.fromArray(values(element.getAttribute('pos'), [0, 0, 0]));
  if (element.hasAttribute('quat')) {
    const [w, x, y, z] = values(element.getAttribute('quat'), [1, 0, 0, 0]);
    object.quaternion.set(x, y, z, w).normalize();
  } else {
    const [x, y, z] = values(element.getAttribute('euler'), [0, 0, 0]).map(degrees);
    object.rotation.set(x, y, z, 'XYZ');
  }
  object.updateMatrix();
}
function emptyDocument(name: string) {
  const root = new THREE.Group();
  root.name = name;
  root.userData = { kind: 'scene', actuators: [] };
  const world = new THREE.Group();
  world.name = 'World';
  world.userData = { nodeId: 'world', kind: 'world' };
  root.add(world);
  const doc: SceneDocument = {
    name,
    root,
    nodes: [{ id: 'world', name: 'World', kind: 'world', object: world, visible: true }],
    joints: [],
    files: [],
    warnings: [],
  };
  return { doc, world };
}

/** Reproduce the maintained joint acceptance scene; this is a kinematic scene adapter. */
export function createNativeJointScene(xml: string): SceneDocument {
  const parsed = new DOMParser().parseFromString(xml, 'application/xml');
  if (parsed.querySelector('parsererror')) throw new Error('Invalid joint example XML.');
  const source = parsed.documentElement,
    worldbody = children(source, 'worldbody')[0];
  if (!worldbody) throw new Error('The joint example has no world body.');
  const { doc, world } = emptyDocument('Joint gizmos');
  doc.source = xml;
  doc.root.userData.catalogId = 'joint_gizmo';
  doc.root.userData.source = 'assets/joint_gizmo.xml';
  const model = new THREE.Group();
  model.name = 'joint_gizmo_acceptance';
  model.userData = { nodeId: 'robot', kind: 'robot' };
  world.add(model);
  doc.nodes.push({
    id: 'robot',
    name: model.name,
    kind: 'robot',
    object: model,
    parentId: 'world',
    visible: true,
  });
  const materials = new Map<string, THREE.MeshStandardMaterial>();
  const asset = children(source, 'asset')[0];
  const checker = document.createElement('canvas');
  checker.width = checker.height = 64;
  const context = checker.getContext('2d')!;
  context.fillStyle = '#262b33';
  context.fillRect(0, 0, 64, 64);
  context.fillStyle = '#3d454f';
  context.fillRect(0, 0, 32, 32);
  context.fillRect(32, 32, 32, 32);
  const gridTexture = new THREE.CanvasTexture(checker);
  gridTexture.colorSpace = THREE.SRGBColorSpace;
  gridTexture.wrapS = gridTexture.wrapT = THREE.RepeatWrapping;
  gridTexture.repeat.set(20, 20);
  gridTexture.magFilter = THREE.NearestFilter;
  for (const element of asset ? children(asset, 'material') : []) {
    const rgba = values(element.getAttribute('rgba'), [1, 1, 1, 1]);
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace),
      roughness: 0.5,
      metalness: 0.12,
      opacity: rgba[3],
      transparent: rgba[3] < 1,
    });
    mat.name = element.getAttribute('name')!;
    if (element.hasAttribute('texture')) {
      mat.map = gridTexture;
      mat.roughness = 0.95;
      mat.metalness = 0;
    }
    materials.set(mat.name, mat);
  }
  let geometryIndex = 0;
  const addGeometry = (element: Element, parent: THREE.Object3D, parentId: string) => {
    const type = element.getAttribute('type') || 'sphere',
      size = values(element.getAttribute('size'), [0.1]),
      fromto = element.hasAttribute('fromto')
        ? values(element.getAttribute('fromto'), [])
        : undefined;
    const a = fromto ? new THREE.Vector3().fromArray(fromto) : undefined,
      b = fromto ? new THREE.Vector3().fromArray(fromto, 3) : undefined;
    const length = a && b ? a.distanceTo(b) : (size[1] ?? size[0]) * 2;
    let geometry: THREE.BufferGeometry;
    if (type === 'sphere') geometry = new THREE.SphereGeometry(size[0], 32, 20);
    else if (type === 'box')
      geometry = new THREE.BoxGeometry(
        size[0] * 2,
        (size[1] ?? size[0]) * 2,
        (size[2] ?? size[0]) * 2,
      );
    else if (type === 'cylinder')
      geometry = buffer(
        new THREE.CylinderGeometry(size[0], size[0], length, 40).rotateX(Math.PI / 2),
      );
    else if (type === 'capsule')
      geometry = buffer(new THREE.CapsuleGeometry(size[0], length, 12, 32).rotateX(Math.PI / 2));
    else if (type === 'plane') geometry = new THREE.PlaneGeometry(20, 20);
    else throw new Error(`Unsupported joint example shape ${type}.`);
    const name = element.getAttribute('name') || `${parent.name}_${type}_${++geometryIndex}`;
    const id = `geom:${name}`,
      placement = new THREE.Group();
    placement.name = name;
    placement.userData = { nodeId: id, kind: 'geom', fitExcluded: type === 'plane' };
    if (type === 'plane') placement.userData.viewportPickable = false;
    pose(element, placement);
    if (a && b) {
      placement.position.addVectors(a, b).multiplyScalar(0.5);
      placement.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 0, 1),
        b.clone().sub(a).normalize(),
      );
    }
    const sourceMaterial =
      materials.get(element.getAttribute('material') || '') ??
      new THREE.MeshStandardMaterial({ color: '#a9afb7', roughness: 0.5 });
    const mat = sourceMaterial.clone(),
      mesh = new THREE.Mesh(geometry, mat);
    mesh.castShadow = type !== 'plane';
    mesh.receiveShadow = true;
    mesh.userData = {
      nodeId: id,
      role: 'visual',
      baseColor: mat.color.getHex(),
      baseOpacity: mat.opacity,
    };
    const role = new THREE.Group();
    role.userData = { nodeId: id, role: 'visual', roleGroup: true };
    role.add(mesh);
    placement.add(role);
    parent.add(placement);
    doc.nodes.push({
      id,
      name,
      kind: 'geom',
      object: placement,
      parentId,
      visible: true,
      properties: {
        Shape: type,
        Role: 'visual',
        Size: size.join(' '),
        Material: mat.name,
        ...(fromto ? { From: fromto.slice(0, 3).join(' '), To: fromto.slice(3).join(' ') } : {}),
      },
    });
  };
  const body = (element: Element, parent: THREE.Object3D, parentId: string, parentName: string) => {
    const name = element.getAttribute('name')!;
    if (!name) throw new Error('Joint example body is missing a name.');
    const id = `link:${name}`,
      object = new THREE.Group();
    object.name = name;
    object.userData = { nodeId: id, kind: 'link' };
    const definitions = Array.from(element.children).filter(
      (child) => child.tagName === 'joint' || child.tagName === 'freejoint',
    );
    let owner = parent,
      ownerId = parentId;
    if (definitions.length) {
      const authored = new THREE.Group();
      pose(element, authored);
      for (const [index, definition] of definitions.entries()) {
        const jointName = definition.getAttribute('name')!,
          jointId = `joint:${jointName}`,
          type =
            definition.tagName === 'freejoint'
              ? 'free'
              : ({ hinge: 'revolute', slide: 'prismatic' }[
                  definition.getAttribute('type') || 'hinge'
                ] ??
                definition.getAttribute('type') ??
                'revolute');
        const movement = new THREE.Group();
        movement.name = jointName;
        movement.userData = { nodeId: jointId, kind: 'joint' };
        if (index === 0) {
          movement.position.copy(authored.position);
          movement.quaternion.copy(authored.quaternion);
        }
        movement.updateMatrix();
        owner.add(movement);
        const range = values(
          definition.getAttribute('range'),
          type === 'prismatic' ? [-1, 1] : [-180, 180],
        );
        const axis = new THREE.Vector3()
          .fromArray(values(definition.getAttribute('axis'), [0, 0, 1]))
          .normalize();
        const joint: Joint = {
          id: jointId,
          name: jointName,
          type,
          axis,
          lower: type === 'prismatic' ? range[0] : degrees(range[0]),
          upper: type === 'prismatic' ? range[1] : degrees(range[1]),
          value: 0,
          object: movement,
          origin: movement.matrix.clone(),
          parent: parentName,
          child: name,
          bodyId: id,
        };
        if (type === 'ball' || type === 'free') {
          joint.position = new THREE.Vector3();
          joint.rotation = new THREE.Quaternion();
          if (type === 'ball' && definition.hasAttribute('range'))
            joint.angularLimit = degrees(range[1]);
        }
        doc.joints.push(joint);
        doc.nodes.push({
          id: jointId,
          name: jointName,
          kind: 'joint',
          object: movement,
          parentId: ownerId,
          visible: true,
          properties: {
            Type: type,
            Axis: axis.toArray().join(', '),
            Parent: parentName,
            Child: name,
          },
        });
        owner = movement;
        ownerId = jointId;
      }
    } else pose(element, object);
    owner.add(object);
    doc.nodes.push({ id, name, kind: 'link', object, parentId: ownerId, visible: true });
    for (const geom of children(element, 'geom')) addGeometry(geom, object, id);
    for (const nested of children(element, 'body')) body(nested, object, id, name);
  };
  for (const geometry of children(worldbody, 'geom')) addGeometry(geometry, world, 'world');
  for (const element of children(worldbody, 'body')) body(element, model, 'robot', model.name);
  for (const element of children(worldbody, 'light')) {
    const name = element.getAttribute('name') || 'Key light',
      id = `light:${name}`,
      light = new THREE.DirectionalLight('#fff6e8', 1.5);
    light.name = name;
    pose(element, light);
    light.target.position.fromArray(values(element.getAttribute('dir'), [0, 0, -1]));
    light.add(light.target);
    light.userData = { nodeId: id, kind: 'light' };
    light.castShadow = true;
    world.add(light);
    doc.nodes.push({
      id,
      name,
      kind: 'light',
      object: light,
      parentId: 'world',
      visible: true,
      properties: { Type: 'Directional' },
    });
  }
  for (const element of children(worldbody, 'camera')) {
    const name = element.getAttribute('name') || 'Camera',
      id = `camera:${name}`,
      camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.02, 30);
    camera.name = name;
    pose(element, camera);
    const axes = values(element.getAttribute('xyaxes'), [1, 0, 0, 0, 1, 0]),
      x = new THREE.Vector3().fromArray(axes).normalize(),
      y = new THREE.Vector3().fromArray(axes, 3).normalize();
    camera.quaternion.setFromRotationMatrix(
      new THREE.Matrix4().makeBasis(x, y, x.clone().cross(y).normalize()),
    );
    camera.userData = { nodeId: id, kind: 'camera' };
    world.add(camera);
    doc.nodes.push({
      id,
      name,
      kind: 'camera',
      object: camera,
      parentId: 'world',
      visible: true,
      properties: { Projection: 'Perspective', 'Field of view': 45 },
    });
  }
  materials.forEach((material) => material.dispose());
  for (const joint of doc.joints)
    if (joint.type === 'ball' || joint.type === 'free') setJointConfiguration(doc, joint.id, {});
    else setJoint(doc, joint.id, 0);
  doc.root.updateMatrixWorld(true);
  return doc;
}

const comparisonUrdf = `<robot name="Geometry comparison">
<material name="sage"><color rgba="0.61 0.75 0.55 1"/></material><material name="blue"><color rgba="0.35 0.59 0.85 1"/></material><material name="orange"><color rgba="0.91 0.57 0.25 1"/></material>
<link name="world"/>
<link name="rounded_body"><visual><origin xyz="0 0 .45"/><geometry><sphere radius=".4"/></geometry><material name="sage"/></visual><collision><origin xyz="0 0 .45"/><geometry><box size=".8 .8 .8"/></geometry></collision></link><joint name="rounded_mount" type="fixed"><parent link="world"/><child link="rounded_body"/><origin xyz="-1.1 0 0"/></joint>
<link name="column"><visual><origin xyz="0 0 .5"/><geometry><cylinder radius=".25" length="1"/></geometry><material name="blue"/></visual><collision><origin xyz="0 0 .5"/><geometry><box size=".55 .55 1"/></geometry></collision></link><joint name="column_mount" type="fixed"><parent link="world"/><child link="column"/></joint>
<link name="capsule_shell"><visual><origin xyz="0 0 .5" rpy="0 .5 0"/><geometry><capsule radius=".2" length=".6"/></geometry><material name="orange"/></visual><collision><origin xyz="0 0 .5" rpy="0 .5 0"/><geometry><cylinder radius=".25" length="1"/></geometry></collision></link><joint name="shell_mount" type="fixed"><parent link="world"/><child link="capsule_shell"/><origin xyz="1.1 0 0"/></joint>
</robot>`;
export async function createCatalogScene(id: string): Promise<SceneDocument> {
  let doc: SceneDocument;
  if (id === 'joint_gizmo') {
    const response = await fetch(new URL('demo/joint_gizmo.xml', document.baseURI));
    if (!response.ok) throw new Error(`Joint example failed to load (${response.status}).`);
    doc = createNativeJointScene(await response.text());
  } else if (id === 'atelier_arm') doc = await createDemo();
  else if (id === 'geometry_lab') doc = await loadUrdf(comparisonUrdf);
  else if (id === 'primitive_gallery') {
    const scene = emptyDocument('Scene objects');
    doc = scene.doc;
    for (const [index, kind] of (['box', 'sphere', 'cylinder', 'capsule'] as const).entries()) {
      const node = createPrimitive(kind);
      node.object.position.x = (index - 1.5) * 1.1;
      scene.world.add(node.object);
      node.parentId = 'world';
      doc.nodes.push(node);
    }
    for (const kind of ['ground', 'light', 'camera'] as const) {
      const node = createPrimitive(kind);
      if (kind === 'ground') node.object.userData.fitExcluded = true;
      scene.world.add(node.object);
      node.parentId = 'world';
      doc.nodes.push(node);
    }
  } else throw new Error(`Unknown example scene: ${id}`);
  doc.root.userData.catalogId = id;
  return doc;
}
