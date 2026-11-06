import * as THREE from 'three';
import type { Joint, SceneDocument, SceneNode, EntityKind } from '../contracts';
import type { PhysicsFrame, PhysicsSource, Transform } from './contracts';
import { createGeometry } from './geometry.ts';
import { applySurface, bindAppearance, createTexture } from './appearance.ts';
import { validateFrame } from './frame.ts';
import { disposeDocument } from '../scene-resources.ts';
import { checkModelLoad, yieldModelLoad, type ModelLoadOptions } from '../resources/loading.ts';

export const ENVIRONMENT_ID = 'physics:environment';

/** Adapts immutable source structure and authoritative frames to the existing renderer. */
export class SceneBinding {
  document: SceneDocument;
  readonly source: PhysicsSource;
  private bodies: THREE.Object3D[] = [];
  private cameras: THREE.Camera[] = [];
  private lights: (THREE.DirectionalLight | THREE.SpotLight | THREE.PointLight)[] = [];
  private jointBodies: THREE.Object3D[] = [];
  private deformableMeshes = new Map<string, THREE.Mesh>();
  private entities: { id: string; kind: EntityKind; parentId?: string; type: string; worldFrame: boolean }[] = [];
  private world = new THREE.Matrix4();
  private local = new THREE.Matrix4();
  private position = new THREE.Vector3();
  private rotation = new THREE.Quaternion();
  private unit = new THREE.Vector3(1, 1, 1);
  private axis = new THREE.Vector3();
  private motion = new THREE.Quaternion();
  private frame: PhysicsFrame;

  constructor(source: PhysicsSource, frame: PhysicsFrame, defer = false) {
    this.source = source;
    this.frame = frame;
    validateFrame(source, frame);
    const root = new THREE.Group();
    root.name = source.name;
    root.userData = { kind: 'scene', physicsSourceId: source.id };
    const doc: SceneDocument = { name: source.name, root, nodes: [], joints: [], files: [], warnings: source.warnings };
    this.document = doc;
    if (!defer) for (const _ of this.build()) { /* Synchronous construction for adapter tests and callers. */ }
  }

  static async create(source: PhysicsSource, frame: PhysicsFrame, options?: ModelLoadOptions): Promise<SceneBinding> {
    checkModelLoad(options);
    const binding = new SceneBinding(source, frame, true);
    let budgetStart = performance.now(), completed = 0;
    const total = (source.textures?.length ?? 0) + source.geometries.length;
    options?.onProgress?.({ stage: 'scene', completed, total });
    const steps = binding.build();
    try {
      await yieldModelLoad(options);
      for (const _ of steps) {
        completed++;
        checkModelLoad(options);
        if (performance.now() - budgetStart >= 8) {
          options?.onProgress?.({ stage: 'scene', completed, total });
          await yieldModelLoad(options); budgetStart = performance.now();
        }
      }
      checkModelLoad(options);
      options?.onProgress?.({ stage: 'scene', completed: total, total });
      return binding;
    } catch (error) {
      steps.return();
      disposeDocument(binding.document);
      throw error;
    }
  }

  private *build(): Generator<void, void, unknown> {
    const { source, frame, document: doc } = this;
    const root = doc.root;
    const objects = new Map<string, THREE.Object3D>();
    const add = (id: string, name: string, kind: EntityKind, object: THREE.Object3D, parentId?: string, spatialParent = true) => {
      object.name = name;
      object.userData = { ...object.userData, nodeId: id, kind, physicsOwned: true,
        physicsWorldFrame: !spatialParent,
        gizmoEnabled: source.capabilities.includes('jointPose') && ['joint', 'link', 'geom'].includes(kind) };
      const parent = spatialParent && parentId ? objects.get(parentId) : root;
      if (!parent) throw new Error(`Missing scene parent: ${parentId}`);
      parent.add(object);
      objects.set(id, object);
      const node: SceneNode = { id, name, kind, object, parentId, visible: true };
      doc.nodes.push(node);
      return node;
    };
    this.bodies = source.bodies.map((body) => {
      const object = new THREE.Group();
      add(body.id, body.name, body.parentId ? 'link' : 'world', object, body.parentId ?? undefined);
      return object;
    });
    this.jointBodies = source.joints.map((joint) => {
      const body = objects.get(joint.bodyId);
      if (!body) throw new Error(`Missing joint body: ${joint.bodyId}`);
      return body;
    });
    if (source.capabilities.includes('environment')) {
      const environment = add(ENVIRONMENT_ID, 'Environment', 'environment', new THREE.Group());
      environment.properties = { Backend: source.backend, Version: source.version };
    }
    const textures: THREE.Texture[] = [];
    try {
      for (const texture of source.textures ?? []) { textures.push(createTexture(texture)); yield; }
      for (const geom of source.geometries) {
        const material = new THREE.MeshStandardMaterial({
          color: new THREE.Color().setRGB(geom.color[0], geom.color[1], geom.color[2], THREE.SRGBColorSpace),
          opacity: geom.color[3], transparent: geom.color[3] < 1,
          roughness: geom.roughness, metalness: geom.metallic, side: geom.shape.type === 'plane' || geom.deformable ? THREE.DoubleSide : THREE.FrontSide,
        });
        let geometry: THREE.BufferGeometry;
        try { geometry = createGeometry(geom.shape); }
        catch (error) { material.dispose(); throw error; }
        const mesh = new THREE.Mesh(geometry, material);
        try { applySurface(material, geometry, geom, textures); }
        catch (error) { geometry.dispose(); material.dispose(); throw error; }
        if (geom.deformable) this.deformableMeshes.set(geom.id, mesh);
        mesh.position.fromArray(geom.position); mesh.quaternion.fromArray(geom.rotation);
        mesh.castShadow = geom.shape.type !== 'plane'; mesh.receiveShadow = true;
        mesh.userData = { baseColor: material.color.getHex(), baseOpacity: material.opacity,
          visualGroup: geom.group, collidable: geom.collidable, role: geom.role,
          defaultVisible: geom.defaultVisible ?? true,
          viewportPickable: geom.shape.type !== 'plane', fitExcluded: geom.shape.type === 'plane' };
        const node = add(geom.id, geom.name, 'geom', mesh, geom.bodyId);
        node.properties = { Shape: geom.shape.type, 'Visual group': geom.group };
        yield;
      }
      for (const sourceJoint of source.joints) {
        const object = new THREE.Group();
        // Joint frames are engine-derived, not an invented chain for multi-joint bodies.
        add(sourceJoint.id, sourceJoint.name, 'joint', object, sourceJoint.bodyId, false);
        const body = source.bodies.find((item) => item.id === sourceJoint.bodyId)!;
        const joint: Joint = { id: sourceJoint.id, name: sourceJoint.name, type: sourceJoint.type,
          axis: new THREE.Vector3(...sourceJoint.axis), lower: sourceJoint.limits?.[0] ?? -Infinity,
          upper: sourceJoint.limits?.[1] ?? Infinity, value: 0, object, origin: new THREE.Matrix4(),
          parent: source.bodies.find((item) => item.id === body.parentId)?.name ?? 'World', child: body.name,
          bodyId: body.id,
          ...(sourceJoint.width > 1 ? { position: new THREE.Vector3(), rotation: new THREE.Quaternion() } : {}),
          ...(sourceJoint.type === 'ball' && sourceJoint.limits ? { angularLimit: sourceJoint.limits[1] } : {}),
        };
        doc.joints.push(joint);
      }
      this.cameras = source.cameras.map((camera) => {
        const [width, height] = camera.resolution;
        const aspect = width / height, halfHeight = camera.fov / 2;
        const near = source.visual?.near ?? .01, far = source.visual?.far ?? 1000;
        const object = camera.orthographic
          ? new THREE.OrthographicCamera(-halfHeight * aspect, halfHeight * aspect, halfHeight, -halfHeight, near, far)
          : new THREE.PerspectiveCamera(camera.fov, aspect, near, far);
        object.userData.camera = { resolution: [...camera.resolution], targetDistance: 5,
          sensorWidth: 35, orthoHeight: camera.orthographic ? camera.fov : 4,
          fov: camera.orthographic ? 45 : camera.fov };
        add(camera.id, camera.name, 'camera', object, camera.bodyId, false);
        return object;
      });
      this.lights = source.lights.map((light) => {
        const color = new THREE.Color(...light.color);
        const object = light.type === 'directional' ? new THREE.DirectionalLight(color, Math.PI)
          : light.type === 'point' ? new THREE.PointLight(color, Math.PI, 0, 0)
            : new THREE.SpotLight(color, Math.PI, 0, THREE.MathUtils.degToRad(light.cutoff), 0.2, 0);
        object.castShadow = light.castShadow ?? true;
        object.visible = light.active ?? true;
        const extent = source.visual?.extent ?? 5;
        object.shadow.mapSize.set(2048, 2048);
        object.shadow.camera.near = Math.max(.001, extent * .001);
        object.shadow.camera.far = extent * 10;
        if (object instanceof THREE.DirectionalLight) {
          Object.assign(object.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent });
          object.shadow.camera.updateProjectionMatrix();
        }
        if ('target' in object) root.add(object.target);
        add(light.id, light.name, 'light', object, light.bodyId, false);
        return object;
      });
      this.entities = doc.nodes.map((node) => ({ id: node.id, kind: node.kind, parentId: node.parentId,
        type: node.object.type, worldFrame: node.object.userData.physicsWorldFrame }));
      bindAppearance(root, source);
      this.apply(frame);
    } finally { textures.forEach((texture) => texture.dispose()); }
  }

  bodyAt(entityId: string): (Transform & { id: string }) | null {
    const { source, frame } = this;
    const id = source.geometries.find((item) => item.id === entityId)?.bodyId
      ?? source.joints.find((item) => item.id === entityId)?.bodyId ?? entityId;
    const index = source.bodies.findIndex((body) => body.id === id && body.dynamic);
    if (index < 0) return null;
    const offset = index * 7, data = frame.bodyTransforms;
    return { id, position: [data[offset], data[offset + 1], data[offset + 2]],
      rotation: [data[offset + 3], data[offset + 4], data[offset + 5], data[offset + 6]] };
  }

  validateDocument(document: SceneDocument): void {
    if (document.root.userData.physicsSourceId !== this.source.id)
      throw new Error('The document belongs to a different physics source.');
    const nodes = new Map(document.nodes.map((node) => [node.id, node]));
    const expected = new Map(this.entities.map((entity) => [entity.id, entity]));
    for (const [id, reference] of expected) {
      const node = nodes.get(id);
      if (!node || node.kind !== reference.kind || node.parentId !== reference.parentId ||
        node.object.type !== reference.type || node.object.userData.physicsOwned !== true ||
        node.object.userData.physicsWorldFrame !== reference.worldFrame)
        throw new Error(`Physics entity does not match its source: ${id}`);
    }
    for (const node of document.nodes)
      if (node.object.userData.physicsOwned && !expected.has(node.id))
        throw new Error(`Unknown physics entity: ${node.id}`);
    if (document.joints.length !== this.source.joints.length ||
      document.joints.some((joint, i) => joint.id !== this.source.joints[i].id))
      throw new Error('Physics joint order does not match its source.');
    for (const [index, joint] of document.joints.entries()) {
      const definition = this.source.joints[index];
      if (joint.bodyId !== definition.bodyId || joint.type !== definition.type ||
        !joint.axis.equals(new THREE.Vector3(...definition.axis)) ||
        joint.lower !== (definition.limits?.[0] ?? -Infinity) ||
        joint.upper !== (definition.limits?.[1] ?? Infinity))
        throw new Error(`Physics joint definition does not match its source: ${joint.id}`);
    }
  }

  rebind(document: SceneDocument): void {
    this.validateDocument(document);
    const objects = new Map(document.nodes.map((node) => [node.id, node.object]));
    this.bodies = this.source.bodies.map((body) => objects.get(body.id)!);
    this.jointBodies = this.source.joints.map((joint) => objects.get(joint.bodyId)!);
    this.cameras = this.source.cameras.map((camera) => objects.get(camera.id)! as THREE.Camera);
    this.lights = this.source.lights.map((light) => objects.get(light.id)! as typeof this.lights[number]);
    this.deformableMeshes = new Map(this.source.geometries.filter((g) => g.deformable).map((g) => [g.id, objects.get(g.id)! as THREE.Mesh]));
    this.document = document;
    bindAppearance(document.root, this.source);
  }

  apply(frame: PhysicsFrame): void {
    const { source, document: doc } = this;
    validateFrame(source, frame);
    this.frame = frame;
    for (const update of frame.geometryUpdates ?? []) {
      const geometry = this.deformableMeshes.get(update.id)!.geometry;
      for (const [name, values] of [['position', update.positions], ['normal', update.normals]] as const) {
        const attribute = geometry.getAttribute(name) as THREE.BufferAttribute;
        attribute.array.set(values); attribute.needsUpdate = true;
      }
      geometry.setDrawRange(0, update.count);
      geometry.computeBoundingBox(); geometry.computeBoundingSphere();
    }
    doc.root.updateMatrixWorld(true);
    for (let i = 0; i < this.bodies.length; i++) {
      const object = this.bodies[i];
      this.position.fromArray(frame.bodyTransforms, i * 7);
      this.rotation.fromArray(frame.bodyTransforms, i * 7 + 3);
      this.world.compose(this.position, this.rotation, this.unit);
      this.local.copy(object.parent!.matrixWorld).invert().multiply(this.world);
      this.local.decompose(object.position, object.quaternion, object.scale);
      object.updateMatrix(); object.updateWorldMatrix(false, false);
    }
    for (let i = 0; i < source.joints.length; i++) {
      const definition = source.joints[i], joint = doc.joints[i], offset = definition.offset;
      this.position.fromArray(frame.jointFrames, i * 6);
      this.axis.fromArray(frame.jointFrames, i * 6 + 3).normalize();
      this.rotation.setFromUnitVectors(joint.axis, this.axis);
      if (definition.width === 1) {
        joint.value = frame.jointPositions[offset];
        if (definition.type === 'prismatic') {
          const body = this.jointBodies[i];
          body.getWorldPosition(this.position).addScaledVector(this.axis, -joint.value);
        }
        joint.origin.compose(this.position, this.rotation, this.unit);
        joint.object.position.copy(this.position);
        joint.object.quaternion.copy(this.rotation);
        if (definition.type === 'prismatic') joint.object.position.addScaledVector(this.axis, joint.value);
        else joint.object.quaternion.multiply(this.motion.setFromAxisAngle(joint.axis, joint.value));
      } else {
        const qOffset = offset + (definition.width === 7 ? 3 : 0);
        joint.rotation!.fromArray(frame.jointPositions, qOffset);
        if (definition.width === 7) {
          joint.position!.fromArray(frame.jointPositions, offset);
          joint.origin.identity();
          joint.object.position.copy(joint.position!);
          joint.object.quaternion.copy(joint.rotation!);
        } else {
          const body = this.jointBodies[i];
          body.getWorldQuaternion(this.rotation);
          joint.object.position.copy(this.position); joint.object.quaternion.copy(this.rotation);
          this.motion.copy(joint.rotation!).invert(); this.rotation.multiply(this.motion);
          joint.origin.compose(this.position, this.rotation, this.unit);
        }
      }
    }
    for (let i = 0; i < this.cameras.length; i++) {
      this.cameras[i].position.fromArray(frame.cameraTransforms, i * 7);
      this.cameras[i].quaternion.fromArray(frame.cameraTransforms, i * 7 + 3);
    }
    for (let i = 0; i < this.lights.length; i++) {
      const light = this.lights[i];
      light.position.fromArray(frame.lightFrames, i * 6);
      if ('target' in light)
        light.target.position.copy(light.position).add(this.axis.fromArray(frame.lightFrames, i * 6 + 3));
    }
    doc.root.updateMatrixWorld(true);
  }
}
