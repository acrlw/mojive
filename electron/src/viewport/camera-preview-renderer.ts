import { installSurfaceTexture } from '../scene-rendering';
import * as THREE from 'three';
import type { GeometryMode, SceneDocument } from '../contracts';
import { cameraFromView, type CameraViewState } from '../scene-camera';
import { ViewportPipeline } from './pipeline';
import { createViewportScene } from './scene-environment';

/** A peer owns render objects and GPU resources; only decoded image data is shared. */
export class CameraPreviewRenderer {
  readonly renderer: THREE.WebGLRenderer;
  private environment = createViewportScene();
  private pipeline: ViewportPipeline;
  private source: SceneDocument | null = null;
  private document: SceneDocument | null = null;
  private pairs: [THREE.Object3D, THREE.Object3D][] = [];
  private objects = new Map<THREE.Object3D, THREE.Object3D>();
  private structure = '';
  private geometries = new Map<
    THREE.BufferGeometry,
    { copy: THREE.BufferGeometry; version: string }
  >();
  private materials = new Map<THREE.Material, { copy: THREE.Material; version: number }>();
  private textures = new Map<THREE.Texture, { copy: THREE.Texture; version: number }>();
  private used = new Set<THREE.BufferGeometry | THREE.Material | THREE.Texture>();
  private images = new Map<HTMLImageElement, () => void>();
  constructor(
    canvas: HTMLCanvasElement,
    private invalidate: () => void = () => {},
  ) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: 'low-power',
    });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.pipeline = new ViewportPipeline(this.renderer);
    this.environment.grid.visible = false;
  }
  private releaseObjects() {
    if (this.document) {
      this.environment.scene.remove(this.document.root);
      this.document.root.traverse((object) => {
        if (object instanceof THREE.Light && object.shadow) object.shadow.dispose();
      });
    }
    this.document = null;
    this.pairs = [];
    this.objects.clear();
    for (const map of [this.geometries, this.materials, this.textures]) {
      for (const entry of map.values()) entry.copy.dispose();
      map.clear();
    }
    this.used.clear();
    for (const [image, loaded] of this.images) {
      image.removeEventListener('load', loaded);
      image.removeEventListener('error', loaded);
    }
    this.images.clear();
  }
  private texture(source: THREE.Texture) {
    this.used.add(source);
    for (const image of Array.isArray(source.image) ? source.image : [source.image]) {
      if (!(image instanceof HTMLImageElement) || image.complete || this.images.has(image))
        continue;
      // ObjectLoader restores saved images asynchronously. Their completion must wake an idle preview.
      const loaded = () => {
        image.removeEventListener('load', loaded);
        image.removeEventListener('error', loaded);
        this.images.delete(image);
        this.invalidate();
      };
      this.images.set(image, loaded);
      image.addEventListener('load', loaded);
      image.addEventListener('error', loaded);
    }
    let entry = this.textures.get(source);
    if (!entry) {
      entry = { copy: source.clone(), version: source.version };
      // Texture.copy shares Source, whose version belongs to GPU upload tracking.
      // A private Source keeps upload/disposal independent; decoded pixels remain shared.
      entry.copy.source = new THREE.Source(source.source.data);
      this.textures.set(source, entry);
    } else if (entry.version !== source.version) {
      const pixels = entry.copy.source;
      entry.copy.copy(source);
      pixels.data = source.source.data;
      pixels.needsUpdate = true;
      entry.copy.source = pixels;
      entry.version = source.version;
    }
    entry.copy.offset.copy(source.offset);
    entry.copy.repeat.copy(source.repeat);
    entry.copy.center.copy(source.center);
    entry.copy.rotation = source.rotation;
    entry.copy.matrix.copy(source.matrix);
    return entry.copy;
  }
  private material(source: THREE.Material) {
    this.used.add(source);
    let entry = this.materials.get(source);
    if (!entry) {
      entry = { copy: source.clone(), version: source.version };
      this.materials.set(source, entry);
    } else {
      entry.copy.copy(source);
      if (entry.version !== source.version) entry.copy.needsUpdate = true;
      entry.version = source.version;
    }
    const values = entry.copy as unknown as Record<string, unknown>;
    for (const [key, value] of Object.entries(source))
      if (value instanceof THREE.Texture) values[key] = this.texture(value);
    installSurfaceTexture(entry.copy);
    return entry.copy;
  }
  private geometry(source: THREE.BufferGeometry) {
    this.used.add(source);
    const version = [
      source.index?.version,
      ...Object.entries(source.attributes).map(
        ([name, attribute]) =>
          name + ':' + ('version' in attribute ? attribute.version : attribute.data.version),
      ),
    ].join(';');
    let entry = this.geometries.get(source);
    if (!entry || entry.version !== version) {
      entry?.copy.dispose();
      entry = { copy: source.clone(), version };
      this.geometries.set(source, entry);
    }
    return entry.copy;
  }
  private sync(doc: SceneDocument) {
    let structure = '';
    doc.root.traverse((object) => {
      structure += object.uuid + ':' + object.type + ':' + object.children.length + ';';
    });
    if (this.source !== doc || this.structure !== structure) {
      this.releaseObjects();
      this.source = doc;
      this.structure = structure;
      const root = doc.root.clone(true);
      const pair = (source: THREE.Object3D, copy: THREE.Object3D) => {
        this.pairs.push([source, copy]);
        this.objects.set(source, copy);
        source.children.forEach((child, index) => pair(child, copy.children[index]));
      };
      pair(doc.root, root);
      this.document = {
        ...doc,
        root,
        nodes: doc.nodes.map((node) => ({ ...node, object: this.objects.get(node.object)! })),
        joints: doc.joints.map((joint) => ({ ...joint, object: this.objects.get(joint.object)! })),
      };
      this.environment.scene.add(root);
      this.environment.configure(doc.root);
    }
    this.used.clear();
    for (const [source, copy] of this.pairs) {
      copy.position.copy(source.position);
      copy.quaternion.copy(source.quaternion);
      copy.scale.copy(source.scale);
      copy.matrix.copy(source.matrix);
      copy.matrixAutoUpdate = source.matrixAutoUpdate;
      copy.visible = source.visible;
      copy.castShadow = source.castShadow;
      copy.receiveShadow = source.receiveShadow;
      copy.userData = { ...source.userData };
      if (
        (source instanceof THREE.Mesh ||
          source instanceof THREE.Line ||
          source instanceof THREE.Points) &&
        (copy instanceof THREE.Mesh || copy instanceof THREE.Line || copy instanceof THREE.Points)
      ) {
        copy.geometry = this.geometry(source.geometry);
        copy.material = Array.isArray(source.material)
          ? source.material.map((material) => this.material(material))
          : this.material(source.material);
      }
      if (source instanceof THREE.Light && copy instanceof THREE.Light) {
        copy.color.copy(source.color);
        copy.intensity = source.intensity;
        if (source.shadow && copy.shadow) {
          copy.shadow.camera.copy(source.shadow.camera, false);
          copy.shadow.bias = source.shadow.bias;
          copy.shadow.normalBias = source.shadow.normalBias;
        }
      }
      if (source instanceof THREE.PointLight && copy instanceof THREE.PointLight) {
        copy.distance = source.distance;
        copy.decay = source.decay;
      }
      if (source instanceof THREE.SpotLight && copy instanceof THREE.SpotLight) {
        copy.distance = source.distance;
        copy.decay = source.decay;
        copy.angle = source.angle;
        copy.penumbra = source.penumbra;
        copy.target = this.objects.get(source.target) ?? copy.target;
        copy.map = source.map ? this.texture(source.map) : null;
      }
      if (source instanceof THREE.DirectionalLight && copy instanceof THREE.DirectionalLight)
        copy.target = this.objects.get(source.target) ?? copy.target;
      if (source instanceof THREE.RectAreaLight && copy instanceof THREE.RectAreaLight) {
        copy.width = source.width;
        copy.height = source.height;
      }
      if (source instanceof THREE.HemisphereLight && copy instanceof THREE.HemisphereLight)
        copy.groundColor.copy(source.groundColor);
    }
    this.document!.nodes.forEach((node, index) => {
      node.visible = doc.nodes[index].visible;
    });
    let removed = false;
    for (const map of [this.geometries, this.materials, this.textures]) {
      for (const [source, entry] of map)
        if (!this.used.has(source)) {
          entry.copy.dispose();
          // Each map has its own key type; iteration's union cannot express that correlation.
          (map as Map<unknown, unknown>).delete(source);
          removed = true;
        }
    }
    if (removed) this.document = { ...this.document! };
  }
  render(
    doc: SceneDocument,
    view: CameraViewState,
    geometry: GeometryMode,
    width: number,
    height: number,
  ) {
    this.sync(doc);
    const w = Math.max(1, Math.round(width)),
      h = Math.max(1, Math.round(height));
    if (this.renderer.domElement.width !== w || this.renderer.domElement.height !== h) {
      this.renderer.setSize(w, h, false);
      this.pipeline.resize(w, h);
    }
    this.environment.scene.updateMatrixWorld(true);
    this.pipeline.render(
      this.environment.scene,
      cameraFromView(view),
      this.document!,
      null,
      geometry,
      { selectionOutline: false, selectionHighlight: false },
    );
  }
  dispose() {
    this.releaseObjects();
    this.pipeline.dispose();
    this.environment.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
  }
}
