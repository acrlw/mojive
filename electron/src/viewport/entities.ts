import * as THREE from 'three';
import type { SceneDocument } from '../contracts';
import { isViewportPickable } from '../scene-selection';
import masters from './entity-masters.json';
// Direct production triangles from interaction/scene_icons.py. The marker is 24
// logical points, scaled with the UI, camera-facing, with a matching 13-point pick radius.
const vertex = `precision highp float;in vec3 position;uniform mat4 projectionMatrix;uniform mat4 viewMatrix;uniform vec3 u_anchor;uniform vec2 u_size;uniform float u_scale;void main(){vec4 clip=projectionMatrix*viewMatrix*vec4(u_anchor,1.);clip.xy+=position.xy*u_scale*vec2(2.,-2.)/u_size*clip.w;gl_Position=clip;}`;
const fragment = `precision highp float;uniform vec4 u_color;out vec4 o_color;void main(){o_color=u_color;}`;
export class SceneEntityMarkers {
  private scene = new THREE.Scene();
  private geometry: Record<'camera' | 'light', THREE.BufferGeometry>;
  private meshes = new Map<string, THREE.Mesh>();
  private pickRadius = 13;
  private anchors: { id: string; object: THREE.Object3D; x: number; y: number }[] = [];
  constructor() {
    this.geometry = Object.fromEntries(
      (['camera', 'light'] as const).map((kind) => {
        const xy = masters[kind].flat(2);
        const positions: number[] = [];
        for (let i = 0; i < xy.length; i += 2) positions.push(xy[i], xy[i + 1], 0);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        return [kind, geometry];
      }),
    ) as Record<'camera' | 'light', THREE.BufferGeometry>;
  }
  render(
    renderer: THREE.WebGLRenderer,
    camera: THREE.Camera,
    doc: SceneDocument,
    selected: string | null,
    width: number,
    height: number,
    styleScale = 1,
  ) {
    this.anchors = [];
    this.pickRadius = 13 * styleScale;
    const ids = new Set<string>();
    for (const node of doc.nodes) {
      if (node.kind !== 'camera' && node.kind !== 'light') continue;
      ids.add(node.id);
      let mesh = this.meshes.get(node.id);
      if (!mesh) {
        const material = new THREE.RawShaderMaterial({
          vertexShader: vertex,
          fragmentShader: fragment,
          glslVersion: THREE.GLSL3,
          uniforms: {
            u_anchor: { value: new THREE.Vector3() },
            u_size: { value: new THREE.Vector2() },
            u_scale: { value: 1 },
            u_color: { value: new THREE.Vector4() },
          },
          depthTest: false,
          depthWrite: false,
          transparent: true,
          side: THREE.DoubleSide,
        });
        mesh = new THREE.Mesh(this.geometry[node.kind], material);
        mesh.frustumCulled = false;
        this.meshes.set(node.id, mesh);
        this.scene.add(mesh);
      }
      const world = node.object.getWorldPosition(new THREE.Vector3()),
        view = world.clone().applyMatrix4(camera.matrixWorldInverse);
      let visible = node.visible;
      for (let o: THREE.Object3D | null = node.object; o; o = o.parent) visible &&= o.visible;
      mesh.visible = visible && view.z < 0;
      if (!mesh.visible) continue;
      const material = mesh.material as THREE.RawShaderMaterial;
      material.uniforms.u_anchor.value.copy(world);
      material.uniforms.u_size.value.set(width, height);
      material.uniforms.u_scale.value = styleScale;
      material.uniforms.u_color.value.fromArray(
        selected === node.id && isViewportPickable(node.object)
          ? masters.selected_color
          : masters.helper_color,
      );
      const p = world.project(camera);
      this.anchors.push({
        id: node.id,
        object: node.object,
        x: ((p.x + 1) * width) / 2,
        y: ((1 - p.y) * height) / 2,
      });
    }
    for (const [id, mesh] of this.meshes)
      if (!ids.has(id)) {
        this.scene.remove(mesh);
        (mesh.material as THREE.Material).dispose();
        this.meshes.delete(id);
      }
    const auto = renderer.autoClear;
    renderer.autoClear = false;
    renderer.render(this.scene, camera);
    renderer.autoClear = auto;
  }
  pick(point: [number, number]) {
    const radiusSquared = this.pickRadius * this.pickRadius;
    let found: string | null = null,
      nearest = Infinity;
    for (const anchor of this.anchors) {
      if (!isViewportPickable(anchor.object)) continue;
      const distance = (anchor.x - point[0]) ** 2 + (anchor.y - point[1]) ** 2;
      if (distance <= radiusSquared && distance < nearest) {
        nearest = distance;
        found = anchor.id;
      }
    }
    return found;
  }
  dispose() {
    this.geometry.camera.dispose();
    this.geometry.light.dispose();
    for (const mesh of this.meshes.values()) (mesh.material as THREE.Material).dispose();
    this.meshes.clear();
  }
}
