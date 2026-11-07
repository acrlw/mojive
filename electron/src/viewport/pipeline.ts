import { installSurfaceTexture } from '../scene-rendering';
import * as THREE from 'three';
import type { SceneDocument, SceneNode, GeometryMode } from '../contracts';
import { selectionGeometry } from '../scene-selection';
import { outlineFragment } from './shaders';
const vertex = `precision highp float; in vec3 position; out vec2 v_uv;void main(){v_uv=position.xy*.5+.5;gl_Position=vec4(position,1.);}`;
// Native comparison.frag blends display-encoded nearest-role surfaces once. Three
// renders linear HDR into intermediate targets; apply the same display transform
// used by its ordinary scene pass to each role before the native weighted sum.
const comparison = `precision highp float;in vec2 v_uv;uniform sampler2D u_visual;uniform sampler2D u_collision;out vec4 o_color;
${THREE.ShaderChunk.tonemapping_pars_fragment}
vec3 display(vec3 c){c=ACESFilmicToneMapping(c);return mix(12.92*c,1.055*pow(max(c,vec3(0.)),vec3(1./2.4))-.055,step(vec3(.0031308),c));}
void main(){vec4 v=texture(u_visual,v_uv);vec4 c=texture(u_collision,v_uv);vec4 value=vec4(display(v.rgb)*v.a*.65,v.a*.65)+vec4(display(c.rgb)*c.a*.35,c.a*.35);o_color=vec4(value.rgb/max(value.a,1e-5),min(value.a,1.));}`;
export class ViewportPipeline {
  private targets = [0, 1, 2].map(
    (index) =>
      new THREE.WebGLRenderTarget(1, 1, {
        type: index < 2 ? THREE.HalfFloatType : THREE.UnsignedByteType,
        samples: 4,
        minFilter: THREE.NearestFilter,
        magFilter: THREE.NearestFilter,
      }),
  );
  private screen = new THREE.Scene();
  private camera = new THREE.Camera();
  private plane = new THREE.PlaneGeometry(2, 2);
  private quad: THREE.Mesh;
  private depthMaterial = new THREE.MeshDepthMaterial({ colorWrite: false });
  private maskMaterial = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
  private renderUnlit(scene: THREE.Scene, camera: THREE.Camera) {
    // Depth and selection masks do not sample shadows. Rebuilding the shadow
    // atlas here repeats every caster and can overwrite the color pass's atlas.
    const shadow = this.renderer.shadowMap;
    const autoUpdate = shadow.autoUpdate;
    const needsUpdate = shadow.needsUpdate;
    shadow.autoUpdate = false;
    shadow.needsUpdate = false;
    try {
      this.renderer.render(scene, camera);
    } finally {
      shadow.autoUpdate = autoUpdate;
      shadow.needsUpdate = needsUpdate;
    }
  }
  private outline = new THREE.RawShaderMaterial({
    vertexShader: vertex,
    fragmentShader: outlineFragment,
    glslVersion: THREE.GLSL3,
    uniforms: {
      u_mask: { value: this.targets[2].texture },
      u_size: { value: new THREE.Vector2(1, 1) },
      u_xray_alpha: { value: 0 },
      u_color: { value: new THREE.Vector4(1, 0.63, 0.2, 1) },
    },
    depthTest: false,
    depthWrite: false,
    transparent: true,
  });
  private composite = new THREE.RawShaderMaterial({
    vertexShader: vertex,
    fragmentShader: comparison,
    glslVersion: THREE.GLSL3,
    uniforms: {
      toneMappingExposure: { value: 1.1 },
      u_visual: { value: this.targets[0].texture },
      u_collision: { value: this.targets[1].texture },
    },
    depthTest: false,
    depthWrite: false,
    transparent: true,
  });
  private document: SceneDocument | null = null;
  private selectedObjects = new Set<THREE.Mesh>();
  private selectionNodes = new Map<THREE.Object3D, SceneNode>();
  private comparisonMaterials = new Map<THREE.Material, THREE.Material>();
  private highlighted = new Map<THREE.Material, THREE.Material>();
  private collision = new Map<THREE.Material, THREE.Material>();
  private outlineBounds = new THREE.Vector4();
  private previousScissor = new THREE.Vector4();
  private outlineProjection = new THREE.Matrix4();
  private outlineCorner = new THREE.Vector4();

  private selectedScissor(camera: THREE.Camera, padding: number): THREE.Vector4 | null {
    const { width, height } = this.targets[2];
    let left = Infinity,
      bottom = Infinity,
      right = -Infinity,
      top = -Infinity;
    for (const mesh of this.selectedObjects) {
      if (!mesh.visible) continue;
      // These paths may deform vertices outside the static geometry bounds.
      if (
        mesh instanceof THREE.SkinnedMesh ||
        mesh instanceof THREE.InstancedMesh ||
        mesh.morphTargetInfluences?.length
      )
        return null;
      const box = mesh.geometry.boundingBox;
      if (!box) return null;
      this.outlineProjection
        .multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
        .multiply(mesh.matrixWorld);
      for (let i = 0; i < 8; i++) {
        const p = this.outlineCorner
          .set(
            i & 1 ? box.max.x : box.min.x,
            i & 2 ? box.max.y : box.min.y,
            i & 4 ? box.max.z : box.min.z,
            1,
          )
          .applyMatrix4(this.outlineProjection);
        // A box crossing the near/eye plane needs the full-frame reference path.
        if (!Number.isFinite(p.x + p.y + p.z + p.w) || p.w <= 1e-6 || p.z <= -p.w) return null;
        const x = ((p.x / p.w + 1) * width) / 2,
          y = ((p.y / p.w + 1) * height) / 2;
        left = Math.min(left, x);
        right = Math.max(right, x);
        bottom = Math.min(bottom, y);
        top = Math.max(top, y);
      }
    }
    if (!Number.isFinite(left + bottom + right + top)) return null;
    const x = Math.max(0, Math.min(width, Math.floor(left) - padding));
    const y = Math.max(0, Math.min(height, Math.floor(bottom) - padding));
    right = Math.max(0, Math.min(width, Math.ceil(right) + padding));
    top = Math.max(0, Math.min(height, Math.ceil(top) + padding));
    return this.outlineBounds.set(x, y, right - x, top - y);
  }
  constructor(private renderer: THREE.WebGLRenderer) {
    this.quad = new THREE.Mesh(this.plane, this.outline);
    this.screen.add(this.quad);
  }
  resize(w: number, h: number) {
    this.targets.forEach((t) => t.setSize(w, h));
    this.outline.uniforms.u_size.value.set(w, h);
  }
  role(o: THREE.Object3D): string | undefined {
    for (let p: THREE.Object3D | null = o; p; p = p.parent) {
      if (p.userData.role) return p.userData.role;
    }
    return undefined;
  }
  private syncAppearance(target: THREE.Material, source: THREE.Material, color = true) {
    target.opacity = source.opacity;
    if (
      target instanceof THREE.MeshStandardMaterial &&
      source instanceof THREE.MeshStandardMaterial
    ) {
      if (color) target.color.copy(source.color);
      target.emissive.copy(source.emissive);
      target.roughness = source.roughness;
      target.metalness = source.metalness;
      target.emissiveIntensity = source.emissiveIntensity;
    }
  }
  private appearance(m: THREE.Material, selected: boolean, collision: boolean, comparison = false) {
    let material = m;
    if (collision) {
      let cached = this.collision.get(m);
      if (!cached) {
        cached = m.clone();
        installSurfaceTexture(cached);
        if ('color' in cached)
          (cached as THREE.MeshStandardMaterial).color.setRGB(
            0.8,
            0.45,
            0.12,
            THREE.SRGBColorSpace,
          );
        cached.transparent = false;
        cached.opacity = 1;
        cached.depthWrite = true;
        this.collision.set(m, cached);
      }
      this.syncAppearance(cached, m, false);
      cached.opacity = 1;
      material = cached;
    }
    if (comparison) {
      let opaque = this.comparisonMaterials.get(material);
      if (!opaque) {
        opaque = material.clone();
        installSurfaceTexture(opaque);
        opaque.transparent = true;
        opaque.blending = THREE.NoBlending;
        opaque.depthWrite = true;
        this.comparisonMaterials.set(material, opaque);
      }
      this.syncAppearance(opaque, material);
      material = opaque;
    }
    if (!selected) return material;
    let cached = this.highlighted.get(material);
    if (!cached) {
      cached = material.clone();
      const source = material;
      cached.onBeforeCompile = (shader, renderer) => {
        source.onBeforeCompile(shader, renderer);
        shader.fragmentShader = shader.fragmentShader
          .replace(
            '#include <color_fragment>',
            '#include <color_fragment>\ndiffuseColor.rgb = mix(diffuseColor.rgb, vec3(1.0, 0.82, 0.45), 0.35);',
          )
          .replace(
            '#include <emissivemap_fragment>',
            '#include <emissivemap_fragment>\ntotalEmissiveRadiance += 0.16 * diffuseColor.rgb;',
          );
      };
      cached.customProgramCacheKey = () => source.customProgramCacheKey() + ':mojive-selected';
      this.highlighted.set(material, cached);
    }
    this.syncAppearance(cached, material);
    return cached;
  }
  render(
    scene: THREE.Scene,
    camera: THREE.Camera,
    doc: SceneDocument,
    selected: string | null,
    mode: GeometryMode,
    options: {
      renderHelpers?: () => void;
      geometryVisible?: boolean;
      selectionOutline?: boolean;
      selectionHighlight?: boolean;
      outlineWidth?: number;
    } = {},
  ) {
    if (this.document !== doc) {
      for (const map of [this.highlighted, this.collision, this.comparisonMaterials]) {
        map.forEach((m) => m.dispose());
        map.clear();
      }
      this.document = doc;
    }
    const outlineWidth = Math.max(1, Math.min(8, Math.round(options.outlineWidth ?? 3)));
    if (this.outline.defines.OUTLINE_RADIUS !== outlineWidth) {
      this.outline.defines.OUTLINE_RADIUS = outlineWidth;
      this.outline.needsUpdate = true;
    }
    this.composite.uniforms.toneMappingExposure.value = this.renderer.toneMappingExposure;
    const renderer = this.renderer,
      records: {
        o: THREE.Mesh;
        visible: boolean;
        material: THREE.Material | THREE.Material[];
        role: string | undefined;
        selected: boolean;
      }[] = [];
    const selectedObjects = selectionGeometry(
      doc,
      selected,
      this.selectedObjects,
      this.selectionNodes,
    );
    doc.root.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        records.push({
          o,
          visible: o.visible,
          material: o.material,
          role: this.role(o),
          selected: selectedObjects.has(o),
        });
      }
    });
    const groups: { o: THREE.Object3D; visible: boolean }[] = [];
    doc.root.traverse((o) => {
      if (o.userData.roleGroup === true) groups.push({ o, visible: o.visible });
    });
    const helpers = scene.children
      .filter((o) => o.userData.viewportHelper)
      .map((o) => ({ o, visible: o.visible }));
    const savedBackground = scene.background;
    const set = (role?: string, mask = false) => {
      for (const g of groups) {
        g.o.visible =
          mode === 'default'
            ? g.visible
            : role
              ? g.o.userData.role === role
              : mode === 'both' || g.o.userData.role === mode;
      }
      for (const helper of helpers) helper.o.visible = helper.visible && !mask;
      for (const r of records) {
        r.o.visible =
          r.visible &&
          (mode !== 'default' || r.o.userData.defaultVisible !== false) &&
          options.geometryVisible !== false &&
          (Array.isArray(r.material)
            ? r.material.some((m) => m.opacity > 0)
            : r.material.opacity > 0) &&
          (mask
            ? r.selected
            : role
              ? r.role !== (role === 'visual' ? 'collision' : 'visual')
              : mode === 'default' ||
                mode === 'both' ||
                r.role !== (mode === 'visual' ? 'collision' : 'visual'));
        if (mask)
          r.o.visible &&=
            mode === 'default' ||
            mode === 'both' ||
            r.role !== (mode === 'visual' ? 'collision' : 'visual');
        r.o.material = mask
          ? this.maskMaterial
          : Array.isArray(r.material)
            ? r.material.map((m) =>
                this.appearance(
                  m,
                  r.selected && options.selectionHighlight !== false,
                  mode !== 'default' && r.role === 'collision',
                  mode === 'both',
                ),
              )
            : this.appearance(
                r.material,
                r.selected && options.selectionHighlight !== false,
                mode !== 'default' && r.role === 'collision',
                mode === 'both',
              );
      }
    };
    renderer.autoClear = true;
    renderer.setRenderTarget(null);
    if (mode === 'both') {
      scene.background = null;
      renderer.setClearColor(0x000000, 0);
      for (let i = 0; i < 2; i++) {
        set(i === 0 ? 'visual' : 'collision');
        renderer.setRenderTarget(this.targets[i]);
        renderer.render(scene, camera);
      }
      scene.background = savedBackground;
      renderer.setRenderTarget(null);
      renderer.setClearColor(0x22272d, 1);
      renderer.clear();
      this.quad.material = this.composite;
      renderer.autoClear = false;
      renderer.render(this.screen, this.camera);
    } else {
      set();
      renderer.setClearColor(0x22272d, 1);
      renderer.render(scene, camera);
    }
    if (options.renderHelpers) {
      // Comparison composites color from two targets; restore the union depth for ghosted helpers.
      if (mode === 'both') {
        set();
        scene.background = null;
        const previous = scene.overrideMaterial;
        scene.overrideMaterial = this.depthMaterial;
        for (const helper of helpers) helper.o.visible = false;
        renderer.autoClear = false;
        this.renderUnlit(scene, camera);
        scene.overrideMaterial = previous;
      }
      options.renderHelpers();
    }
    if (selectedObjects.size && options.selectionOutline !== false) {
      set(undefined, true);
      scene.background = null;
      renderer.setClearColor(0x000000, 0);
      renderer.setRenderTarget(this.targets[2]);
      renderer.autoClear = true;
      this.renderUnlit(scene, camera);
      renderer.setRenderTarget(null);
      renderer.autoClear = false;
      this.quad.material = this.outline;
      // Match the native pass: keep the exact AA shader, but don't evaluate its
      // neighborhood samples over millions of pixels outside the selection.
      const scissor = this.selectedScissor(camera, outlineWidth + 2);
      const scissorTest = renderer.getScissorTest();
      renderer.getScissor(this.previousScissor);
      if (scissor) {
        const density = renderer.getPixelRatio();
        if (scissorTest) {
          const previous = this.previousScissor;
          // setRenderTarget(null) floors each scissor component, while
          // setScissor rounds. Intersect in physical pixels to preserve edges
          // at fractional density before converting to the public API's units.
          const x = Math.floor(previous.x * density);
          const y = Math.floor(previous.y * density);
          const right = Math.min(scissor.x + scissor.z, x + Math.floor(previous.z * density));
          const top = Math.min(scissor.y + scissor.w, y + Math.floor(previous.w * density));
          scissor.x = Math.max(scissor.x, x);
          scissor.y = Math.max(scissor.y, y);
          scissor.z = Math.max(0, right - scissor.x);
          scissor.w = Math.max(0, top - scissor.y);
        }
        renderer.setScissor(scissor.divideScalar(density));
        renderer.setScissorTest(true);
      }
      try {
        renderer.render(this.screen, this.camera);
      } finally {
        if (scissor) {
          renderer.setScissor(this.previousScissor);
          renderer.setScissorTest(scissorTest);
          renderer.setRenderTarget(null);
        }
      }
    }
    scene.background = savedBackground;
    for (const g of groups) g.o.visible = g.visible;
    for (const helper of helpers) helper.o.visible = helper.visible;
    for (const r of records) {
      r.o.visible = r.visible;
      r.o.material = r.material;
    }
    renderer.autoClear = true;
  }
  dispose() {
    this.selectedObjects.clear();
    this.selectionNodes.clear();
    this.targets.forEach((t) => t.dispose());
    this.plane.dispose();
    this.maskMaterial.dispose();
    this.depthMaterial.dispose();
    this.outline.dispose();
    this.composite.dispose();
    this.comparisonMaterials.forEach((m) => m.dispose());
    this.highlighted.forEach((m) => m.dispose());
    this.collision.forEach((m) => m.dispose());
  }
}
