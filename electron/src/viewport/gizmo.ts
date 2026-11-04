import * as THREE from 'three';
import type { Frame, SceneDocument, Tool, Joint } from '../contracts';
import { setJoint, setJointConfiguration, authorJointTransform } from '../scene';
import { snapValue } from './interaction_math';
import { NativeRaster } from './viewcube';
import { fillSurface } from './surfaces';
import { DragLinkRaster } from './drag-link';
import { activeLabel, activePalette, jointMarker, readoutMarker } from './drag_feedback';
import { translationRuler, rotationGuide, rotationAxisGuide } from './drag_guides';
import type { PreciseGizmoTarget } from './precise-input';
import {
  JointDwell,
  projectJoint,
  jointHit,
  inRail,
  type JointProjection,
  type JointTick,
} from './joint_interaction';
import {
  type P,
  AXIS_COLORS,
  JOINT_COLOR,
  HOVER_COLOR,
  SIZE,
  axisAlpha,
  ringAlpha,
  project,
  worldScale,
  arrow,
  affineCorners,
  dimensionAxis,
  basisForAxis,
  dial,
  frontArc,
  inside,
  pathDistance,
} from './geometry';
export interface GizmoSettings {
  translate?: (text: string) => string;
  readoutStyle?: {
    font: string;
    size: number;
    surface: string;
    ink: string;
    controls?: {
      primary: string;
      bright: string;
      dim: string;
      field: string;
      hovered: string;
      active: string;
      disabled: string;
    };
  };
  snapEnabled?: boolean;
  translationSnap?: number;
  rotationSnap?: number;
  rotationTickScale?: number;
  activeJoint?: string | null;
}
interface Handle {
  key: string;
  kind: 'axis' | 'plane' | 'ring' | 'screen' | 'trackball';
  axis: number;
  direction: THREE.Vector3;
  path: P[];
  alpha: number;
  closed?: boolean;
}
interface Drag {
  trackballAngles: P;
  snapping: boolean;
  basis: THREE.Vector3[];
  startRadial: THREE.Vector3;
  dragOrigin: THREE.Vector3;
  handle: Handle;
  start: P;
  last: P;
  object: THREE.Object3D;
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  scale: THREE.Vector3;
  worldPosition: THREE.Vector3;
  worldQuaternion: THREE.Quaternion;
  point: THREE.Vector3 | null;
  angle: number;
  total: number;
  jointValue?: number;
  jointOriginValue?: number;
  configurationPosition?: number[];
  configurationRotation?: number[];
  lastVector: THREE.Vector3;
  ring: P[];
  tangent: P;
  linear: boolean;
  linearAxis: number;
  linearSign: number;
  linearOrigin: P;
  linearAngle: number;
  axisScreen: P;
  worldPerPoint: number;
  slideAxis: number;
  slideSign: number;
  slideOrigin: P;
  slideTravel: number;
  slideLast: P;
}
export class GizmoOverlay {
  private jointProjection: JointProjection | null = null;
  private hingeAxis: ReturnType<typeof rotationAxisGuide> = null;
  private dwell = new JointDwell();
  private precisionVisible = false;
  private precisionSnapping = false;
  private precisionHovered = false;
  private limitHover: JointTick | null = null;
  private dwellTimer: number | undefined;
  private jointEdit: {
    kind: 'limit' | 'rail';
    joint: Joint;
    original: number;
    limit?: JointTick['kind'];
  } | null = null;
  private dragLink: DragLinkRaster | null = null;
  private raster = new NativeRaster();
  private ctx: CanvasRenderingContext2D;
  private camera!: THREE.Camera;
  private doc!: SceneDocument;
  private selected: string | null = null;
  private tool: Tool = 'select';
  private frame: Frame = 'world';
  private width = 1;
  private height = 1;
  private origin = new THREE.Vector3();
  private axes = [
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, 0, 1),
  ];
  private scale = 1;
  private handles: Handle[] = [];
  private hover: string | null = null;
  private drag: Drag | null = null;
  private joint: Joint | undefined;
  private configurationJoint: Joint | undefined;
  private settings: GizmoSettings = {};
  private snapModifier = false;
  private get translationStep() {
    return this.settings.translationSnap ?? 0.1;
  }
  private get rotationStep() {
    return THREE.MathUtils.degToRad(this.settings.rotationSnap ?? 5);
  }
  configure(settings: GizmoSettings) {
    const changed =
      this.settings.snapEnabled !== settings.snapEnabled ||
      this.settings.translationSnap !== settings.translationSnap ||
      this.settings.rotationSnap !== settings.rotationSnap;
    if (this.settings.activeJoint !== settings.activeJoint && this.active) this.end(true);
    this.settings = settings;
    if (changed && this.active) this.move(this.currentPointer, false, this.snapModifier);
  }

  private object: THREE.Object3D | undefined;
  private currentPointer: P = [0, 0];
  constructor(
    private canvas: HTMLCanvasElement,
    private callbacks: {
      onBegin?: () => void;
      onChange: () => void;
      onCommit?: () => void;
      invalidate: () => void;
    },
  ) {
    this.ctx = canvas.getContext('2d')!;
  }
  get active() {
    return this.drag !== null || this.jointEdit !== null;
  }
  update(
    camera: THREE.Camera,
    doc: SceneDocument,
    selected: string | null,
    tool: Tool,
    frame: Frame,
    w: number,
    h: number,
  ) {
    if (this.doc !== doc || this.selected !== selected || this.tool !== tool) {
      if (this.active) this.end(true);
      this.dwell.reset();
      this.precisionVisible = false;
      this.precisionHovered = false;
      this.limitHover = null;
      this.hover = null;
    }
    this.camera = camera;
    this.doc = doc;
    this.selected = selected;
    this.tool = tool;
    this.frame = frame;
    this.width = w;
    this.height = h;
    let targetNode = doc.nodes.find((n) => n.id === selected);
    this.object = targetNode?.object;
    while (targetNode?.kind === 'geom' && targetNode.parentId)
      targetNode = doc.nodes.find((n) => n.id === targetNode!.parentId);
    const candidates = doc.joints.filter(
      (j) =>
        j.type !== 'fixed' &&
        !j.mimic &&
        (j.id === selected ||
          j.bodyId === targetNode?.id ||
          j.child === targetNode?.id ||
          j.child === targetNode?.name),
    );
    const target =
      candidates.find((j) => j.id === selected) ??
      candidates.find((j) => j.id === this.settings.activeJoint) ??
      (candidates.length === 1 ? candidates[0] : undefined);
    this.joint =
      tool === 'joint' && target && target.type !== 'ball' && target.type !== 'free'
        ? target
        : undefined;
    this.configurationJoint =
      target &&
      ((target.type === 'ball' && (tool === 'joint' || tool === 'rotation')) ||
        (target.type === 'free' && tool !== 'select' && tool !== 'scale'))
        ? target
        : undefined;
    if (this.configurationJoint) this.object = this.configurationJoint.object;
    this.rebuild();
    this.refreshJointHover(this.currentPointer);
    this.draw();
  }
  private viewDirection() {
    const blend = Number(
      this.camera.userData.projectionBlend ??
        (this.camera instanceof THREE.OrthographicCamera ? 1 : 0),
    );
    return this.origin
      .clone()
      .sub(this.camera.position)
      .normalize()
      .multiplyScalar(1 - blend)
      .addScaledVector(this.camera.getWorldDirection(new THREE.Vector3()), blend)
      .normalize();
  }
  private rebuild() {
    this.handles = [];
    this.jointProjection = null;
    this.hingeAxis = null;
    if (!this.object || this.tool === 'select') return;
    this.object.updateWorldMatrix(true, false);
    this.object.getWorldPosition(this.origin);
    const q =
      this.frame === 'body' || this.tool === 'scale'
        ? this.object.getWorldQuaternion(new THREE.Quaternion())
        : new THREE.Quaternion();
    this.axes = [
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 0, 1),
    ].map((v) => v.applyQuaternion(q));
    if (this.tool === 'joint' && !this.configurationJoint) {
      if (!this.joint) return;
      const world = new THREE.Matrix4().multiplyMatrices(
        this.joint.object.parent?.matrixWorld ?? new THREE.Matrix4(),
        this.joint.origin,
      );
      this.origin.setFromMatrixPosition(world);
      const jq = new THREE.Quaternion().setFromRotationMatrix(world);
      this.axes = [
        ...basisForAxis(this.joint.axis.clone().applyQuaternion(jq).normalize()),
        this.joint.axis.clone().applyQuaternion(jq).normalize(),
      ];
    }
    if (this.configurationJoint?.type === 'ball') {
      const anchor = new THREE.Matrix4().multiplyMatrices(
        this.configurationJoint.object.parent?.matrixWorld ?? new THREE.Matrix4(),
        this.configurationJoint.origin,
      );
      this.origin.setFromMatrixPosition(anchor);
    }
    if (this.drag && !this.joint) this.axes = this.drag.basis;
    this.scale = worldScale(this.currentWorldPosition(), this.camera, this.height);
    if (this.scale <= 0) return;
    if (this.joint && this.joint.type !== 'prismatic')
      this.hingeAxis = rotationAxisGuide(
        this.camera,
        this.origin,
        this.axes[2],
        this.width,
        this.height,
        { hinge: true },
      );
    if (this.joint)
      this.jointProjection = projectJoint(
        this.joint,
        this.origin,
        this.axes[2],
        this.scale,
        this.camera,
        this.width,
        this.height,
      );
    const view = this.viewDirection(),
      center = this.screen(this.origin),
      rotation =
        this.tool === 'rotation' ||
        (this.tool === 'joint' &&
          this.configurationJoint?.type !== 'free' &&
          this.joint?.type !== 'prismatic');
    const push = (handle: Handle) => {
      if (
        (handle.alpha > 0.001 || (this.hingeAxis && handle.key === 'r2')) &&
        (!this.drag || this.drag.handle.key === handle.key)
      )
        this.handles.push(handle);
    };
    if (rotation) {
      if (!this.joint) {
        push({
          key: 'trackball',
          kind: 'trackball',
          axis: -1,
          direction: view.clone().negate(),
          path: [center],
          alpha: 1,
        });
      }
      for (let a = 0; a < 3; a++) {
        if (this.joint && a !== 2) continue;
        const dir = this.axes[a],
          active = this.drag?.handle.key === `r${a}`,
          alpha =
            active && !this.joint && this.configurationJoint?.type !== 'ball'
              ? 1
              : ringAlpha(dir.dot(view));
        const full = !!this.joint || active;
        const path = full
          ? dial(
              this.origin,
              dir,
              basisForAxis(dir)[0],
              this.scale * 0.78,
              0,
              Math.PI * 2,
              this.camera,
              this.width,
              this.height,
            )
          : frontArc(this.origin, dir, this.scale * 0.78, this.camera, this.width, this.height);
        push({ key: `r${a}`, kind: 'ring', axis: a, direction: dir, path, alpha, closed: full });
      }
      if (!this.joint)
        push({
          key: 'screen-ring',
          kind: 'ring',
          axis: -1,
          direction: view.clone().negate(),
          path: Array.from(
            { length: 65 },
            (_, i) =>
              [
                center[0] + Math.cos((i / 64) * Math.PI * 2) * SIZE * 0.96,
                center[1] + Math.sin((i / 64) * Math.PI * 2) * SIZE * 0.96,
              ] as P,
          ),
          alpha: 1,
          closed: true,
        });
    } else {
      if (
        this.tool === 'position' ||
        (this.tool === 'joint' && this.configurationJoint?.type === 'free')
      ) {
        const planes = [0, 1, 2].sort(
          (a, b) =>
            this.axes[(b + 1) % 3]
              .clone()
              .add(this.axes[(b + 2) % 3])
              .dot(view) -
            this.axes[(a + 1) % 3]
              .clone()
              .add(this.axes[(a + 2) % 3])
              .dot(view),
        );
        for (const a of planes) {
          const dir = this.axes[a],
            alpha = this.drag?.handle.key === `p${a}` ? 1 : ringAlpha(dir.dot(view));
          const path = affineCorners(
            [
              [0.22, 0.22],
              [0.4, 0.22],
              [0.4, 0.4],
              [0.22, 0.4],
            ].map(([u, v]) =>
              this.screen(
                this.origin
                  .clone()
                  .addScaledVector(this.axes[(a + 1) % 3], u * this.scale)
                  .addScaledVector(this.axes[(a + 2) % 3], v * this.scale),
              ),
            ),
          );
          push({ key: `p${a}`, kind: 'plane', axis: a, direction: dir, path, alpha });
        }
      }
      const order = [0, 1, 2].sort((a, b) => this.axes[b].dot(view) - this.axes[a].dot(view));
      for (const a of order) {
        if (this.joint && a !== 2) continue;
        const dir = this.axes[a],
          alpha = this.drag?.handle.key === `a${a}` && !this.joint ? 1 : axisAlpha(dir.dot(view));
        const end = this.screen(this.origin.clone().addScaledVector(dir, this.scale));
        const dx = end[0] - center[0],
          dy = end[1] - center[1],
          len = Math.hypot(dx, dy),
          start: P =
            len <= SIZE * 0.11
              ? end
              : [center[0] + (dx / len) * SIZE * 0.11, center[1] + (dy / len) * SIZE * 0.11];
        push({
          key: `a${a}`,
          kind: 'axis',
          axis: a,
          direction: dir,
          path: this.tool === 'scale' ? dimensionAxis(start, end) : arrow(start, end),
          alpha,
        });
      }
      if (!this.joint)
        push({
          key: 'center',
          kind: 'screen',
          axis: -1,
          direction: view.clone().negate(),
          path: [center],
          alpha: 1,
        });
    }
  }
  private screen(v: THREE.Vector3): P {
    return project(v, this.camera, this.width, this.height);
  }
  private fill(path: P[], color: string, alpha = 1) {
    if (path.length < 3) return;
    this.ctx.globalAlpha = 1;
    this.raster.fill(path, color, 1, true, alpha);
    this.raster.flush(this.ctx);
  }
  private stroke(path: P[], color: string, width: number, alpha = 1, closed = false) {
    if (!path.length) return;
    if (closed) {
      this.ctx.globalAlpha = 1;
      this.raster.stroke(
        path.length > 1 &&
          Math.hypot(path[0][0] - path[path.length - 1][0], path[0][1] - path[path.length - 1][1]) <
            1e-5
          ? path.slice(0, -1)
          : path,
        color,
        width,
        alpha,
      );
      this.raster.flush(this.ctx);
      return;
    }
    this.ctx.globalAlpha = 1;
    this.raster.strokeOpen(path, color, width, alpha);
    this.raster.flush(this.ctx);
  }
  private circle(p: P, r: number, color: string, alpha = 1, segments = 24) {
    const points = Array.from(
      { length: segments },
      (_, i): P => [
        p[0] + r * Math.cos((i / segments) * Math.PI * 2),
        p[1] + r * Math.sin((i / segments) * Math.PI * 2),
      ],
    );
    this.ctx.globalAlpha = 1;
    this.raster.fill(points, color, 1, true, alpha);
    this.raster.flush(this.ctx);
  }
  private draw() {
    const c = this.ctx,
      dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (
      this.canvas.width !== Math.round(this.width * dpr) ||
      this.canvas.height !== Math.round(this.height * dpr)
    ) {
      this.canvas.width = Math.round(this.width * dpr);
      this.canvas.height = Math.round(this.height * dpr);
    }
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, this.width, this.height);
    this.raster.resize(this.width, this.height, dpr);
    if (this.drag && !this.drag.snapping && !this.joint && this.tool === 'position')
      this.drawTranslationLink();
    for (const h of this.jointProjection ? [] : this.handles) {
      const hot = h.key === this.hover || h.key === this.drag?.handle.key,
        active = h.key === this.drag?.handle.key,
        base = this.joint ? JOINT_COLOR : h.axis >= 0 ? AXIS_COLORS[h.axis] : '#ebebeb';
      if (h.kind === 'trackball') {
        this.circle(h.path[0], SIZE * 0.78, '#b3b3b3', active ? 0.22 : hot ? 0.16 : 0.035, 64);
        continue;
      }
      if (h.kind === 'screen') {
        this.circle(h.path[0], SIZE * 0.075 + 0.75, '#adb5c2');
        this.circle(h.path[0], SIZE * 0.075, hot ? HOVER_COLOR : base);
        continue;
      }
      if (h.kind === 'ring') {
        if (active) continue;
        const center = this.screen(this.origin),
          paintPath =
            h.axis < 0
              ? h.path.map(
                  ([x, y]): P => [
                    center[0] + (x - center[0]) * (1 - 0.5 / (SIZE * 0.96)),
                    center[1] + (y - center[1]) * (1 - 0.5 / (SIZE * 0.96)),
                  ],
                )
              : h.path;
        if (h.axis < 0) this.stroke(paintPath, '#adb5c2', 4, h.alpha, true);
        const color =
          hot && h.axis >= 0
            ? `rgb(${[0, 1, 2]
                .map((i) => {
                  const v = parseInt(base.slice(1 + i * 2, 3 + i * 2), 16);
                  return v + (255 - v) * (active ? 0.36 : 0.2);
                })
                .join(',')})`
            : hot
              ? HOVER_COLOR
              : base;
        this.stroke(
          paintPath,
          color,
          h.axis < 0 ? 2.5 : 3.5,
          h.alpha * (h.axis < 0 ? 1 : active ? 0.96 : hot ? 0.88 : 0.68),
          h.closed,
        );
        continue;
      }
      this.fill(
        h.path,
        hot ? HOVER_COLOR : base,
        h.alpha * (h.kind === 'plane' ? (active ? 0.68 : 0.42) : 1),
      );
    }
    const deferredRange =
      !!this.drag &&
      this.drag.snapping &&
      this.drag.handle.kind === 'ring' &&
      !!this.jointProjection;
    if (this.jointProjection && !deferredRange) this.drawJointProjection();
    if (!this.drag && this.joint && this.joint.type !== 'prismatic')
      this.drawAxisGuide(this.origin, this.axes[2], true);
    this.drawPrecision();
    if (this.drag) this.drawDrag();
    c.globalAlpha = 1;
  }
  private hingeAxisHit(p: P) {
    const polygon = this.hingeAxis?.hitPolygon;
    return !!polygon && (inside(p, polygon) || pathDistance(p, polygon, true) <= 4);
  }
  private refreshJointHover(p: P) {
    const projection = this.jointProjection,
      now = performance.now() / 1000,
      axisHit = this.hingeAxisHit(p),
      hit = projection ? jointHit(p, projection, axisHit) : { hit: axisHit, limit: null },
      rail = projection?.rail ?? null;
    this.precisionHovered = this.precisionVisible && inRail(p, rail);
    this.precisionVisible = this.dwell.update(
      projection?.id ?? null,
      !!rail,
      hit.hit,
      this.precisionHovered,
      this.jointEdit?.kind === 'rail',
      now,
    );
    this.limitHover = this.precisionHovered ? null : hit.limit;
    if (this.joint)
      this.hover = hit.hit || this.precisionHovered ? (projection?.slide ? 'a2' : 'r2') : null;
    if (rail && (hit.hit || this.precisionVisible) && this.dwellTimer === undefined)
      this.dwellTimer = window.setTimeout(() => {
        this.dwellTimer = undefined;
        this.callbacks.invalidate();
      }, 40);
    return hit.hit || this.precisionHovered;
  }
  private tickStroke(a: P, b: P, color: string, width: number, alpha = 1) {
    this.ctx.globalAlpha = 1;
    this.raster.strokeLineG3(a, b, color, width, alpha);
    this.raster.flush(this.ctx);
  }
  private drawJointProjection() {
    const p = this.jointProjection!;
    const hot = !!this.hover || !!this.drag,
      active = !!this.drag,
      color = active ? HOVER_COLOR : hot ? '#b0cca4' : JOINT_COLOR;
    if (p.allowed.length) {
      if (p.slide) {
        this.raster.stroke(p.allowed, color, 3.5, p.alpha, false);
        this.raster.flush(this.ctx);
      } else this.stroke(p.allowed, color, 3.5, p.alpha, p.full);
    }
    if (p.complement.length) {
      this.raster.stroke(p.complement, color, 3.5, p.alpha * (hot ? 0.26 : 0.12), false);
      this.raster.flush(this.ctx);
    }
    for (const arrow of p.arrows) {
      this.raster.fill(arrow, color, 1, false, p.alpha);
      this.raster.flush(this.ctx);
    }
    for (const tick of p.ticks) {
      const ink =
        tick.kind === 'lower' ? AXIS_COLORS[2] : tick.kind === 'upper' ? AXIS_COLORS[0] : color;
      this.tickStroke(tick.start, tick.end, ink, 3.5, p.alpha);
    }
    if (this.limitHover && !this.jointEdit && !this.drag) {
      const tick = this.limitHover,
        name = tick.kind === 'lower' ? 'MIN' : 'MAX',
        value = p.slide
          ? `${tick.value >= 0 ? '+' : ''}${tick.value.toFixed(3)} m`
          : `${tick.value >= 0 ? '+' : ''}${THREE.MathUtils.radToDeg(tick.value).toFixed(1)}°`;
      this.jointReadout(
        tick.end,
        `${name} ${value}`,
        tick.kind === 'lower' ? AXIS_COLORS[2] : AXIS_COLORS[0],
        tick.kind === 'lower',
      );
    }
  }
  private jointReadout(anchor: P, text: string, color: string, above = true) {
    const c = this.ctx;
    c.globalAlpha = 1;
    c.font = `${this.settings.readoutStyle?.size ?? 16}px ${this.settings.readoutStyle?.font ?? '"JetBrains Mono",monospace'}`;
    const width = c.measureText(text).width + 28,
      height = Math.max(28, (this.settings.readoutStyle?.size ?? 16) + 12),
      x = Math.max(4, Math.min(this.width - width - 4, anchor[0] - width * 0.5)),
      y = Math.max(4, Math.min(this.height - height - 4, anchor[1] + (above ? -height - 8 : 8)));
    c.fillStyle = this.settings.readoutStyle?.surface ?? '#252a2f';
    fillSurface(c, x, y, width, height, 4);
    this.circle(
      [x + 11, y + height * 0.5],
      3,
      readoutMarker(color, this.settings.readoutStyle?.surface ?? '#252a2f'),
    );
    c.font = `${this.settings.readoutStyle?.size ?? 16}px ${this.settings.readoutStyle?.font ?? '"JetBrains Mono",monospace'}`;
    c.fillStyle = this.settings.readoutStyle?.ink ?? '#e6e8eb';
    c.textAlign = 'left';
    c.textBaseline = 'middle';
    c.fillText(text, x + 20, y + height * 0.5);
    c.textBaseline = 'alphabetic';
  }
  private drawPrecision() {
    const rail = this.jointProjection?.rail;
    if (!this.precisionVisible || !rail || !this.joint) return;
    const c = this.ctx,
      [x0, y0, x1, y1] = rail.rect,
      pressed = this.jointEdit?.kind === 'rail',
      hot = this.precisionHovered,
      controls = this.settings.readoutStyle?.controls ?? {
        primary: '#9cbf8d',
        bright: '#b8d2ac',
        dim: '#67875a',
        field: '#2b2f34',
        hovered: '#363b41',
        active: '#40474e',
        disabled: '#8e98a5',
      },
      knob = pressed || hot ? controls.bright : controls.primary,
      fill = pressed ? controls.bright : hot ? controls.primary : controls.dim,
      track = pressed ? controls.active : hot ? controls.hovered : controls.field,
      position = rail.current[0],
      left: P = [rail.start[0] + 2, rail.start[1]],
      right: P = [rail.end[0] - 2, rail.end[1]];
    this.raster.stroke(
      [
        rail.source,
        [Math.max(x0, Math.min(x1, rail.source[0])), rail.source[1] >= (y0 + y1) * 0.5 ? y1 : y0],
      ],
      controls.disabled,
      1,
      0.6,
      false,
    );
    this.raster.flush(c);
    c.globalAlpha = 1;
    c.fillStyle = this.settings.readoutStyle?.surface ?? '#252a2f';
    fillSurface(c, x0, y0, x1 - x0, y1 - y0, 4);
    if (pressed) {
      const original = this.jointEdit!.original,
        reference =
          rail.start[0] +
          Math.max(
            0,
            Math.min(1, (original - rail.lower) / Math.max(rail.upper - rail.lower, 1e-12)),
          ) *
            (rail.end[0] - rail.start[0]);
      const a = Math.min(right[0], Math.max(left[0], Math.min(reference, position))),
        b = Math.max(left[0], Math.min(right[0], Math.max(reference, position)));
      for (const [start, end, color] of [
        [left[0], a, fill],
        [a, b, controls.dim],
        [b, right[0], track],
      ] as [number, number, string][]) {
        if (end > start)
          this.raster.strokeOpen(
            [
              [start, left[1]],
              [end, left[1]],
            ],
            color,
            4,
            1,
            start === left[0],
            end === right[0],
          );
      }
      this.raster.flush(c);
      this.tickStroke([reference, left[1] - 4], [reference, left[1] + 4], controls.dim, 2);
    } else {
      this.tickStroke(left, right, track, 4);
      if (position > left[0])
        this.tickStroke(left, [Math.min(position, right[0]), left[1]], fill, 4);
    }
    this.tickStroke(
      [rail.start[0], rail.start[1] - 8],
      [rail.start[0], rail.start[1] + 8],
      AXIS_COLORS[2],
      3,
    );
    this.tickStroke(
      [rail.end[0], rail.end[1] - 8],
      [rail.end[0], rail.end[1] + 8],
      AXIS_COLORS[0],
      3,
    );
    this.circle(rail.current, 6, knob);
    if (pressed) {
      const span = rail.upper - rail.lower,
        slide = this.joint.type === 'prismatic',
        shown = slide ? this.joint.value : THREE.MathUtils.radToDeg(this.joint.value),
        shownSpan = slide ? span : THREE.MathUtils.radToDeg(span),
        decimals = slide
          ? span < 0.01
            ? 5
            : span < 0.1
              ? 4
              : 3
          : shownSpan < 1
            ? 3
            : shownSpan < 10
              ? 2
              : 1;
      this.jointReadout(
        [(x0 + x1) * 0.5, y0],
        `${this.joint.name} ${shown >= 0 ? '+' : ''}${shown.toFixed(decimals)}${slide ? ' m' : '°'}${this.precisionSnapping ? ` · ${this.settings.translate?.('SNAP') ?? 'SNAP'} ${slide ? this.translationStep : (this.settings.rotationSnap ?? 5)}${slide ? ' m' : '°'}` : ''}`,
        HOVER_COLOR,
      );
    }
  }
  private beginJointEdit(p: P) {
    if (!this.joint || !this.jointProjection) return false;
    this.refreshJointHover(p);
    if (this.limitHover) {
      this.jointEdit = {
        kind: 'limit',
        joint: this.joint,
        original: this.joint.value,
        limit: this.limitHover.kind,
      };
      this.callbacks.invalidate();
      return true;
    }
    if (this.precisionHovered && this.jointProjection.rail) {
      this.callbacks.onBegin?.();
      this.jointEdit = { kind: 'rail', joint: this.joint, original: this.joint.value };
      this.moveJointEdit(p, !!this.settings.snapEnabled || this.snapModifier);
      return true;
    }
    return false;
  }
  private moveJointEdit(p: P, snap: boolean) {
    const edit = this.jointEdit;
    if (!edit) return;
    this.precisionSnapping = snap;
    this.currentPointer = p;
    this.refreshJointHover(p);
    if (edit.kind === 'rail') {
      const rail = this.jointProjection?.rail;
      if (!rail) return;
      const t = Math.max(0, Math.min(1, (p[0] - rail.start[0]) / (rail.end[0] - rail.start[0])));
      let value = rail.lower + t * (rail.upper - rail.lower);
      if (snap) {
        const step = edit.joint.type === 'prismatic' ? this.translationStep : this.rotationStep;
        value = Math.max(rail.lower, Math.min(rail.upper, snapValue(value, step)));
      }
      if (Math.abs(value - edit.joint.value) > 1e-12) {
        setJoint(this.doc, edit.joint.id, value);
        this.callbacks.onChange();
        this.rebuild();
      }
    }
    this.draw();
    this.callbacks.invalidate();
  }
  private rebaseJointDrag(d: Drag, p: P) {
    d.jointValue = this.joint!.value;
    if (this.joint!.type === 'prismatic') d.worldPosition.copy(this.currentWorldPosition());
    d.start = p;
    d.angle = 0;
    d.total = 0;
    d.linearAngle = 0;
    d.linearOrigin = p;
    d.slideLast = p;
    d.slideOrigin = p;
    d.slideTravel = 0;
  }
  private endJointEdit(cancel: boolean) {
    const edit = this.jointEdit;
    if (!edit) return;
    this.jointEdit = null;
    if (edit.kind === 'rail') {
      if (cancel) {
        setJoint(this.doc, edit.joint.id, edit.original);
        this.callbacks.onChange();
      }
      this.callbacks.onCommit?.();
    } else if (
      !cancel &&
      this.limitHover &&
      this.limitHover.kind === edit.limit &&
      this.joint?.id === edit.joint.id
    ) {
      this.callbacks.onBegin?.();
      setJoint(this.doc, edit.joint.id, this.limitHover.value);
      this.callbacks.onChange();
      this.callbacks.onCommit?.();
    }
    this.rebuild();
    this.refreshJointHover(this.currentPointer);
    this.draw();
    this.callbacks.invalidate();
  }
  private currentWorldPosition() {
    return this.joint?.type === 'prismatic'
      ? this.origin.clone().addScaledVector(this.axes[2], this.joint.value)
      : this.origin.clone();
  }
  private drawTranslationLink() {
    const d = this.drag!;
    this.dragLink ??= new DragLinkRaster();
    this.dragLink.draw(
      this.ctx,
      this.screen(d.dragOrigin),
      this.screen(this.currentWorldPosition()),
      this.width,
      this.height,
      Math.min(window.devicePixelRatio || 1, 2),
    );
  }
  private drawAxisGuide(origin: THREE.Vector3, axis: THREE.Vector3, hinge: boolean) {
    const guide = hinge
      ? this.hingeAxis
      : rotationAxisGuide(this.camera, origin, axis, this.width, this.height, { hinge });
    if (!guide) return;
    const color =
      hinge && !this.drag
        ? this.hover
          ? '#b0cca4'
          : JOINT_COLOR
        : activePalette(this.drag?.handle.axis ?? 2, hinge).pressed;
    if (guide.segment) {
      this.raster.stroke(guide.segment, color, 1.6, guide.alpha, false);
    }
    for (const polygon of guide.polygons) this.raster.fill(polygon, color, 1, false, guide.alpha);
    this.raster.flush(this.ctx);
  }
  private drawActiveReadout(anchor: P, text: string, marker?: string) {
    const c = this.ctx;
    c.globalAlpha = 1;
    c.font = `${this.settings.readoutStyle?.size ?? 16}px ${this.settings.readoutStyle?.font ?? '"JetBrains Mono",monospace'}`;
    const prefix = marker ? 12 : 0,
      width = 16 + prefix + c.measureText(text).width,
      height = Math.max(28, (this.settings.readoutStyle?.size ?? 16) + 12),
      x = Math.max(4, Math.min(this.width - width - 4, anchor[0] + 16)),
      y = Math.max(4, Math.min(this.height - height - 4, anchor[1] + 16));
    c.fillStyle = this.settings.readoutStyle?.surface ?? '#252a2f';
    fillSurface(c, x, y, width, height, 4);
    if (marker)
      this.circle(
        [x + 11, y + height * 0.5],
        3,
        readoutMarker(marker, this.settings.readoutStyle?.surface ?? '#252a2f'),
      );
    c.font = `${this.settings.readoutStyle?.size ?? 16}px ${this.settings.readoutStyle?.font ?? '"JetBrains Mono",monospace'}`;
    c.fillStyle = this.settings.readoutStyle?.ink ?? '#e6e8eb';
    c.textAlign = 'left';
    c.textBaseline = 'middle';
    c.fillText(text, x + 8 + prefix, y + height * 0.5);
    c.textBaseline = 'alphabetic';
  }
  private drawDrag() {
    const d = this.drag!,
      h = d.handle,
      current = this.currentWorldPosition();
    let anchor = this.screen(current);
    if (h.kind === 'axis' && d.snapping && this.tool !== 'scale') {
      const ruler = translationRuler(
        this.camera,
        current,
        d.dragOrigin,
        h.direction,
        this.width,
        this.height,
        { slide: !!this.joint, step: this.translationStep },
      );
      if (ruler) {
        for (const segment of ruler.segments)
          this.raster.stroke(
            segment,
            this.joint ? HOVER_COLOR : AXIS_COLORS[h.axis],
            1.2,
            0.92,
            false,
          );
        for (const tick of ruler.ticks)
          this.raster.strokeLineG3(
            tick.start,
            tick.end,
            tick.active || this.joint ? HOVER_COLOR : AXIS_COLORS[h.axis],
            tick.width,
            tick.alpha,
            tick.roundStart,
            tick.roundEnd,
          );
        this.raster.flush(this.ctx);
      }
    }
    if (h.kind !== 'ring' && h.kind !== 'trackball' && this.tool !== 'scale') {
      if (d.snapping) this.drawTranslationLink();
      else if (this.joint && !this.jointProjection) {
        const start = this.screen(d.dragOrigin),
          end = this.screen(current),
          delta: P = [end[0] - start[0], end[1] - start[1]],
          distance = Math.hypot(...delta),
          tip = this.screen(current.clone().addScaledVector(h.direction, this.scale)),
          fallback: P = [tip[0] - end[0], tip[1] - end[1]],
          length = distance || Math.hypot(...fallback),
          normal: P =
            length > 1e-6
              ? [
                  -(distance ? delta[1] : fallback[1]) / length,
                  (distance ? delta[0] : fallback[0]) / length,
                ]
              : [0, 1];
        if (distance > 1e-6) {
          this.raster.stroke([start, end], '#67875a', 3.5, 1, false);
          this.raster.strokeLineG3(
            [start[0] - normal[0] * 6, start[1] - normal[1] * 6],
            [start[0] + normal[0] * 6, start[1] + normal[1] * 6],
            '#67875a',
            3.5,
          );
        }
        this.raster.strokeLineG3(
          [end[0] - normal[0] * 10, end[1] - normal[1] * 10],
          [end[0] + normal[0] * 10, end[1] + normal[1] * 10],
          HOVER_COLOR,
          3.5,
        );
        this.raster.flush(this.ctx);
      }
    }
    if (h.kind === 'ring') {
      if (h.axis >= 0) this.drawAxisGuide(d.worldPosition, h.direction, !!this.joint);
      const joint = this.joint
        ? {
            start: d.jointOriginValue ?? 0,
            current: this.joint.value,
            lower: this.joint.lower,
            upper: this.joint.upper,
          }
        : undefined;
      const guide = rotationGuide(
          this.camera,
          d.worldPosition,
          h.direction,
          d.startRadial,
          this.width,
          this.height,
          {
            angle: d.total,
            screen: h.axis < 0,
            joint,
            snapping: d.snapping,
            projectionFade: !!this.joint || this.configurationJoint?.type === 'ball',
            stepDegrees: this.settings.rotationSnap ?? 5,
            tickScale: this.settings.rotationTickScale ?? 1.25,
          },
        ),
        palette = activePalette(h.axis, !!this.joint);
      if (guide) {
        for (const tick of guide.ticks)
          this.raster.strokeLineG3(
            tick.start,
            tick.end,
            this.joint || tick.active ? palette.pressed : '#fafafc',
            tick.width,
            tick.alpha,
            tick.roundStart,
            tick.roundEnd,
          );
        this.raster.flush(this.ctx);
        if (this.jointProjection && d.snapping) this.drawJointProjection();
        if (guide.fillAlpha) this.raster.fan(guide.sector, palette.fill, guide.fillAlpha);
        if (guide.reference.length)
          this.raster.stroke(guide.reference, palette.pressed, 3.5, 0.96 * guide.alpha);
        if (guide.stroke.length)
          this.raster.ribbon(
            { vertices: guide.stroke, indices: guide.strokeIndices },
            palette.dark,
            guide.alpha,
          );
        this.raster.flush(this.ctx);
        anchor = guide.anchor;
      }
    }
    const text = activeLabel({
      kind: h.kind,
      axis: h.axis,
      angle: d.total,
      snapping: d.snapping,
      translationSnap: this.translationStep,
      rotationSnap: this.settings.rotationSnap ?? 5,
      translate: this.settings.translate,
      delta: current.clone().sub(d.dragOrigin),
      basis: d.basis,
      scale: this.tool === 'scale' ? { current: d.object.scale, original: d.scale } : undefined,
      joint: this.joint,
      trackball: d.trackballAngles,
    });
    this.drawActiveReadout(
      anchor,
      text,
      this.joint
        ? jointMarker(this.joint)
        : this.configurationJoint?.type === 'ball' && h.kind !== 'trackball'
          ? activePalette(h.axis).pressed
          : undefined,
    );
  }

  hoverAt(p: P) {
    this.currentPointer = p;
    if (this.joint) {
      const hit = this.refreshJointHover(p);
      this.draw();
      this.callbacks.invalidate();
      return hit;
    }
    const h = this.hit(p),
      changed = this.hover !== (h?.key ?? null);
    this.hover = h?.key ?? null;
    if (changed) {
      this.draw();
      this.callbacks.invalidate();
    }
    return !!h;
  }
  private hit(p: P) {
    if (this.jointProjection) {
      const hit = jointHit(p, this.jointProjection, this.hingeAxisHit(p));
      return hit.hit
        ? this.handles.find((h) => h.key === (this.jointProjection!.slide ? 'a2' : 'r2'))
        : undefined;
    }
    if (this.hingeAxisHit(p)) return this.handles.find((h) => h.key === 'r2');
    for (let i = this.handles.length - 1; i >= 0; i--) {
      const h = this.handles[i];
      if (h.alpha <= 0.05) continue;
      if (h.kind === 'screen' && Math.hypot(p[0] - h.path[0][0], p[1] - h.path[0][1]) <= 9)
        return h;
      if (h.kind === 'ring' && pathDistance(p, h.path, h.closed) <= 5.5) return h;
      if (
        (h.kind === 'plane' || h.kind === 'axis') &&
        (inside(p, h.path) || pathDistance(p, h.path, true) < (h.kind === 'axis' ? 4 : 2))
      )
        return h;
    }
    return this.handles.find(
      (h) =>
        h.kind === 'trackball' &&
        Math.hypot(p[0] - h.path[0][0], p[1] - h.path[0][1]) <= SIZE * 0.78,
    );
  }
  private rayPoint(p: P, normal: THREE.Vector3, origin: THREE.Vector3) {
    const ray = new THREE.Raycaster();
    ray.setFromCamera(
      new THREE.Vector2((p[0] / this.width) * 2 - 1, 1 - (p[1] / this.height) * 2),
      this.camera,
    );
    return ray.ray.intersectPlane(
      new THREE.Plane().setFromNormalAndCoplanarPoint(normal, origin),
      new THREE.Vector3(),
    );
  }
  begin(p: P) {
    this.currentPointer = p;
    if (this.beginJointEdit(p)) return true;
    const h = this.hit(p);
    if (!h || !this.object) return false;
    this.callbacks.onBegin?.();
    const normal =
      h.kind === 'axis'
        ? this.viewDirection()
            .addScaledVector(h.direction, -this.viewDirection().dot(h.direction))
            .normalize()
        : h.direction;
    this.drag = {
      trackballAngles: [0, 0],
      snapping: !!this.settings.snapEnabled || this.snapModifier,
      basis: this.axes.map((v) => v.clone()),
      startRadial: new THREE.Vector3(),
      dragOrigin: this.currentWorldPosition(),
      handle: h,
      start: p,
      last: p,
      object: this.object,
      position: this.object.position.clone(),
      quaternion: this.object.quaternion.clone(),
      scale: this.object.scale.clone(),
      worldPosition: this.currentWorldPosition(),
      worldQuaternion: this.object.getWorldQuaternion(new THREE.Quaternion()),
      point: this.rayPoint(p, normal, this.currentWorldPosition()),
      angle: 0,
      total: 0,
      jointValue: this.joint?.value,
      jointOriginValue: this.joint?.value,
      configurationPosition: this.configurationJoint?.position?.toArray(),
      configurationRotation: this.configurationJoint?.rotation?.toArray(),
      lastVector: new THREE.Vector3(),
      ring: [],
      tangent: [1, 0],
      linear: false,
      linearAxis: -1,
      linearSign: 1,
      linearOrigin: p,
      linearAngle: 0,
      axisScreen: [1, 0],
      worldPerPoint: 0,
      slideAxis: -1,
      slideSign: 1,
      slideOrigin: p,
      slideTravel: 0,
      slideLast: p,
    };
    if (h.kind === 'axis') {
      const d = this.drag,
        center = this.screen(d.worldPosition),
        tip = this.screen(d.worldPosition.clone().addScaledVector(h.direction, this.scale)),
        delta: P = [tip[0] - center[0], tip[1] - center[1]],
        length = Math.hypot(...delta);
      d.axisScreen = [delta[0] / length, delta[1] / length];
      d.worldPerPoint = this.scale / length;
    }
    if (h.kind === 'ring') {
      const d = this.drag,
        hit = this.rayPoint(p, h.direction, this.origin);
      const radial = hit?.sub(this.origin);
      // Native hinge axes remain usable at the center and edge-on plane singularity.
      const degenerate = !radial || radial.lengthSq() < 1e-18;
      d.lastVector.copy(
        this.joint && degenerate
          ? this.axes[0]
          : radial
            ? radial.normalize()
            : basisForAxis(h.direction)[0],
      );
      d.startRadial.copy(d.lastVector);
      d.ring = dial(
        this.origin,
        h.direction,
        d.lastVector,
        this.scale * (h.axis < 0 ? 0.96 : 0.78),
        0,
        Math.PI * 2,
        this.camera,
        this.width,
        this.height,
      );
      const a = d.ring[1],
        b = d.ring[d.ring.length - 2],
        len = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (len > 1e-6) d.tangent = [(a[0] - b[0]) / len, (a[1] - b[1]) / len];
      d.linear =
        (!!this.joint && degenerate) ||
        (h.axis >= 0 && ringAlpha(h.direction.dot(this.viewDirection())) < 0.18);
    }
    this.callbacks.invalidate();
    return true;
  }
  move(p: P, shift = false, snap = false) {
    this.snapModifier = snap;
    snap ||= !!this.settings.snapEnabled;
    this.currentPointer = p;
    if (this.jointEdit) {
      this.moveJointEdit(p, snap);
      return;
    }
    const d = this.drag;
    if (!d) return;
    d.snapping = snap;
    const h = d.handle,
      precision = shift ? 0.1 : 1;
    let scalar = 0,
      rawScalar = 0;
    if (h.kind === 'ring') {
      if (!d.linear && pathDistance(p, d.ring, true) > 8) d.linear = true;
      if (d.linear) {
        if (d.linearAxis < 0 && Math.hypot(p[0] - d.start[0], p[1] - d.start[1]) >= 2) {
          const total: P = [p[0] - d.start[0], p[1] - d.start[1]];
          d.linearAxis =
            Math.abs(total[0]) === Math.abs(total[1])
              ? Math.abs(d.tangent[0]) > Math.abs(d.tangent[1])
                ? 0
                : 1
              : Math.abs(total[0]) > Math.abs(total[1])
                ? 0
                : 1;
          const component = d.tangent[d.linearAxis];
          d.linearSign =
            Math.abs(component) >= 0.15 ? Math.sign(component) : d.linearAxis === 0 ? 1 : -1;
          d.linearOrigin = p;
          d.angle += (p[d.linearAxis] - d.last[d.linearAxis]) * d.linearSign * 0.01 * precision;
          d.linearAngle = d.angle;
        }
        if (d.linearAxis >= 0)
          d.angle =
            d.linearAngle +
            (p[d.linearAxis] - d.linearOrigin[d.linearAxis]) * d.linearSign * 0.01 * precision;
      } else {
        const hit = this.rayPoint(p, h.direction, d.worldPosition);
        if (hit) {
          const v = hit.sub(d.worldPosition).normalize(),
            delta = Math.atan2(
              h.direction.dot(new THREE.Vector3().crossVectors(d.lastVector, v)),
              d.lastVector.dot(v),
            );
          d.angle += delta * precision;
          d.lastVector.copy(v);
        }
      }
      d.total = snap ? snapValue(d.angle, this.rotationStep) : d.angle;
      if (this.joint) {
        const requested = (d.jointValue ?? 0) + d.total;
        setJoint(this.doc, this.joint.id, requested);
        if (
          Math.abs(this.joint.value - requested) > 1e-12 ||
          (this.joint.value <= this.joint.lower + 1e-12 &&
            (d.jointValue ?? 0) + d.angle < this.joint.lower - 1e-12) ||
          (this.joint.value >= this.joint.upper - 1e-12 &&
            (d.jointValue ?? 0) + d.angle > this.joint.upper + 1e-12)
        )
          this.rebaseJointDrag(d, p);
      } else {
        const world = new THREE.Quaternion()
          .setFromAxisAngle(h.direction, d.total)
          .multiply(d.worldQuaternion);
        const parent =
          d.object.parent?.getWorldQuaternion(new THREE.Quaternion()).invert() ??
          new THREE.Quaternion();
        d.object.quaternion.copy(parent.multiply(world));
      }
    } else if (h.kind === 'trackball') {
      const angles: P = [
        (p[1] - d.start[1]) * 0.01 * precision,
        (p[0] - d.start[0]) * 0.01 * precision,
      ];
      if (snap) {
        const step = this.rotationStep;
        angles[0] = snapValue(angles[0], step);
        angles[1] = snapValue(angles[1], step);
      }
      d.trackballAngles = angles;
      const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0),
        up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 1),
        rotvec = right.multiplyScalar(angles[0]).addScaledVector(up, angles[1]);
      d.total = rotvec.length();
      const q = new THREE.Quaternion()
          .setFromAxisAngle(rotvec.normalize(), d.total)
          .multiply(d.worldQuaternion),
        parent =
          d.object.parent?.getWorldQuaternion(new THREE.Quaternion()).invert() ??
          new THREE.Quaternion();
      d.object.quaternion.copy(parent.multiply(q));
    } else {
      const normal =
          h.kind === 'axis'
            ? this.viewDirection()
                .addScaledVector(h.direction, -this.viewDirection().dot(h.direction))
                .normalize()
            : h.direction,
        point = this.rayPoint(p, normal, d.worldPosition),
        delta =
          point && d.point ? point.sub(d.point).multiplyScalar(precision) : new THREE.Vector3();
      if (h.kind === 'axis') {
        const screenDelta: P = [p[0] - d.start[0], p[1] - d.start[1]],
          projected = screenDelta[0] * d.axisScreen[0] + screenDelta[1] * d.axisScreen[1];
        let travel = projected;
        if (this.joint?.type === 'prismatic') {
          if (
            d.slideAxis < 0 &&
            Math.hypot(
              screenDelta[0] - d.axisScreen[0] * projected,
              screenDelta[1] - d.axisScreen[1] * projected,
            ) > 8
          ) {
            d.slideAxis = Math.abs(screenDelta[0]) > Math.abs(screenDelta[1]) ? 0 : 1;
            const component = d.axisScreen[d.slideAxis];
            d.slideSign =
              Math.abs(component) >= 0.15 ? Math.sign(component) : d.slideAxis === 0 ? 1 : -1;
            d.slideOrigin = p;
            d.slideTravel =
              (d.slideLast[0] - d.start[0]) * d.axisScreen[0] +
              (d.slideLast[1] - d.start[1]) * d.axisScreen[1] +
              (p[d.slideAxis] - d.slideLast[d.slideAxis]) * d.slideSign;
          }
          if (d.slideAxis >= 0)
            travel = d.slideTravel + (p[d.slideAxis] - d.slideOrigin[d.slideAxis]) * d.slideSign;
          d.slideLast = p;
        }
        scalar = travel * d.worldPerPoint * precision;
        rawScalar = scalar;
        if (snap) scalar = snapValue(scalar, this.translationStep);
        delta.copy(h.direction).multiplyScalar(scalar);
      } else if (snap) {
        const local = d.basis.map((axis) => snapValue(axis.dot(delta), this.translationStep));
        delta.set(0, 0, 0);
        for (let i = 0; i < 3; i++)
          if (h.kind !== 'plane' || i !== h.axis) delta.addScaledVector(d.basis[i], local[i]);
      }
      if (this.joint) {
        const requested = (d.jointValue ?? 0) + scalar;
        setJoint(this.doc, this.joint.id, requested);
        if (
          Math.abs(this.joint.value - requested) > 1e-12 ||
          (this.joint.value <= this.joint.lower + 1e-12 &&
            (d.jointValue ?? 0) + rawScalar < this.joint.lower - 1e-12) ||
          (this.joint.value >= this.joint.upper - 1e-12 &&
            (d.jointValue ?? 0) + rawScalar > this.joint.upper + 1e-12)
        )
          this.rebaseJointDrag(d, p);
      } else if (this.tool === 'scale') {
        const amount =
          h.axis < 0
            ? Math.exp((p[0] - d.start[0] - (p[1] - d.start[1])) * 0.01 * precision)
            : Math.max(0.001, 1 + scalar / this.scale);
        d.object.scale.copy(d.scale);
        if (h.axis < 0) d.object.scale.multiplyScalar(amount);
        else
          d.object.scale.setComponent(
            h.axis,
            Math.max(0.001, d.scale.getComponent(h.axis) * amount),
          );
      } else {
        const target = d.worldPosition.clone().add(delta);
        if (d.object.parent) d.object.parent.worldToLocal(target);
        d.object.position.copy(target);
      }
    }
    d.last = p;
    d.object.updateMatrixWorld(true);
    if (this.configurationJoint)
      this.writeConfigurationPose(
        this.configurationJoint,
        d.object.getWorldPosition(new THREE.Vector3()),
        d.object.getWorldQuaternion(new THREE.Quaternion()),
      );
    if (
      !this.configurationJoint &&
      this.tool !== 'joint' &&
      this.doc.nodes.find((n) => n.id === this.selected)?.kind === 'joint'
    )
      authorJointTransform(this.doc, this.selected!);
    this.callbacks.onChange();
    this.rebuild();
    this.draw();
    this.callbacks.invalidate();
  }
  end(cancel = false) {
    if (this.jointEdit) {
      this.endJointEdit(cancel);
      return;
    }
    const d = this.drag;
    if (!d) return;
    if (cancel) {
      d.object.position.copy(d.position);
      d.object.quaternion.copy(d.quaternion);
      d.object.scale.copy(d.scale);
      if (this.joint) setJoint(this.doc, this.joint.id, d.jointOriginValue ?? 0);
      if (this.configurationJoint)
        setJointConfiguration(this.doc, this.configurationJoint.id, {
          position: d.configurationPosition,
          rotation: d.configurationRotation,
        });
      d.object.updateMatrixWorld(true);
      if (
        !this.configurationJoint &&
        this.tool !== 'joint' &&
        this.doc.nodes.find((n) => n.id === this.selected)?.kind === 'joint'
      )
        authorJointTransform(this.doc, this.selected!);
      this.callbacks.onChange();
    }
    this.drag = null;
    this.callbacks.onCommit?.();
    this.rebuild();
    this.draw();
    this.callbacks.invalidate();
  }
  private writeConfigurationPose(
    joint: Joint,
    position: THREE.Vector3,
    rotation: THREE.Quaternion,
  ) {
    joint.object.parent?.updateWorldMatrix(true, false);
    const origin = new THREE.Matrix4().multiplyMatrices(
        joint.object.parent?.matrixWorld ?? new THREE.Matrix4(),
        joint.origin,
      ),
      baseRotation = new THREE.Quaternion().setFromRotationMatrix(origin).invert();
    setJointConfiguration(this.doc, joint.id, {
      position:
        joint.type === 'free'
          ? position.clone().applyMatrix4(origin.clone().invert()).toArray()
          : undefined,
      rotation: baseRotation.multiply(rotation).normalize().toArray(),
    });
  }
  setSnapping(snap: boolean) {
    this.snapModifier = snap;
    if (this.active) this.move(this.currentPointer, false, snap);
  }
  preciseInput(p: P): PreciseGizmoTarget | null {
    const handle = this.hit(p),
      object = this.object,
      doc = this.doc,
      selected = this.selected,
      tool = this.tool,
      frame = this.frame,
      joint = this.joint,
      configurationJoint = this.configurationJoint;
    if (
      !handle ||
      !object ||
      handle.kind === 'trackball' ||
      handle.kind === 'plane' ||
      (handle.kind === 'screen' && tool !== 'scale')
    )
      return null;
    const axis = handle.axis,
      direction = handle.direction.clone(),
      angular = handle.kind === 'ring',
      uniform = axis < 0 && tool === 'scale';
    object.updateWorldMatrix(true, false);
    const worldPosition = object.getWorldPosition(new THREE.Vector3()),
      worldQuaternion = object.getWorldQuaternion(new THREE.Quaternion());
    let absoluteValue: number | undefined;
    if (joint) absoluteValue = angular ? THREE.MathUtils.radToDeg(joint.value) : joint.value;
    else if (tool === 'scale') {
      if (!uniform) absoluteValue = object.scale.getComponent(axis);
      else if (
        Math.abs(object.scale.x - object.scale.y) + Math.abs(object.scale.y - object.scale.z) <
        1e-12
      )
        absoluteValue = object.scale.x;
    } else if (frame === 'world' && axis >= 0)
      absoluteValue = angular
        ? THREE.MathUtils.radToDeg(
            new THREE.Euler().setFromQuaternion(worldQuaternion, 'ZYX').toArray()[axis] as number,
          )
        : worldPosition.getComponent(axis);
    const valid = () =>
      this.doc === doc &&
      this.selected === selected &&
      this.tool === tool &&
      this.frame === frame &&
      (configurationJoint
        ? this.configurationJoint === configurationJoint
        : doc.nodes.some((n) => n.id === selected && n.object === object));
    return {
      action: tool === 'scale' ? 'Scale' : angular ? 'Rotate' : 'Move',
      label: joint?.name ?? (uniform ? 'Uniform' : axis < 0 ? 'Screen' : 'XYZ'[axis]),
      unit: tool === 'scale' ? '×' : angular ? '°' : 'm',
      absoluteValue,
      valid,
      apply: (value, absolute) => {
        if (!valid()) throw new Error('The gizmo target changed; reopen precise input');
        if (!Number.isFinite(value)) throw new Error('Enter a finite numeric value');
        if (absolute && absoluteValue === undefined)
          throw new Error('Absolute input is unavailable for this gizmo handle');
        object.updateWorldMatrix(true, false);
        const position = object.getWorldPosition(new THREE.Vector3()),
          rotation = object.getWorldQuaternion(new THREE.Quaternion()),
          scale = object.scale.clone();
        let jointValue = joint?.value;
        if (joint) {
          const amount = angular ? THREE.MathUtils.degToRad(value) : value;
          jointValue = absolute ? amount : joint.value + amount;
        } else if (tool === 'scale') {
          if (uniform) {
            if (absolute) scale.setScalar(value);
            else scale.multiplyScalar(1 + value);
          } else scale.setComponent(axis, absolute ? value : scale.getComponent(axis) + value);
          if (Math.min(scale.x, scale.y, scale.z) < 0.001)
            throw new Error('Scale must be at least 0.001');
        } else if (angular) {
          if (absolute) {
            const euler = new THREE.Euler().setFromQuaternion(rotation, 'ZYX'),
              values = [euler.x, euler.y, euler.z];
            values[axis] = THREE.MathUtils.degToRad(value);
            rotation.setFromEuler(new THREE.Euler(values[0], values[1], values[2], 'ZYX'));
          } else
            rotation.premultiply(
              new THREE.Quaternion().setFromAxisAngle(direction, THREE.MathUtils.degToRad(value)),
            );
        } else if (absolute) position.setComponent(axis, value);
        else position.addScaledVector(direction, value);
        const oldPosition = object.position.clone(),
          oldRotation = object.quaternion.clone(),
          oldScale = object.scale.clone(),
          oldJoint = joint?.value,
          oldConfigurationPosition = configurationJoint?.position?.toArray(),
          oldConfigurationRotation = configurationJoint?.rotation?.toArray();
        this.callbacks.onBegin?.();
        try {
          if (joint) setJoint(doc, joint.id, jointValue!);
          else if (configurationJoint)
            this.writeConfigurationPose(configurationJoint, position, rotation);
          else {
            if (object.parent) {
              object.parent.worldToLocal(position);
              rotation.premultiply(
                object.parent.getWorldQuaternion(new THREE.Quaternion()).invert(),
              );
            }
            object.position.copy(position);
            object.quaternion.copy(rotation);
            object.scale.copy(scale);
            object.updateMatrixWorld(true);
            if (doc.nodes.find((n) => n.id === selected)?.kind === 'joint')
              authorJointTransform(doc, selected!);
          }
          this.callbacks.onChange();
        } catch (error) {
          object.position.copy(oldPosition);
          object.quaternion.copy(oldRotation);
          object.scale.copy(oldScale);
          if (joint) setJoint(doc, joint.id, oldJoint!);
          if (configurationJoint)
            setJointConfiguration(doc, configurationJoint.id, {
              position: oldConfigurationPosition,
              rotation: oldConfigurationRotation,
            });
          object.updateMatrixWorld(true);
          if (
            !joint &&
            !configurationJoint &&
            doc.nodes.find((n) => n.id === selected)?.kind === 'joint'
          )
            authorJointTransform(doc, selected!);
          this.callbacks.onChange();
          throw error;
        } finally {
          this.callbacks.onCommit?.();
          this.rebuild();
          this.draw();
          this.callbacks.invalidate();
        }
      },
    };
  }
  dispose() {
    if (this.dwellTimer !== undefined) clearTimeout(this.dwellTimer);
    this.dragLink?.dispose();
    this.raster.dispose();
    this.handles = [];
  }
}
