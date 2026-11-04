/** Readout wording and semantic palette from ui/gizmo/dragging.py and overlay.py. */
import { roundEven } from './interaction_math';
import * as THREE from 'three';
import { AXIS_COLORS, type P } from './geometry';
export interface ActiveFeedback {
  kind: 'axis' | 'plane' | 'screen' | 'ring' | 'trackball';
  axis: number;
  angle: number;
  translate?: (text: string) => string;
  snapping: boolean;
  translationSnap?: number;
  rotationSnap?: number;
  delta: THREE.Vector3;
  basis: THREE.Vector3[];
  scale?: { current: THREE.Vector3; original: THREE.Vector3 };
  joint?: { name: string; type: string; value: number; lower: number; upper: number };
  trackball?: P;
}
const signed = (value: number, digits: number) =>
  `${value < 0 || Object.is(value, -0) ? '-' : '+'}${Math.abs(value).toFixed(digits)}`;
export function activeLabel(input: ActiveFeedback) {
  const tr = input.translate ?? ((text: string) => text);
  const { kind, axis, joint, snapping } = input,
    angular = kind === 'ring' || kind === 'trackball',
    snap = snapping
      ? angular
        ? ` · ${tr('SNAP')} ${input.rotationSnap ?? 5}°`
        : ` · ${tr('SNAP')} ${input.translationSnap ?? 0.1} m`
      : '';
  if (input.scale) {
    const factor =
      input.scale.current.getComponent(Math.max(0, axis)) /
      input.scale.original.getComponent(Math.max(0, axis));
    return `${tr('Scale')} ${axis >= 0 ? 'XYZ'[axis] + ' ' : ''}${factor.toFixed(3)}×${snap}`;
  }
  if (kind === 'trackball') {
    const angles = input.trackball ?? [0, 0];
    return `${tr('Trackball')} ${signed(THREE.MathUtils.radToDeg(angles[0]), 1)}° ${signed(THREE.MathUtils.radToDeg(angles[1]), 1)}°${snap}`;
  }
  const name = joint?.name ?? (axis < 0 ? tr('Screen') : 'XYZ'[axis]);
  if (angular) {
    const degrees = roundEven(THREE.MathUtils.radToDeg(joint?.value ?? input.angle) * 10) / 10,
      turns = Math.floor(Math.abs(degrees) / 360);
    return `${name} ${signed(degrees, 1)}°${turns ? ` · ${turns}×360°` : ''}${snap}`;
  }
  const local = input.basis.map((v) => v.dot(input.delta));
  if (kind === 'axis') return `${name} ${signed(joint?.value ?? local[axis], 3)} m${snap}`;
  if (kind === 'plane') {
    const a = (axis + 1) % 3,
      b = (axis + 2) % 3;
    return `${'XYZ'[a]} ${signed(local[a], 3)}  ${'XYZ'[b]} ${signed(local[b], 3)} m${snap}`;
  }
  return `X ${signed(local[0], 3)}  Y ${signed(local[1], 3)}  Z ${signed(local[2], 3)} m${snap}`;
}
const raw = (hex: string) => [0, 1, 2].map((i) => parseInt(hex.slice(1 + 2 * i, 3 + 2 * i), 16));
const rgb = (v: number[]) => `rgb(${v.join(',')})`;
export function activePalette(axis: number, joint = false) {
  if (joint) return { fill: '#b8d2ac', pressed: '#b8d2ac', dark: '#67875a' };
  if (axis < 0) return { fill: '#67875a', pressed: '#b8d2ac', dark: '#67875a' };
  const color = AXIS_COLORS[axis],
    base = raw(color);
  return {
    fill: color,
    pressed: rgb(base.map((v) => v + (255 - v) * 0.36)),
    dark: rgb(base.map((v) => v * 0.58)),
  };
}
export function jointMarker(_joint: NonNullable<ActiveFeedback['joint']>) {
  return '#b8d2ac';
}
/** Native readout_marker_ink: preserve hue while guaranteeing 3:1 opaque contrast. */
export function readoutMarker(color: string, surface: string) {
  const components = (value: string) =>
    value.startsWith('#')
      ? [0, 1, 2].map((i) => parseInt(value.slice(1 + i * 2, 3 + i * 2), 16) / 255)
      : (value.match(/[\d.]+/g) ?? [])
          .slice(0, 3)
          .map(Number)
          .map((v) => v / 255);
  const luminance = (rgb: number[]) =>
      rgb
        .map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4))
        .reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i], 0),
    base = components(color),
    background = luminance(components(surface)),
    contrast = (rgb: number[]) => {
      const foreground = luminance(rgb);
      return (Math.max(background, foreground) + 0.05) / (Math.min(background, foreground) + 0.05);
    };
  if (contrast(base) >= 3) return color;
  const target = background > 0.5 ? 0 : 1;
  let lo = 0,
    hi = 1;
  for (let i = 0; i < 12; i++) {
    const amount = (lo + hi) / 2,
      candidate = base.map((v) => v + (target - v) * amount);
    if (contrast(candidate) >= 3) hi = amount;
    else lo = amount;
  }
  return `rgb(${base.map((v) => Math.round((v + (target - v) * hi) * 255)).join(',')})`;
}
