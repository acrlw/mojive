import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import type { Joint, SceneDocument } from './contracts';
import { setJoint } from './scene';
import { jointDisplayPose, setJointDisplayPose } from './joint-pose';
import { Numeric, VectorFields } from './components';
import { t } from './i18n';
import './joint-panel.css';

export const jointIcon = (type: string) =>
  type === 'prismatic'
    ? 'joint-slide'
    : type === 'ball'
      ? 'joint-ball'
      : type === 'free'
        ? 'joint-free'
        : 'joint-hinge';
export const jointTypeLabel = (type: string) =>
  t(
    type === 'prismatic'
      ? 'Slide joint'
      : type === 'ball'
        ? 'Ball joint'
        : type === 'free'
          ? 'Free joint'
          : 'Hinge joint',
  );
export function jointContext(joint: Joint) {
  if (joint.type === 'ball' || joint.type === 'free')
    return `${joint.type === 'ball' ? 3 : 6} ${t('DoF')}`;
  if (joint.type === 'continuous') return t('Unbounded');
  const display = (value: number) =>
    Number((joint.type === 'prismatic' ? value : THREE.MathUtils.radToDeg(value)).toPrecision(4))
      .toString()
      .replace('-', '−');
  return `${display(joint.lower)}…${display(joint.upper)}${joint.type === 'prismatic' ? ' m' : '°'}`;
}

type JointEditorProps = {
  doc: SceneDocument;
  joint: Joint;
  onBegin: () => void;
  onCommit: () => void;
  onChange: () => void;
  controlRadius?: number;
};

/** One pointer/key owner, including outside release and Escape at a limit. */
function JointRail({
  joint,
  onBegin,
  onCommit,
  onChange,
}: Omit<JointEditorProps, 'doc' | 'onChange'> & { onChange: (value: number) => void }) {
  const drag = useRef<{
    initial: number;
    x: number;
    offset: number;
    pointer: number | null;
  } | null>(null);
  const element = useRef<HTMLDivElement>(null);
  const [initial, setInitial] = useState<number | null>(null);
  const callbacks = useRef({ onBegin, onCommit, onChange, joint });
  callbacks.current = { onBegin, onCommit, onChange, joint };
  const bounded = joint.type !== 'continuous';
  const lo = joint.lower,
    hi = joint.upper;
  const fraction = (value: number) =>
    bounded ? THREE.MathUtils.clamp((value - lo) / (hi - lo || 1), 0, 1) : 0.5;
  const current = fraction(joint.value);
  const finish = (cancel = false) => {
    const state = drag.current;
    if (!state) return;
    drag.current = null;
    setInitial(null);
    if (cancel) callbacks.current.onChange(state.initial);
    callbacks.current.onCommit();
  };
  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || !drag.current) return;
      event.preventDefault();
      event.stopPropagation();
      finish(true);
    };
    window.addEventListener('keydown', escape, true);
    return () => window.removeEventListener('keydown', escape, true);
  }, []);
  const pointerValue = (x: number, precision = false) => {
    const state = drag.current,
      rect = element.current!.getBoundingClientRect();
    if (!state) return;
    const span = bounded ? hi - lo : Math.PI * 2;
    const scale = rect.width / element.current!.offsetWidth,
      inset = 8 * scale,
      travel = Math.max(1, rect.width - 16 * scale);
    const value =
      bounded && !precision
        ? lo + ((x - state.offset - rect.left - inset) / travel) * span
        : state.initial + ((x - state.x) / travel) * span * (precision ? 0.1 : 1);
    onChange(bounded ? THREE.MathUtils.clamp(value, lo, hi) : value);
  };
  return (
    <div
      ref={element}
      className={`joint-rail ${initial !== null ? 'held' : ''} ${bounded ? 'bounded' : 'unbounded'}`}
      role="slider"
      tabIndex={joint.mimic ? -1 : 0}
      aria-label={`${joint.name} slider`}
      aria-disabled={!!joint.mimic}
      aria-valuemin={bounded ? lo : undefined}
      aria-valuemax={bounded ? hi : undefined}
      aria-valuenow={joint.value}
      aria-valuetext={`${(joint.type === 'prismatic' ? joint.value : THREE.MathUtils.radToDeg(joint.value)).toFixed(3)} ${joint.type === 'prismatic' ? 'm' : '°'}`}
      onPointerDown={(event) => {
        if (event.button !== 0 || joint.mimic) return;
        event.preventDefault();
        event.currentTarget.focus();
        const rect = event.currentTarget.getBoundingClientRect(),
          scale = rect.width / event.currentTarget.offsetWidth;
        const center = rect.left + 8 * scale + fraction(joint.value) * (rect.width - 16 * scale);
        const offset = Math.abs(event.clientX - center) <= 7 * scale ? event.clientX - center : 0;
        drag.current = { initial: joint.value, x: event.clientX, offset, pointer: event.pointerId };
        setInitial(joint.value);
        onBegin();
        event.currentTarget.setPointerCapture(event.pointerId);
        pointerValue(event.clientX, event.altKey);
      }}
      onPointerMove={(event) => {
        if (drag.current?.pointer === event.pointerId) pointerValue(event.clientX, event.altKey);
      }}
      onPointerUp={(event) => {
        if (drag.current?.pointer !== event.pointerId) return;
        finish();
        if (event.currentTarget.hasPointerCapture(event.pointerId))
          event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onPointerCancel={() => finish(true)}
      onLostPointerCapture={() => finish()}
      onBlur={() => {
        if (drag.current?.pointer === null) finish();
      }}
      onKeyDown={(event) => {
        if (
          joint.mimic ||
          !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)
        )
          return;
        event.preventDefault();
        event.stopPropagation();
        if (!drag.current) {
          drag.current = { initial: joint.value, x: 0, offset: 0, pointer: null };
          setInitial(joint.value);
          onBegin();
        }
        const step =
          ((bounded ? hi - lo : Math.PI * 2) / 200) *
          (event.shiftKey ? 10 : event.altKey ? 0.1 : 1);
        const value =
          event.key === 'Home' && bounded
            ? lo
            : event.key === 'End' && bounded
              ? hi
              : joint.value + step * (['ArrowLeft', 'ArrowDown'].includes(event.key) ? -1 : 1);
        onChange(bounded ? THREE.MathUtils.clamp(value, lo, hi) : value);
      }}
      onKeyUp={(event) => {
        if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key))
          finish();
      }}
    >
      <div className="joint-rail-track">
        <span className="joint-rail-fill" style={{ width: `calc(${current * 100}% + 4px)` }} />
        <span className="joint-rail-thumb" style={{ left: `${current * 100}%` }} />
      </div>
    </div>
  );
}

export function JointEditor({
  doc,
  joint: j,
  onBegin,
  onCommit,
  onChange,
  controlRadius = 4,
}: JointEditorProps) {
  const displayedRotation = useRef(new THREE.Vector3());
  const displayedJoint = useRef(j.id);
  if (displayedJoint.current !== j.id) {
    displayedJoint.current = j.id;
    displayedRotation.current.set(0, 0, 0);
  }
  if (j.type === 'fixed') return null;
  if (j.type === 'ball' || j.type === 'free') {
    const pose = jointDisplayPose(j);
    const e = new THREE.Euler().setFromQuaternion(pose.rotation, 'ZYX');
    // Choose the equivalent caption nearest the preceding edit, preserving turns at ±180°.
    const candidates = [
      [e.x, e.y, e.z],
      [e.x + Math.PI, Math.PI - e.y, e.z + Math.PI],
    ].map((candidate) =>
      candidate.map((value, axis) => {
        const degrees = THREE.MathUtils.radToDeg(value),
          previous = displayedRotation.current.getComponent(axis);
        return degrees + Math.round((previous - degrees) / 360) * 360;
      }),
    );
    const distance = (candidate: number[]) =>
      candidate.reduce(
        (sum, value, axis) => sum + (value - displayedRotation.current.getComponent(axis)) ** 2,
        0,
      );
    const angles =
      distance(candidates[0]) <= distance(candidates[1]) ? candidates[0] : candidates[1];
    displayedRotation.current.fromArray(angles);
    return (
      <div className="joint-pose-fields">
        {j.type === 'free' && (
          <VectorFields
            controlRadius={controlRadius}
            label="Position"
            values={pose.position.toArray()}
            unit="m"
            onBegin={onBegin}
            onCommit={onCommit}
            onChange={(axis, value) => {
              const p = jointDisplayPose(j).position.toArray();
              p[axis] = value;
              setJointDisplayPose(doc, j, { position: p });
              onChange();
            }}
          />
        )}
        <VectorFields
          controlRadius={controlRadius}
          label="Rotation"
          values={angles}
          unit="°"
          onBegin={onBegin}
          onCommit={onCommit}
          onChange={(axis, value) => {
            const a = [...angles];
            a[axis] = value;
            displayedRotation.current.fromArray(a);
            const q = new THREE.Quaternion().setFromEuler(
              new THREE.Euler(
                ...(a.map(THREE.MathUtils.degToRad) as [number, number, number]),
                'ZYX',
              ),
            );
            setJointDisplayPose(doc, j, { rotation: q.toArray() });
            onChange();
          }}
        />
      </div>
    );
  }
  const slide = j.type === 'prismatic';
  const display = (value: number) => (slide ? value : THREE.MathUtils.radToDeg(value));
  const update = (value: number) => {
    setJoint(doc, j.id, value);
    onChange();
  };
  return (
    <div
      className={`joint-scalar-fields ${j.mimic ? 'mimic' : ''} ${j.type === 'continuous' ? 'unbounded' : ''}`}
      title={j.mimic ? `${t('Mimic')}: ${j.mimic.joint}` : undefined}
    >
      {j.type !== 'continuous' && (
        <JointRail joint={j} onBegin={onBegin} onCommit={onCommit} onChange={update} />
      )}
      <Numeric
        value={display(j.value)}
        label={`${j.name} value`}
        unit={slide ? 'm' : '°'}
        step={slide ? Math.min(0.01, (j.upper - j.lower) / 100) : 0.5}
        min={j.type === 'continuous' ? undefined : display(j.lower)}
        max={j.type === 'continuous' ? undefined : display(j.upper)}
        disabled={!!j.mimic}
        onBegin={onBegin}
        onCommit={onCommit}
        onChange={(value) => update(slide ? value : THREE.MathUtils.degToRad(value))}
      />
    </div>
  );
}
