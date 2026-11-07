import { RangeRail } from './RangeRail';
import { useRef } from 'react';
import { ScalarRail } from './ScalarRail';
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
    (
      {
        prismatic: 'Slide joint',
        ball: 'Ball joint',
        free: 'Free joint',
        fixed: 'Fixed joint',
        continuous: 'Continuous joint',
        revolute: 'Hinge joint',
      } as Record<string, string>
    )[type] ?? type,
  );
export function jointContext(joint: Joint) {
  if (joint.type === 'ball' || joint.type === 'free')
    return `${joint.type === 'ball' ? 3 : 6} ${t('DoF')}`;
  if (!Number.isFinite(joint.lower) || !Number.isFinite(joint.upper)) return t('Unbounded');
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
  onCancel?: () => void;
  onChange: () => void;
  showScalarLabel?: boolean;
  scalarPart?: 'all' | 'value' | 'rail';
  compactPose?: boolean;
  panel?: boolean;
};

export function JointEditor({
  doc,
  joint: j,
  onBegin,
  onCommit,
  onCancel,
  onChange,
  showScalarLabel = false,
  scalarPart = 'all',
  compactPose = false,
  panel = false,
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
      <div className={`joint-pose-fields ${compactPose ? 'joint-pose-summary' : ''}`}>
        {j.type === 'free' && (
          <VectorFields
            compact={compactPose}
            copyable={panel}
            label="Position"
            values={pose.position.toArray()}
            unit="m"
            onBegin={onBegin}
            onCommit={onCommit}
            onCancel={onCancel}
            onChange={(axis, value) => {
              const p = jointDisplayPose(j).position.toArray();
              p[axis] = value;
              setJointDisplayPose(doc, j, { position: p });
              onChange();
            }}
          />
        )}
        <VectorFields
          compact={compactPose}
          copyable={panel}
          label="Rotation"
          values={angles}
          unit="°"
          onBegin={onBegin}
          onCommit={onCommit}
          onCancel={onCancel}
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
  const bounded = Number.isFinite(j.lower) && Number.isFinite(j.upper) && j.upper > j.lower;
  const display = (value: number) => (slide ? value : THREE.MathUtils.radToDeg(value));
  const update = (value: number) => {
    setJoint(doc, j.id, value);
    onChange();
  };
  const value = (
    <Numeric
      value={display(j.value)}
      label={`${j.name} value`}
      unit={slide ? 'm' : '°'}
      appearance={panel ? 'readout' : 'field'}
      copyable={panel}
      step={slide ? (bounded ? Math.min(0.01, (j.upper - j.lower) / 100) : 0.01) : 0.5}
      min={bounded ? display(j.lower) : undefined}
      max={bounded ? display(j.upper) : undefined}
      disabled={!!j.mimic}
      onBegin={onBegin}
      onCommit={onCommit}
      onCancel={onCancel}
      onChange={(value) => update(slide ? value : THREE.MathUtils.degToRad(value))}
    />
  );
  const Rail = panel ? RangeRail : ScalarRail;
  const rail = bounded ? (
    <Rail
      value={j.value}
      min={j.lower}
      max={j.upper}
      referenceValue={j.referenceValue ?? 0}
      {...(panel ? { display } : {})}
      label={t('{{name}} slider', { name: j.name })}
      disabled={!!j.mimic}
      valueText={`${display(j.value).toFixed(3)} ${slide ? 'm' : '°'}`}
      onBegin={onBegin}
      onCommit={onCommit}
      onCancel={onCancel}
      onChange={update}
    />
  ) : null;
  if (scalarPart === 'value') return value;
  if (scalarPart === 'rail') return rail;
  const fields = (
    <div
      className={`joint-scalar-fields ${j.mimic ? 'mimic' : ''} ${!bounded ? 'unbounded' : ''}`}
      data-tooltip={j.mimic ? `${t('Mimic')}: ${j.mimic.joint}` : undefined}
    >
      {rail}
      {value}
    </div>
  );
  if (!showScalarLabel) return fields;
  const caption = t(slide ? 'Joint position' : 'Joint angle');
  return (
    <div className="property-row joint-value-property" role="group" aria-label={caption}>
      <span>{caption}</span>
      {fields}
    </div>
  );
}
