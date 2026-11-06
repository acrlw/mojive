import { useEffect, useId, useRef, useState } from 'react';
import * as THREE from 'three';
import type { Joint, SceneDocument } from './contracts';
import { SearchField, Button, Empty, Icon } from './components';
import { JointEditor, jointContext, jointIcon, jointTypeLabel } from './JointEditor';
import { t } from './i18n';
import { jointDisplayPose } from './joint-pose';
import { EntityCard } from './EntityCard';
import './joint-panel.css';

type Props = {
  doc: SceneDocument | null;
  selectedId?: string | null;
  activeJointId?: string | null;
  onSelect: (id: string) => void;
  onFocus: (id: string) => void;
  onBegin: () => void;
  onCommit: () => void;
  onCancel?: () => void;
  onChange: () => void;
  showHeader?: boolean;
};
function PoseSummary({ joint }: { joint: Joint }) {
  const pose = jointDisplayPose(joint);
  const rotation = new THREE.Euler().setFromQuaternion(pose.rotation, 'ZYX');
  const groups: [string, number[]][] =
    joint.type === 'free' ? [['Position', pose.position.toArray()]] : [];
  groups.push(['Rotation', [rotation.x, rotation.y, rotation.z].map(THREE.MathUtils.radToDeg)]);
  return (
    <div className="joint-pose-summary">
      {groups.map(([label, values]) => (
        <div className="joint-summary-group" key={label}>
          <span className="joint-summary-label">
            {t(label)} <span>({label === 'Position' ? 'm' : '°'})</span>
          </span>
          <div className="joint-summary-values">
            {values.map((value, index) => (
              <span key={index}>
                <i className={`axis-${'xyz'[index]}`}>{'XYZ'[index]}</i>
                <span>{(Math.abs(value) < 0.0005 ? 0 : value).toFixed(3)}</span>
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
export function JointPanel({
  doc,
  selectedId,
  activeJointId,
  onSelect,
  onFocus,
  onBegin,
  onCommit,
  onCancel,
  onChange,
  showHeader = true,
}: Props) {
  const [searchVisible, setSearchVisible] = useState(false),
    [search, setSearch] = useState(''),
    [sorted, setSorted] = useState(false);
  const panelId = useId(),
    headingSelection = useRef<string | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({}),
    [status, setStatus] = useState('');
  useEffect(() => {
    setExpanded({});
    setStatus('');
  }, [doc?.root.uuid]);
  useEffect(() => {
    const id = activeJointId || selectedId,
      joint = doc?.joints.find((j) => j.id === id);
    // Selection echoed by this heading must not override its explicit collapse.
    const selectedFromHeading = headingSelection.current === selectedId;
    headingSelection.current = null;
    if (selectedFromHeading && joint?.id === selectedId) return;
    if (joint && (joint.type === 'ball' || joint.type === 'free'))
      setExpanded((previous) => ({ ...previous, [joint.id]: true }));
  }, [activeJointId, selectedId, doc?.root.uuid]);
  const movable = doc?.joints.filter((j) => j.type !== 'fixed') ?? [];
  const shown = movable.filter((j) =>
    `${j.name} ${jointTypeLabel(j.type)}`.toLocaleLowerCase().includes(search.toLocaleLowerCase()),
  );
  if (sorted) shown.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  const copy = async (text: string) => {
    try {
      if (window.mojive) await window.mojive.writeClipboardText(text);
      else await navigator.clipboard.writeText(text);
      setStatus('');
    } catch (error) {
      setStatus(`${t('Copy failed')}: ${String(error)}`);
    }
  };
  const copyState = () =>
    copy(
      JSON.stringify(
        movable.flatMap((j) => {
          if (j.type !== 'ball' && j.type !== 'free') return [j.value];
          const pose = jointDisplayPose(j),
            q = pose.rotation,
            rotation = [q.w, q.x, q.y, q.z];
          return j.type === 'free' ? [...pose.position.toArray(), ...rotation] : rotation;
        }),
      ),
    );
  return (
    <div className="joint-panel">
      {showHeader && (
        <div className="joint-panel-header">
          <h2>{t('Joints')}</h2>
          <div className="joint-panel-tools">
            <Button
              icon="search"
              title="Search joints"
              active={searchVisible}
              onClick={() => {
                setSearchVisible(!searchVisible);
                if (searchVisible) setSearch('');
              }}
            />
            <Button
              icon="sort"
              title="Sort by name"
              active={sorted}
              onClick={() => setSorted(!sorted)}
            />
            <Button icon="copy" title="Copy state" disabled={!movable.length} onClick={copyState} />
          </div>
        </div>
      )}
      {searchVisible && (
        <SearchField
          className="joint-panel-search"
          autoFocus
          label={t('Search joints')}
          placeholder={`${t('Search joints')}…`}
          value={search}
          onChange={setSearch}
        />
      )}
      {status && (
        <p className="joint-panel-status" role="status">
          {status}
        </p>
      )}
      <div className="joint-panel-list">
        {shown.map((joint) => {
          const pose = joint.type === 'ball' || joint.type === 'free',
            open = expanded[joint.id] ?? false;
          const selected = selectedId === joint.id || activeJointId === joint.id;
          return (
            <EntityCard
              className="joint-entry"
              name={joint.name}
              icon={jointIcon(joint.type)}
              context={jointContext(joint)}
              selected={selected}
              expanded={pose ? open : undefined}
              contentId={`${panelId}-${joint.id}-pose`}
              disclosureTitle={t(open ? 'Collapse joint pose' : 'Expand joint pose')}
              key={joint.id}
              data-joint-id={joint.id}
              heading={{
                title: `${joint.name}\n${jointTypeLabel(joint.type)}\n${jointContext(joint)}`,
                'aria-label': pose
                  ? `${t(open ? 'Collapse joint pose' : 'Expand joint pose')}: ${joint.name}`
                  : joint.name,
                onClick: (event) => {
                  // A double-click focuses after its first activation; it must not toggle twice.
                  if (event.detail > 1) return;
                  if (selectedId !== joint.id) headingSelection.current = joint.id;
                  onSelect(joint.id);
                  if (pose) setExpanded((previous) => ({ ...previous, [joint.id]: !open }));
                },
                onDoubleClick: () => onFocus(joint.id),
                onKeyDown: (event) => {
                  if (event.key === ' ' || event.key === 'Enter') {
                    event.stopPropagation();
                    if (event.repeat) event.preventDefault();
                  }
                },
                onKeyUp: (event) => {
                  if (event.key === ' ' || event.key === 'Enter') event.stopPropagation();
                },
                onContextMenu: (event) => {
                  event.preventDefault();
                  void copy(joint.name);
                },
              }}
            >
              {pose && !open ? (
                <PoseSummary joint={joint} />
              ) : (
                <JointEditor
                  doc={doc!}
                  joint={joint}
                  onBegin={() => {
                    onSelect(joint.id);
                    onBegin();
                  }}
                  onCommit={onCommit}
                  onCancel={onCancel}
                  onChange={onChange}
                />
              )}
            </EntityCard>
          );
        })}
        {!shown.length && (
          <Empty
            title={t(movable.length ? 'No matching joints' : 'No movable joints')}
            action={
              movable.length ? (
                <Button onClick={() => setSearch('')}>{t('Clear search')}</Button>
              ) : undefined
            }
          />
        )}
      </div>
    </div>
  );
}
