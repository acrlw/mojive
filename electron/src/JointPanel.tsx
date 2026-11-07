import { useEffect, useId, useRef, useState } from 'react';
import type { SceneDocument } from './contracts';
import { SearchField, Button, Empty } from './components';
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
    localSelection = useRef<string | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({}),
    [status, setStatus] = useState('');
  useEffect(() => {
    setExpanded({});
    setStatus('');
  }, [doc?.root.uuid]);
  useEffect(() => {
    const id = activeJointId || selectedId,
      joint = doc?.joints.find((j) => j.id === id);
    // Selection from a card must not expand a compact pose during an edit.
    const selectedLocally = localSelection.current === selectedId;
    localSelection.current = null;
    if (selectedLocally && joint?.id === selectedId) return;
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
              context={pose ? jointContext(joint) : undefined}
              controls={
                !pose ? (
                  <JointEditor
                    doc={doc!}
                    joint={joint}
                    panel
                    scalarPart="value"
                    onBegin={() => {
                      onSelect(joint.id);
                      onBegin();
                    }}
                    onCommit={onCommit}
                    onCancel={onCancel}
                    onChange={onChange}
                  />
                ) : undefined
              }
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
                  if (selectedId !== joint.id) localSelection.current = joint.id;
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
              <JointEditor
                panel
                compactPose={pose && !open}
                scalarPart={pose ? 'all' : 'rail'}
                doc={doc!}
                joint={joint}
                onBegin={() => {
                  if (selectedId !== joint.id) localSelection.current = joint.id;
                  onSelect(joint.id);
                  onBegin();
                }}
                onCommit={onCommit}
                onCancel={onCancel}
                onChange={onChange}
              />
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
