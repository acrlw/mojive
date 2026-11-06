import { useEffect, useId, useRef, useState } from 'react';
import type { SceneDocument, SceneNode } from './contracts';
import { SearchField, Button, Empty, Icon, Section, Squircle } from './components';
import { readCameraView } from './scene-camera';
import { useControlPopover } from './CustomControls';
import { t, useLanguage } from './i18n';
import './camera-panel.css';
export interface CamerasPanelProps {
  doc: SceneDocument | null;
  selectedId?: string | null;
  activeCameraId?: string | null;
  piloting?: boolean;
  onSelect: (id: string) => void;
  onView: (id: string | null, pilot?: boolean) => void;
  onFocus: (id?: string) => void;
  onAlign: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
  onCapture: (id: string) => void;
  onPreset: (preset: string) => void;
}
export function CamerasPanel({
  doc,
  selectedId,
  activeCameraId,
  piloting,
  onSelect,
  onView,
  onFocus,
  onAlign,
  onCreate,
  onDelete,
  onCapture,
  onPreset,
}: CamerasPanelProps) {
  useLanguage();
  const [searchVisible, setSearchVisible] = useState(false),
    [search, setSearch] = useState(''),
    [sorted, setSorted] = useState(false);
  useEffect(() => setSearch(''), [doc?.root.uuid]);
  const cameras = doc?.nodes.filter((node) => node.kind === 'camera') ?? [];
  const shown = cameras.filter((node) =>
    node.name.toLocaleLowerCase().includes(search.toLocaleLowerCase()),
  );
  if (sorted) shown.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  return (
    <div className="cameras-panel">
      <div className="camera-panel-header">
        <h2>{t('Cameras')}</h2>
        <div className="camera-panel-tools">
          <Button
            icon="search"
            title="Search cameras"
            aria-label={t('Search cameras')}
            active={searchVisible}
            onClick={() => {
              setSearchVisible(!searchVisible);
              if (searchVisible) setSearch('');
            }}
          />
          <Button
            icon="sort"
            title="Sort by name"
            aria-label={t('Sort by name')}
            active={sorted}
            onClick={() => setSorted(!sorted)}
          />
          <Button
            icon="plus"
            title="Create camera from view"
            aria-label={t('Create camera from view')}
            disabled={!doc}
            onClick={onCreate}
          />
        </div>
      </div>
      {searchVisible && (
        <SearchField
          className="camera-panel-search"
          autoFocus
          label={t('Search cameras')}
          placeholder={`${t('Search cameras')}…`}
          value={search}
          onChange={setSearch}
        />
      )}
      <div className="camera-panel-scroll">
        <div className="camera-editor-source">
          <Button icon="perspective" active={!activeCameraId} onClick={() => onView(null)}>
            {t('Editor Camera')}
          </Button>
          {activeCameraId && <span>{t(piloting ? 'Pilot camera' : 'View Camera')}</span>}
        </div>
        <Section title="Scene cameras">
          {shown.map((node) => (
            <CameraCard
              key={node.id}
              node={node}
              selected={selectedId === node.id}
              active={activeCameraId === node.id}
              piloting={piloting}
              onSelect={onSelect}
              onView={onView}
              onFocus={onFocus}
              onAlign={onAlign}
              onCapture={onCapture}
              onDelete={onDelete}
            />
          ))}
          {!shown.length && (
            <Empty
              title={t(cameras.length ? 'No matching cameras' : 'No scene cameras')}
              action={
                cameras.length ? (
                  <Button onClick={() => setSearch('')}>{t('Clear search')}</Button>
                ) : (
                  <Button icon="plus" disabled={!doc} onClick={onCreate}>
                    {t('Create camera from view')}
                  </Button>
                )
              }
            />
          )}
        </Section>
        <Section title="View presets">
          <div className="camera-preset-grid">
            {['front', 'back', 'left', 'right', 'top', 'bottom', 'iso'].map((preset) => (
              <Button icon={`view-${preset}`} key={preset} onClick={() => onPreset(preset)}>
                {t(preset === 'iso' ? 'Isometric' : preset[0].toUpperCase() + preset.slice(1))}
              </Button>
            ))}
            <Button icon="fit" onClick={() => onFocus()}>
              {t('Frame all')}
            </Button>
          </div>
        </Section>
      </div>
    </div>
  );
}

function CameraCard({
  node,
  selected,
  active,
  piloting,
  onSelect,
  onView,
  onFocus,
  onAlign,
  onCapture,
  onDelete,
}: {
  node: SceneNode;
  selected: boolean;
  active: boolean;
  piloting?: boolean;
} & Pick<
  CamerasPanelProps,
  'onSelect' | 'onView' | 'onFocus' | 'onAlign' | 'onCapture' | 'onDelete'
>) {
  const [open, setOpen] = useState(false),
    trigger = useRef<HTMLDivElement>(null),
    popup = useRef<HTMLDivElement>(null),
    initial = useRef(0),
    menuId = useId();
  useControlPopover(trigger, popup, open, () => setOpen(false));
  useEffect(() => {
    if (open) {
      const items = popup.current?.querySelectorAll<HTMLButtonElement>('[role=menuitem]');
      items?.[initial.current < 0 ? items.length - 1 : 0]?.focus();
    }
  }, [open]);
  const close = (focus = false) => {
    setOpen(false);
    if (focus) trigger.current?.querySelector('button')?.focus();
  };
  const view = readCameraView(node),
    projection = view.projection === 'perspective' ? 'Perspective' : 'Orthographic',
    viewing = active && !piloting,
    pilotingCamera = active && !!piloting;
  const actions = [
    { label: 'Focus camera', icon: 'fit', run: () => onFocus(node.id) },
    { label: 'Align to view', icon: 'camera-align', run: () => onAlign(node.id) },
    { label: 'Capture image', icon: 'capture-image', run: () => onCapture(node.id) },
    { label: 'Delete camera', icon: 'delete', run: () => onDelete(node.id) },
  ].filter((action) => !node.object.userData.physicsOwned || !['Align to view', 'Delete camera'].includes(action.label));
  return (
    <Squircle
      radius={12}
      className={`camera-entry ${selected ? 'selected' : ''}`}
      data-camera-id={node.id}
    >
      <div className="camera-entry-heading">
        <button
          className="camera-entry-name"
          onClick={() => onSelect(node.id)}
          onDoubleClick={() => onView(node.id)}
          data-tooltip={node.name}
          data-tooltip-overflow="span:not(.icon)"
          aria-label={node.name}
        >
          <Icon name="camera" />
          <span>{node.name}</span>
        </button>
        <div ref={trigger} className="camera-entry-more">
          <Button
            icon="more"
            title="More camera actions"
            aria-label={t('More camera actions')}
            aria-haspopup="menu"
            aria-expanded={open}
            aria-controls={open ? menuId : undefined}
            onClick={() => {
              initial.current = 0;
              setOpen(!open);
            }}
            onKeyDown={(event) => {
              if (['ArrowDown', 'ArrowUp'].includes(event.key)) {
                event.preventDefault();
                initial.current = event.key === 'ArrowUp' ? -1 : 0;
                setOpen(true);
              }
            }}
          />
        </div>
      </div>
      <div className="camera-entry-body">
        <div className="camera-entry-meta">
          <span className="camera-entry-projection">
            <Icon name={view.projection} size={14} />
            <span>{t(projection)}</span>
          </span>
          <span className="camera-entry-resolution">
            {view.resolution[0]}×{view.resolution[1]}
          </span>
        </div>
        <div
          className="camera-entry-actions"
          role="group"
          aria-label={`${node.name} ${t('Camera view')}`}
        >
          <Button
            icon="eye"
            title={viewing ? 'Return to Editor Camera' : 'View Camera'}
            aria-label={t(viewing ? 'Return to Editor Camera' : 'View Camera')}
            aria-pressed={viewing}
            active={viewing}
            onClick={() => onView(viewing ? null : node.id)}
          />
          <Button
            icon="move"
            disabled={!!node.object.userData.physicsOwned}
            title={pilotingCamera ? 'Return to Editor Camera' : 'Pilot camera'}
            aria-label={t(pilotingCamera ? 'Return to Editor Camera' : 'Pilot camera')}
            aria-pressed={pilotingCamera}
            active={pilotingCamera}
            onClick={() => (pilotingCamera ? onView(null) : onView(node.id, true))}
          />
        </div>
      </div>
      {open && (
        <div
          ref={popup}
          id={menuId}
          className="camera-entry-menu"
          role="menu"
          aria-label={`${node.name} ${t('More camera actions')}`}
          popover="auto"
          onKeyDown={(event) => {
            event.stopPropagation();
            const items = [
                ...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role=menuitem]'),
              ],
              index = items.indexOf(document.activeElement as HTMLButtonElement);
            if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
              event.preventDefault();
              const next =
                event.key === 'Home'
                  ? 0
                  : event.key === 'End'
                    ? items.length - 1
                    : (index + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
              items[next]?.focus();
            } else if (event.key === 'Escape') {
              event.preventDefault();
              close(true);
            } else if (event.key === 'Tab') close(true);
          }}
        >
          {actions.map(({ label, icon, run }) => (
            <Button
              key={label}
              icon={icon}
              role="menuitem"
              title={label}
              onClick={() => {
                close(true);
                run();
              }}
            >
              {t(label)}
            </Button>
          ))}
        </div>
      )}
    </Squircle>
  );
}
