import { useEffect, useRef, useState } from 'react';
import type { GeometryMode, SceneDocument } from './contracts';
import { Button, Icon, Squircle } from './components';
import { useControlPopover } from './CustomControls';
import { readCameraView, type CameraViewState } from './scene-camera';
import { CameraPreviewRenderer } from './viewport/camera-preview-renderer';
import { t, useLanguage } from './i18n';
import './camera-preview.css';
export interface CameraPreviewProps {
  doc: SceneDocument;
  selectedId: string | null;
  enabled: boolean;
  revision: number;
  geometry: GeometryMode;
  visible?: boolean;
  onClose: () => void;
}
type FrozenView = { view: CameraViewState; name: string };
export function CameraPreview({
  doc,
  selectedId,
  enabled,
  revision,
  geometry,
  visible = true,
  onClose,
}: CameraPreviewProps) {
  useLanguage();
  const [pinned, setPinned] = useState<FrozenView | null>(null),
    [locked, setLocked] = useState<string | null>(null),
    [position, setPosition] = useState<[number, number] | null>(null),
    [error, setError] = useState(''),
    [budget, setBudget] = useState(340),
    [actionsOpen, setActionsOpen] = useState(false);
  const more = useRef<HTMLDivElement>(null),
    popup = useRef<HTMLDivElement>(null);
  useControlPopover(more, popup, actionsOpen, () => setActionsOpen(false));
  const surface = useRef<HTMLDivElement>(null),
    canvas = useRef<HTMLCanvasElement>(null),
    peer = useRef<CameraPreviewRenderer | null>(null),
    draw = useRef(() => {}),
    latest = useRef({ doc, geometry, view: null as CameraViewState | null });
  const drag = useRef<{
    id: number;
    x: number;
    y: number;
    start: [number, number];
    zoom: number;
  } | null>(null);
  const node = doc.nodes.find(
    (node) => node.id === (locked ?? selectedId) && node.kind === 'camera',
  );
  const view = pinned?.view ?? (node ? readCameraView(node) : null),
    name = pinned?.name ?? node?.name ?? '';
  const shown = Boolean(enabled && visible && view);
  latest.current = { doc, geometry, view };
  useEffect(() => {
    const panel = surface.current?.parentElement,
      parent = panel?.parentElement;
    if (!shown || !parent || !view) return;
    const resize = () => {
      const width = Math.min(
        340,
        parent.clientWidth - 24,
        ((parent.clientHeight - 76) * view.resolution[0]) / view.resolution[1] + 16,
      );
      setBudget(width);
      setPosition((previous) =>
        previous
          ? [
              Math.max(8, Math.min(parent.clientWidth - Math.max(96, width) - 8, previous[0])),
              Math.max(8, Math.min(parent.clientHeight - panel.offsetHeight - 8, previous[1])),
            ]
          : null,
      );
    };
    const observer = new ResizeObserver(resize);
    observer.observe(parent);
    observer.observe(panel);
    resize();
    return () => observer.disconnect();
  }, [shown, view?.resolution.join(',')]);
  useEffect(() => {
    setPinned(null);
    setLocked(null);
    setPosition(null);
    setActionsOpen(false);
  }, [doc.root.uuid]);
  useEffect(() => {
    if (!enabled) {
      setPinned(null);
      setLocked(null);
      setActionsOpen(false);
      drag.current = null;
    }
  }, [enabled]);
  useEffect(() => {
    if (locked && !doc.nodes.some((node) => node.id === locked && node.kind === 'camera'))
      setLocked(null);
  }, [doc, revision, locked]);
  useEffect(() => {
    if (!shown || !canvas.current) return;
    let queued = 0,
      intersecting = false,
      disposed = false;
    const element = canvas.current;
    const render = () => {
      queued = 0;
      if (disposed || !intersecting || document.hidden) return;
      const current = latest.current;
      if (!current.view) return;
      try {
        const rect = element.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;
        peer.current ??= new CameraPreviewRenderer(element, () => draw.current());
        peer.current.render(
          current.doc,
          current.view,
          current.geometry,
          rect.width * devicePixelRatio,
          rect.height * devicePixelRatio,
        );
        element.dataset.renderCount = String(Number(element.dataset.renderCount ?? 0) + 1);
        setError('');
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : String(reason));
      }
    };
    const invalidate = () => {
      if (!queued && !disposed && intersecting && !document.hidden)
        queued = requestAnimationFrame(render);
    };
    draw.current = invalidate;
    const observer = new IntersectionObserver((entries) => {
      intersecting = entries.some((entry) => entry.isIntersecting);
      invalidate();
    });
    observer.observe(element);
    const resize = new ResizeObserver(invalidate);
    resize.observe(element);
    document.addEventListener('visibilitychange', invalidate);
    return () => {
      disposed = true;
      cancelAnimationFrame(queued);
      observer.disconnect();
      resize.disconnect();
      document.removeEventListener('visibilitychange', invalidate);
      draw.current = () => {};
      peer.current?.dispose();
      peer.current = null;
    };
  }, [shown]);
  useEffect(
    () => draw.current(),
    [
      doc,
      revision,
      geometry,
      view?.projection,
      view?.position.join(','),
      view?.target.join(','),
      view?.up.join(','),
      view?.fov,
      view?.orthoHeight,
      view?.near,
      view?.far,
      view?.resolution.join(','),
      pinned,
      locked,
    ],
  );
  const endDrag = (cancel = false) => {
    const state = drag.current;
    if (!state) return;
    drag.current = null;
    if (cancel) setPosition(state.start);
  };
  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && drag.current) {
        event.preventDefault();
        event.stopPropagation();
        endDrag(true);
      }
    };
    const blur = () => endDrag(true);
    window.addEventListener('keydown', escape);
    window.addEventListener('blur', blur);
    return () => {
      window.removeEventListener('keydown', escape);
      window.removeEventListener('blur', blur);
    };
  }, []);
  const togglePin = () => {
    if (!view) return;
    if (!pinned) {
      const panel = surface.current!.parentElement!,
        parent = panel.parentElement!,
        rect = panel.getBoundingClientRect(),
        bounds = parent.getBoundingClientRect(),
        zoom = rect.width / panel.offsetWidth;
      setPosition([(rect.left - bounds.left) / zoom, (rect.top - bounds.top) / zoom]);
    }
    setPinned(pinned ? null : { view: structuredClone(view), name });
    setLocked(null);
    setActionsOpen(false);
  };
  const toggleLock = () => {
    setLocked(locked ? null : (node?.id ?? null));
    setPinned(null);
    setActionsOpen(false);
  };
  if (!shown || !view) return null;
  return (
    <Squircle
      radius={12}
      className="camera-preview"
      role="region"
      aria-label={t('Camera Preview')}
      data-pinned={!!pinned}
      data-locked={!!locked}
      style={{
        width: Math.max(96, budget),
        display: budget < 96 ? 'none' : undefined,
        ...(position ? { left: position[0], top: position[1], right: 'auto', bottom: 'auto' } : {}),
      }}
    >
      <div ref={surface} className="camera-preview-surface">
        <div className="camera-preview-header">
          <button
            className="camera-preview-title"
            data-tooltip={name}
            aria-label={name}
            disabled={!!pinned}
            onPointerDown={(event) => {
              if (event.button !== 0 || pinned) return;
              event.preventDefault();
              const panel = surface.current!.parentElement!,
                parent = panel.parentElement!,
                rect = panel.getBoundingClientRect(),
                bounds = parent.getBoundingClientRect(),
                zoom = rect.width / panel.offsetWidth;
              const start: [number, number] = [
                (rect.left - bounds.left) / zoom,
                (rect.top - bounds.top) / zoom,
              ];
              drag.current = {
                id: event.pointerId,
                x: event.clientX,
                y: event.clientY,
                start,
                zoom,
              };
              setPosition(start);
              event.currentTarget.setPointerCapture(event.pointerId);
            }}
            onPointerMove={(event) => {
              const state = drag.current;
              if (!state || state.id !== event.pointerId) return;
              const panel = surface.current!.parentElement!,
                parent = panel.parentElement!;
              setPosition([
                Math.max(
                  8,
                  Math.min(
                    parent.clientWidth - panel.offsetWidth - 8,
                    state.start[0] + (event.clientX - state.x) / state.zoom,
                  ),
                ),
                Math.max(
                  8,
                  Math.min(
                    parent.clientHeight - panel.offsetHeight - 8,
                    state.start[1] + (event.clientY - state.y) / state.zoom,
                  ),
                ),
              ]);
            }}
            onPointerUp={() => endDrag()}
            onPointerCancel={() => endDrag(true)}
            onLostPointerCapture={() => endDrag()}
          >
            <Icon name="camera" size={16} />
            <span>{name}</span>
          </button>
          {budget >= 220 ? (
            <>
              <Button
                icon="pin"
                title="Freeze the preview viewpoint and panel position. Scene motion continues."
                aria-label={t('Pin')}
                active={!!pinned}
                aria-pressed={!!pinned}
                onClick={togglePin}
              />
              <Button
                icon="lock"
                title="Follow this camera even when selection changes. The camera can still move."
                aria-label={t('Lock')}
                active={!!locked}
                aria-pressed={!!locked}
                disabled={!node && !locked}
                onClick={toggleLock}
              />
            </>
          ) : (
            <div ref={more}>
              <Button
                icon="more"
                title="Camera preview actions"
                aria-label={t('Camera preview actions')}
                onClick={() => setActionsOpen(!actionsOpen)}
              />
            </div>
          )}
          {actionsOpen && (
            <div
              ref={popup}
              popover="auto"
              className="camera-preview-menu"
              role="menu"
              aria-label={t('Camera preview actions')}
              onKeyDown={(event) => event.stopPropagation()}
            >
              <Button
                icon="pin"
                role="menuitemcheckbox"
                aria-checked={!!pinned}
                onClick={togglePin}
              >
                {t('Pin')}
              </Button>
              <Button
                icon="lock"
                role="menuitemcheckbox"
                aria-checked={!!locked}
                disabled={!node && !locked}
                onClick={toggleLock}
              >
                {t('Lock')}
              </Button>
            </div>
          )}
          <Button
            icon="close"
            title="Close Camera Preview"
            aria-label={t('Close Camera Preview')}
            onClick={onClose}
          />
        </div>
        <canvas
          ref={canvas}
          aria-label={`${t('Camera Preview')} · ${name}`}
          className="camera-preview-image"
          style={{ aspectRatio: `${view.resolution[0]} / ${view.resolution[1]}` }}
        />
        {error && (
          <div className="camera-preview-error" role="status">
            {t(error)}
          </div>
        )}
      </div>
    </Squircle>
  );
}
