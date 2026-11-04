import { t } from './i18n';
import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type KeyboardEvent as ReactKeyboardEvent,
} from 'react';
import type { SceneDocument } from './contracts';
import { assertPoseCompatible, type Keyframe, type WorkspaceTimeline } from './editor';
import { Button, Icon, Numeric, numericCaption, Select, Squircle } from './components';
import { useControlPopover } from './CustomControls';
import './timeline-controls.css';
import {
  clampView,
  fittedView,
  navigationLimits,
  normalizeWheel,
  panView,
  zoomView,
  type TimeRange,
} from './timelineNavigation';
interface Props {
  doc: SceneDocument;
  /** Changes on scene replacement, not Undo document reconstruction. */
  navigationResetKey?: number;
  keys: Keyframe[];
  timeline?: WorkspaceTimeline;
  time: number;
  /** Finite content extent for rendering/serialization, never a recording limit. */
  duration: number;
  playing: boolean;
  recording?: boolean;
  loop: boolean;
  playbackSource?: 'keyframes' | 'take';
  range?: [number, number] | null;
  onRange?: (range: [number, number] | null) => void;
  onSelectTake?: (id: string) => void;
  onCreateTake?: () => void;
  onDeleteTake?: (id: string) => void;
  onExportVideo?: () => void;
  videoActive?: boolean;
  speed?: number;
  onSpeed?: (speed: number) => void;
  setLoop: (v: boolean) => void;
  setTime: (v: number, source?: 'keyframes' | 'take') => void;
  setDuration: (v: number) => void;
  setPlaying: (v: boolean) => void;
  onCapture: () => void;
  onKeys: (keys: Keyframe[]) => void;
  onTimeline?: (data: WorkspaceTimeline) => void;
  onRecord?: () => void;
  onSnapshot?: () => void;
  onLoadSnapshot?: (id: string) => void;
  onLoadKey?: (id: string) => void;
  onBegin: () => void;
  onCommit: () => void;
  onCancel: () => void;
}
type Lane = 'keys' | 'take' | 'snapshots';
type Follow = 'off' | 'page' | 'locked';
type Gesture = {
  kind: 'key' | 'select' | 'scrub' | 'range' | 'pan';
  pointer: number;
  target: HTMLElement;
  startX: number;
  lastX: number;
  anchor: number;
  moved: boolean;
  resume: boolean;
  keys?: Keyframe[];
  previewKeys?: Keyframe[];
  selected?: Set<string>;
  range?: [number, number] | null;
  lane: Lane;
};
const EMPTY_TIMELINE: WorkspaceTimeline = { snapshots: [], takes: [], activeTakeId: null };
const nearest = (frames: readonly { time: number }[], time: number) => {
  let lo = 0,
    hi = frames.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (frames[mid].time < time) lo = mid + 1;
    else hi = mid;
  }
  return lo === 0
    ? 0
    : lo === frames.length
      ? lo - 1
      : time - frames[lo - 1].time <= frames[lo].time - time
        ? lo - 1
        : lo;
};
const orderedKeys = (keys: Keyframe[]) => [...keys].sort((a, b) => a.time - b.time);
const clipboardPrefix = 'mojive-keyframes/v1\n';
/** Fixed idle-caption budget; full-precision input is owned by Numeric. */
export function timelineTimeCaption(value: number) {
  for (let precision = 3; precision >= 0; precision--) {
    const text = numericCaption(value, precision);
    if (text.length <= 8) return text;
  }
  for (let precision = 3; precision >= 0; precision--) {
    const text = value.toExponential(precision);
    if (text.length <= 8) {
      if (Number.isFinite(Number(text))) return text;
      // Avoid rounding the largest finite value into an overflowing decimal caption.
      const [mantissa, exponent] = value.toExponential().split('e');
      const factor = 10 ** precision;
      return `${(Math.trunc(Number(mantissa) * factor) / factor).toFixed(precision)}e${exponent}`;
    }
  }
  return value.toExponential(0);
}

export function Timeline(props: Props) {
  const {
    doc,
    navigationResetKey,
    keys,
    timeline = EMPTY_TIMELINE,
    time,
    duration,
    playing,
    recording = false,
    loop,
    range = null,
    onRange,
    onSelectTake,
    onCreateTake,
    onDeleteTake,
    onExportVideo,
    videoActive = false,
    speed = 1,
    onSpeed,
    setLoop,
    setTime,
    setDuration,
    setPlaying,
    onCapture,
    onKeys,
    onTimeline,
    onRecord,
    onSnapshot,
    onLoadSnapshot,
    onLoadKey,
    onBegin,
    onCommit,
    onCancel,
  } = props;
  const busy = recording || videoActive;
  const root = useRef<HTMLDivElement>(null),
    body = useRef<HTMLDivElement>(null),
    track = useRef<HTMLDivElement>(null);
  const latest = useRef(props);
  latest.current = props;
  const [navigation, setNavigation] = useState({ start: 0, pixelsPerSecond: 88 });
  const navigationRef = useRef(navigation);
  const manualRange = useRef<TimeRange | null>(null);
  const scrollPosition = useRef(0);
  const [visible, setVisible] = useState({ width: 800 });
  const [lane, setLane] = useState<Lane>('keys'),
    [selection, setSelection] = useState(new Set<string>());
  const [primary, setPrimary] = useState<string | null>(null),
    [snapshotId, setSnapshotId] = useState<string | null>(null);
  const [takeSelection, setTakeSelection] = useState<[number, number] | null>(null);
  const [follow, setFollow] = useState<Follow>('page'),
    lockedFraction = useRef(0.15);
  const [draftName, setDraftName] = useState(''),
    [draftTime, setDraftTime] = useState(0),
    [error, setError] = useState('');
  const [keyPreview, setKeyPreview] = useState<Map<string, number> | null>(null);
  const [preview, setPreview] = useState<{
    lane: Lane;
    start: number;
    end: number;
    range: boolean;
  } | null>(null);
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null),
    menuRef = useRef<HTMLDivElement>(null),
    menuAnchor = useRef<HTMLDivElement>(null);
  useControlPopover(menuAnchor, menuRef, !!menu, () => setMenu(null));
  useLayoutEffect(() => {
    if (!videoActive) return;
    const controls = root.current?.querySelector('.timeline-primary-controls');
    const stop = controls?.querySelector('.timeline-video-button');
    if (!controls || !stop) return;
    const viewport = controls.getBoundingClientRect(),
      button = stop.getBoundingClientRect();
    const scale = viewport.width / (controls as HTMLElement).offsetWidth;
    if (button.left < viewport.left) controls.scrollLeft += (button.left - viewport.left) / scale;
    else if (button.right > viewport.right)
      controls.scrollLeft += (button.right - viewport.right) / scale;
  }, [videoActive]);
  const drag = useRef<Gesture | null>(null),
    clipboard = useRef<Keyframe[]>([]);
  const activeTake = timeline.takes.find((item) => item.id === timeline.activeTakeId),
    frames = activeTake?.frames ?? [];
  const sorted = useMemo(() => orderedKeys(keys), [keys]);
  const selectedKey =
    lane === 'keys' && selection.size === 1 ? keys.find((key) => selection.has(key.id)) : undefined;
  const selectedSnapshot =
    lane === 'snapshots' ? timeline.snapshots.find((item) => item.id === snapshotId) : undefined;
  const bounds =
    range && range[0] >= 0 && range[1] < frames.length && range[0] < range[1]
      ? range
      : [0, Math.max(0, frames.length - 1)];
  const first = frames[bounds[0]]?.time ?? 0,
    last = frames[bounds[1]]?.time ?? 0;
  const firstIndex = frames.length ? nearest(frames, time) : 0;
  const previousIndex = Math.min(
    bounds[1],
    Math.max(bounds[0], firstIndex - (frames[firstIndex]?.time >= time - 1e-9 ? 1 : 0)),
  );
  const nextIndex = Math.max(
    bounds[0],
    Math.min(bounds[1], firstIndex + (frames[firstIndex]?.time <= time + 1e-9 ? 1 : 0)),
  );
  const inset = 12,
    width = Math.max(1, visible.width - inset * 2),
    pixelsPerSecond = navigation.pixelsPerSecond,
    zoom = pixelsPerSecond / 88,
    viewStart = navigation.start,
    viewEnd = viewStart + width / pixelsPerSecond,
    x = (value: number) => inset + (value - viewStart) * pixelsPerSecond;
  // Only content and explicit range edits establish navigation context. Never use a
  // previous zoom window (or the serialized duration) as fresh content evidence.
  const navigationTimes = [0, time];
  if (sorted.length) navigationTimes.push(sorted[0].time, sorted.at(-1)!.time);
  for (const take of timeline.takes)
    if (take.frames.length) navigationTimes.push(take.frames[0].time, take.frames.at(-1)!.time);
  for (const snapshot of timeline.snapshots) navigationTimes.push(snapshot.time);
  const limits = navigationLimits(navigationTimes, manualRange.current);
  const currentView = (): TimeRange => ({
    start: navigationRef.current.start,
    end: navigationRef.current.start + width / navigationRef.current.pixelsPerSecond,
  });
  const updateView = (range: TimeRange) => {
    const next = { start: range.start, pixelsPerSecond: width / (range.end - range.start) };
    if (
      !Number.isFinite(next.start) ||
      !Number.isFinite(next.pixelsPerSecond) ||
      next.pixelsPerSecond <= 0
    )
      return;
    navigationRef.current = next;
    setNavigation(next);
  };
  const pointerRatio = (clientX: number) => {
    const rect = track.current!.getBoundingClientRect(),
      scale = rect.width / visible.width;
    return Math.max(0, Math.min(1, ((clientX - rect.left) / scale - inset) / width));
  };
  const position = (clientX: number) => {
    const range = currentView();
    return Math.max(0, range.start + pointerRatio(clientX) * (range.end - range.start));
  };
  // The scrollbar is a bounded navigation proxy. Rendering uses the exact signed
  // range, avoiding CSS size limits and fractional scrollLeft rounding at high zoom.
  const scrollSpan = Math.max(0, limits.upper - limits.lower - (viewEnd - viewStart));
  const scrollTravel = Math.min(1_000_000, scrollSpan * pixelsPerSecond);
  const scrollWidth = visible.width + scrollTravel;
  const extendExtent = (value: number) => {
    if (Number.isFinite(value) && value > duration) setDuration(Math.ceil(value / 10) * 10);
  };
  const changeKeys = (next: Keyframe[]) => {
    const ordered = orderedKeys(next);
    extendExtent(ordered.at(-1)?.time ?? 0);
    onKeys(ordered);
  };
  const unitScale = () => {
    const rect = track.current!.getBoundingClientRect();
    return rect.width / visible.width;
  };
  const edit = (fn: () => void) => {
    onBegin();
    fn();
    onCommit();
  };
  const seek = (value: number, source: 'keyframes' | 'take' = 'take') => {
    if (busy || !Number.isFinite(value)) return;
    extendExtent(value);
    setPlaying(false);
    setTime(value, source);
  };
  const focusTimeline = () => root.current?.focus({ preventScroll: true });
  const selectTake = (id: string) => {
    if (busy) return;
    setPlaying(false);
    setLane('take');
    setTakeSelection(null);
    setSelection(new Set());
    setSnapshotId(null);
    if (onSelectTake) onSelectTake(id);
    else {
      onTimeline?.({ ...timeline, activeTakeId: id });
      setTime(timeline.takes.find((item) => item.id === id)?.frames[0]?.time ?? 0, 'take');
    }
  };
  const applyView = (start: number, end: number, manual = false) => {
    if (
      !Number.isFinite(start) ||
      !Number.isFinite(end) ||
      !Number.isFinite(end - start) ||
      end - start < 0.001
    )
      return;
    if (manual) manualRange.current = { start, end };
    updateView(clampView({ start, end }, navigationLimits(navigationTimes, manualRange.current)));
  };
  const fittedContentView = () => {
    const times = [time, ...timeline.snapshots.map((item) => item.time)];
    if (sorted.length) times.push(sorted[0].time, sorted.at(-1)!.time);
    if (frames.length) times.push(frames[0].time, frames.at(-1)!.time);
    return clampView(fittedView(times, time), navigationLimits(navigationTimes));
  };
  const fit = () => {
    manualRange.current = null;
    updateView(fittedContentView());
  };
  const setFollowMode = (next: Follow) => {
    if (next === 'locked')
      lockedFraction.current = Math.min(
        0.95,
        Math.max(0.05, (time - viewStart) / (viewEnd - viewStart)),
      );
    setFollow(next);
  };
  const setRange = (value: [number, number] | null, enable = false) => {
    if (busy) return;
    onRange?.(value);
    if (enable && value) setLoop(true);
  };
  const rangeFromSelection = () => {
    if (takeSelection && takeSelection[0] < takeSelection[1]) setRange(takeSelection, true);
  };
  const clearTake = () => {
    if (!activeTake || busy) return;
    edit(() => {
      onRange?.(null);
      onTimeline?.({
        ...timeline,
        takes: timeline.takes.map((item) =>
          item.id === activeTake.id ? { ...item, frames: [] } : item,
        ),
      });
    });
    setTakeSelection(null);
  };
  const deleteTake = () => {
    if (!activeTake || busy) return;
    if (onDeleteTake) onDeleteTake(activeTake.id);
    else
      edit(() => {
        const takes = timeline.takes.filter((item) => item.id !== activeTake.id);
        onTimeline?.({ ...timeline, takes, activeTakeId: takes[0]?.id ?? null });
      });
    setTakeSelection(null);
  };
  const remove = () => {
    if (busy) return;
    if (lane === 'keys' && selection.size) {
      edit(() => onKeys(keys.filter((key) => !selection.has(key.id))));
      setSelection(new Set());
      setPrimary(null);
    } else if (lane === 'snapshots' && snapshotId) {
      edit(() =>
        onTimeline?.({
          ...timeline,
          snapshots: timeline.snapshots.filter((item) => item.id !== snapshotId),
        }),
      );
      setSnapshotId(null);
    } else if (lane === 'take' && activeTake) {
      if (takeSelection) {
        const [a, b] = takeSelection;
        edit(() => {
          onRange?.(null);
          onTimeline?.({
            ...timeline,
            takes: timeline.takes.map((item) =>
              item.id === activeTake.id
                ? { ...item, frames: item.frames.filter((_, i) => i < a || i > b) }
                : item,
            ),
          });
        });
      }
      setTakeSelection(null);
    }
  };
  const selectAll = () => {
    if (lane === 'keys') {
      setSelection(new Set(keys.map((key) => key.id)));
      setPrimary(keys.length === 1 ? keys[0].id : null);
    } else if (lane === 'take' && frames.length) setTakeSelection([0, frames.length - 1]);
  };
  const pasteKeys = (items: Keyframe[], at = time) => {
    if (busy || !items.length) return;
    try {
      for (const item of items) {
        if (!Number.isFinite(item.time) || item.time < 0) throw Error('Invalid keyframe clipboard');
        assertPoseCompatible(doc, item.pose);
      }
      const origin = Math.min(...items.map((item) => item.time)),
        end = at + Math.max(...items.map((item) => item.time)) - origin;
      if (!Number.isFinite(end)) throw Error('Invalid keyframe clipboard');
      const copies = items.map((item) => ({
        ...structuredClone(item),
        id: crypto.randomUUID(),
        time: at + item.time - origin,
      }));
      edit(() => changeKeys([...keys, ...copies]));
      setSelection(new Set(copies.map((item) => item.id)));
      setPrimary(copies[0].id);
      setLane('keys');
      setError('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  };
  const copyKeys = async () => {
    const items = keys.filter((key) => selection.has(key.id));
    if (!items.length) return;
    clipboard.current = structuredClone(items);
    const text = clipboardPrefix + JSON.stringify(items);
    try {
      if (window.mojive) await window.mojive.writeClipboardText(text);
      else await navigator.clipboard.writeText(text);
      setError('');
    } catch {
      setError('System clipboard unavailable; keys copied within this timeline');
    }
  };
  const paste = async () => {
    if (busy) return;
    try {
      let text = '';
      try {
        text = window.mojive
          ? await window.mojive.readClipboardText()
          : await navigator.clipboard.readText();
      } catch {
        if (clipboard.current.length) {
          pasteKeys(clipboard.current);
          return;
        }
        throw Error('Clipboard is unavailable');
      }
      if (!text.startsWith(clipboardPrefix))
        throw Error('Clipboard does not contain Mojive keyframes');
      const items = JSON.parse(text.slice(clipboardPrefix.length));
      if (!Array.isArray(items) || items.length > 10000) throw Error('Invalid keyframe clipboard');
      pasteKeys(items);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  };
  const duplicate = () =>
    pasteKeys(
      keys.filter((key) => selection.has(key.id)),
      time + 1 / 60,
    );
  const neighborKey = (direction: number) => {
    const slot = sorted.findIndex((key) => key.id === primary);
    const candidate =
      slot >= 0
        ? sorted[Math.max(0, Math.min(sorted.length - 1, slot + direction))]
        : direction > 0
          ? (sorted.find((key) => key.time > time) ?? sorted.at(-1))
          : ([...sorted].reverse().find((key) => key.time < time) ?? sorted[0]);
    if (!candidate) return;
    setSelection(new Set([candidate.id]));
    setPrimary(candidate.id);
    setLane('keys');
    setPlaying(false);
    onLoadKey?.(candidate.id);
  };
  const play = () => {
    if (busy || frames.length < 2) return;
    if (!playing) setTime(time >= last || time < first ? first : time, 'take');
    setPlaying(!playing);
  };
  const cancelGesture = () => {
    const gesture = drag.current;
    if (!gesture) return;
    drag.current = null;
    setPreview(null);
    setKeyPreview(null);
    if (gesture.target.hasPointerCapture(gesture.pointer))
      gesture.target.releasePointerCapture(gesture.pointer);
    if (gesture.kind === 'key') latest.current.onCancel();
    if (gesture.kind === 'range') latest.current.onRange?.(gesture.range ?? null);
  };
  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && drag.current) {
        event.preventDefault();
        event.stopImmediatePropagation();
        cancelGesture();
      }
    };
    const blur = () => cancelGesture();
    window.addEventListener('keydown', escape, true);
    window.addEventListener('blur', blur);
    return () => {
      window.removeEventListener('keydown', escape, true);
      window.removeEventListener('blur', blur);
    };
  }, []);
  useEffect(() => {
    setDraftName(selectedKey?.name ?? '');
    setDraftTime(selectedKey?.time ?? 0);
  }, [selectedKey?.id, selectedKey?.name, selectedKey?.time]);
  useEffect(() => {
    setTakeSelection(null);
    setPreview(null);
  }, [activeTake?.id, frames.length]);
  useEffect(() => {
    setSelection((old) => new Set([...old].filter((id) => keys.some((key) => key.id === id))));
  }, [keys]);
  useEffect(() => {
    setLane('keys');
    setSelection(new Set());
    setPrimary(null);
    setSnapshotId(null);
    setMenu(null);
    setPreview(null);
  }, [doc]);
  useEffect(() => {
    const element = body.current!;
    const update = () => setVisible({ width: Math.max(100, element.clientWidth - 148) });
    const observer = new ResizeObserver(update);
    observer.observe(element);
    update();
    return () => observer.disconnect();
  }, []);
  useLayoutEffect(() => {
    if (navigationResetKey === undefined) return;
    manualRange.current = null;
    // A reopened scene starts with its authored content in view. Read the actual
    // width here because ResizeObserver has not measured a newly mounted panel yet.
    const measuredWidth = Math.max(100, body.current!.clientWidth - 148);
    const fitted =
      sorted.length || frames.length || timeline.snapshots.length ? fittedContentView() : null;
    const next = fitted
      ? {
          start: fitted.start,
          pixelsPerSecond: (measuredWidth - inset * 2) / (fitted.end - fitted.start),
        }
      : { start: 0, pixelsPerSecond: 88 };
    setVisible({ width: measuredWidth });
    navigationRef.current = next;
    setNavigation(next);
  }, [navigationResetKey]);
  useLayoutEffect(() => {
    if (!body.current) return;
    body.current.scrollLeft =
      scrollSpan > 0 ? ((viewStart - limits.lower) / scrollSpan) * scrollTravel : 0;
    // Read back browser rounding; its resulting scroll event must not feed back
    // into the floating-point view used by the wheel gesture.
    scrollPosition.current = body.current.scrollLeft;
  }, [viewStart, scrollSpan, scrollTravel, visible.width]);
  useEffect(() => {
    if (follow === 'off' || drag.current || (!playing && !recording)) return;
    const range = currentView(),
      span = range.end - range.start;
    const start =
      follow === 'locked'
        ? time - span * lockedFraction.current
        : time > range.end - span * 0.01 || time < range.start + span * 0.01
          ? time - span * 0.08
          : null;
    if (start !== null && Math.abs(start - range.start) > span * 1e-10)
      updateView({ start, end: start + span });
  }, [playing, recording, time, follow, visible.width, pixelsPerSecond]);
  useEffect(() => {
    if (!menu) return;
    const outside = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setMenu(null);
    };
    document.addEventListener('pointerdown', outside, true);
    return () => document.removeEventListener('pointerdown', outside, true);
  }, [menu]);
  useEffect(() => {
    const element = track.current!;
    const wheel = (event: WheelEvent) => {
      if ((event.target as Element).closest('input,[role=combobox]')) return;
      event.preventDefault();
      if (drag.current) return;
      const range = currentView(),
        span = range.end - range.start;
      if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
        setFollow('off');
        const delta = event.deltaX || event.deltaY;
        const pixels =
          event.deltaMode === 1 ? delta * (100 / 3) : event.deltaMode === 2 ? delta * width : delta;
        updateView(
          panView(range, pixels / unitScale() / navigationRef.current.pixelsPerSecond, limits),
        );
        return;
      }
      // Locked follow deliberately anchors the playhead; other modes anchor the cursor.
      const ratio = follow === 'locked' ? (time - range.start) / span : pointerRatio(event.clientX);
      updateView(zoomView(range, ratio, -normalizeWheel(event.deltaY, event.deltaMode), limits));
    };
    element.addEventListener('wheel', wheel, { passive: false });
    return () => element.removeEventListener('wheel', wheel);
  }, [visible.width, follow, time, keys, timeline, navigation]);
  const openMenu = (clientX: number, clientY: number) => {
    const rect = root.current!.getBoundingClientRect(),
      scale = rect.width / root.current!.offsetWidth;
    setMenu({
      x: Math.max(4, Math.min((clientX - rect.left) / scale, root.current!.offsetWidth - 220)),
      y: (clientY - rect.top) / scale,
    });
    focusTimeline();
  };
  const startGesture = (
    event: ReactPointerEvent<HTMLElement>,
    kind: Gesture['kind'],
    selectedLane: Lane,
    key?: Keyframe,
  ) => {
    if (busy || drag.current) return;
    event.preventDefault();
    event.stopPropagation();
    setMenu(null);
    focusTimeline();
    if (event.button === 2 && event.shiftKey) {
      setRange(null);
      return;
    }
    if (event.button === 1 || event.button === 2) kind = 'pan';
    else if (event.button !== 0) return;
    if (event.shiftKey && selectedLane === 'take' && frames.length > 1) kind = 'range';
    const anchor = position(event.clientX),
      base = event.ctrlKey || event.metaKey ? new Set(selection) : new Set<string>();
    if (key && kind === 'key') {
      if (event.ctrlKey || event.metaKey) {
        const next = new Set(selection);
        if (next.has(key.id)) next.delete(key.id);
        else next.add(key.id);
        setSelection(next);
        setPrimary(next.has(key.id) ? key.id : null);
        setLane('keys');
        return;
      }
      if (event.shiftKey && primary) {
        const reference = keys.find((item) => item.id === primary)?.time ?? key.time;
        setSelection(
          new Set(
            keys
              .filter(
                (item) =>
                  item.time >= Math.min(reference, key.time) &&
                  item.time <= Math.max(reference, key.time),
              )
              .map((item) => item.id),
          ),
        );
        setPrimary(key.id);
        setLane('keys');
        return;
      }
      const selected = selection.has(key.id) ? new Set(selection) : new Set([key.id]);
      setSelection(selected);
      setPrimary(key.id);
      setLane('keys');
      setSnapshotId(null);
      seek(key.time, 'keyframes');
      onBegin();
      drag.current = {
        kind,
        pointer: event.pointerId,
        target: event.currentTarget,
        startX: event.clientX,
        lastX: event.clientX,
        anchor,
        moved: false,
        resume: false,
        keys: structuredClone(keys),
        selected,
        lane: 'keys',
      };
    } else {
      setLane(selectedLane);
      setSnapshotId(null);
      if (kind === 'select') {
        if (selectedLane === 'keys') {
          setSelection(base);
          setPrimary(null);
        } else if (selectedLane === 'take') {
          const index = nearest(frames, anchor);
          setTakeSelection(frames.length ? [index, index] : null);
          seek(anchor);
        }
      }
      if (kind === 'scrub') seek(anchor);
      drag.current = {
        kind,
        pointer: event.pointerId,
        target: event.currentTarget,
        startX: event.clientX,
        lastX: event.clientX,
        anchor,
        moved: false,
        resume: kind === 'scrub' && playing,
        selected: base,
        range,
        lane: selectedLane,
      };
    }
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const moveGesture = (event: ReactPointerEvent<HTMLElement>) => {
    const gesture = drag.current;
    if (!gesture || gesture.pointer !== event.pointerId) return;
    event.stopPropagation();
    const value = position(event.clientX),
      distance = event.clientX - gesture.startX;
    gesture.moved ||= Math.abs(distance) > 3 * unitScale();
    if (gesture.kind === 'pan') {
      if (gesture.moved) {
        setFollow('off');
        updateView(
          panView(
            currentView(),
            -(event.clientX - gesture.lastX) / unitScale() / navigationRef.current.pixelsPerSecond,
            limits,
          ),
        );
      }
    } else if (gesture.kind === 'key' && gesture.moved) {
      const originals = gesture.keys!,
        selected = gesture.selected!;
      const times = originals.filter((key) => selected.has(key.id)).map((key) => key.time);
      let delta = value - gesture.anchor;
      delta = Math.round(delta * (event.shiftKey ? 10 : 1000)) / (event.shiftKey ? 10 : 1000);
      delta = Math.max(-Math.min(...times), delta);
      const next = originals.map((key) =>
        selected.has(key.id) ? { ...key, time: key.time + delta } : key,
      );
      gesture.previewKeys = next;
      setKeyPreview(
        new Map(next.filter((key) => selected.has(key.id)).map((key) => [key.id, key.time])),
      );
      const selectedTime = next.find((key) => key.id === primary)?.time;
      if (selectedTime !== undefined) setTime(selectedTime, 'keyframes');
    } else if (gesture.kind === 'scrub') seek(value);
    else if (gesture.kind === 'range')
      setPreview({
        lane: 'take',
        start: Math.min(value, gesture.anchor),
        end: Math.max(value, gesture.anchor),
        range: true,
      });
    else if (gesture.kind === 'select') {
      const lo = Math.min(value, gesture.anchor),
        hi = Math.max(value, gesture.anchor);
      setPreview({ lane: gesture.lane, start: lo, end: hi, range: false });
      if (gesture.lane === 'keys')
        setSelection(
          new Set([
            ...gesture.selected!,
            ...keys.filter((key) => key.time >= lo && key.time <= hi).map((key) => key.id),
          ]),
        );
      else if (gesture.lane === 'take' && frames.length)
        setTakeSelection([nearest(frames, lo), nearest(frames, hi)]);
    }
    gesture.lastX = event.clientX;
  };
  const endGesture = (event: ReactPointerEvent<HTMLElement>) => {
    const gesture = drag.current;
    if (!gesture || gesture.pointer !== event.pointerId) return;
    event.preventDefault();
    event.stopPropagation();
    drag.current = null;
    if (gesture.target.hasPointerCapture(gesture.pointer))
      gesture.target.releasePointerCapture(gesture.pointer);
    if (gesture.kind === 'key') {
      if (gesture.moved && gesture.previewKeys) changeKeys(gesture.previewKeys);
      setKeyPreview(null);
      onCommit();
    } else if (gesture.kind === 'range' && frames.length > 1) {
      const a = nearest(frames, gesture.anchor),
        b = nearest(frames, position(event.clientX));
      setRange(a === b ? null : [Math.min(a, b), Math.max(a, b)], true);
    } else if (gesture.kind === 'pan' && !gesture.moved) openMenu(event.clientX, event.clientY);
    if (gesture.resume) setPlaying(true);
    setPreview(null);
  };
  const pointerHandlers = {
    onPointerMove: moveGesture,
    onPointerUp: endGesture,
    onPointerCancel: cancelGesture,
    onLostPointerCapture: () => {
      if (drag.current) cancelGesture();
    },
  };
  const keyboard = (event: ReactKeyboardEvent) => {
    if ((event.target as HTMLElement).closest('input,textarea,[role=combobox],[role=menu]')) return;
    const command = event.ctrlKey || event.metaKey,
      key = event.key.toLowerCase();
    let handled = true;
    if (key === 'escape') {
      if (menu) setMenu(null);
      else if (range) setRange(null);
      else {
        setSelection(new Set());
        setTakeSelection(null);
        setSnapshotId(null);
      }
    } else if (command && key === 'a') selectAll();
    else if (command && key === 'c' && lane === 'keys') void copyKeys();
    else if (command && key === 'v') void paste();
    else if (command && key === 'd' && lane === 'keys') duplicate();
    else if (key === 'delete' || key === 'backspace') remove();
    else if (key === ' ') {
      play();
    } else if (key === 'home') seek(first);
    else if (key === 'end') seek(last);
    else if (key === 'f') fit();
    else if (key === 'arrowleft' || key === 'arrowright') {
      const direction = key === 'arrowleft' ? -1 : 1;
      if (lane === 'keys' && selection.size && !busy) {
        const selected = keys.filter((key) => selection.has(key.id)),
          step = event.altKey ? 0.001 : event.shiftKey ? 0.1 : 1 / 60,
          delta = Math.max(-Math.min(...selected.map((key) => key.time)), direction * step);
        edit(() =>
          changeKeys(
            keys.map((key) => (selection.has(key.id) ? { ...key, time: key.time + delta } : key)),
          ),
        );
      } else if (frames.length) seek(frames[direction < 0 ? previousIndex : nextIndex].time);
    } else handled = false;
    if (handled) {
      event.preventDefault();
      event.stopPropagation();
    }
  };
  const applyDetails = () => {
    if (!selectedKey || !draftName.trim() || !Number.isFinite(draftTime) || draftTime < 0 || busy)
      return;
    edit(() =>
      changeKeys(
        keys.map((key) =>
          key.id === selectedKey.id ? { ...key, name: draftName.trim(), time: draftTime } : key,
        ),
      ),
    );
  };
  const step = (() => {
    const raw = Math.max(0.001, ((viewEnd - viewStart) * 90) / visible.width),
      power = 10 ** Math.floor(Math.log10(raw)),
      f = raw / power;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * power;
  })();
  const firstTick = Math.floor(viewStart / step),
    lastTick = Math.ceil(viewEnd / step);
  const visibleKeys = useMemo(() => {
    const columns = new Map<number, Keyframe[]>();
    for (const key of sorted) {
      if (key.time < viewStart - 12 / pixelsPerSecond || key.time > viewEnd + 12 / pixelsPerSecond)
        continue;
      const column = Math.floor(x(key.time) / 4),
        items = columns.get(column) ?? [];
      if (items.length < 3 || key.id === primary) items.push(key);
      columns.set(column, items);
    }
    return [...columns.values()].flat();
  }, [sorted, viewStart, viewEnd, pixelsPerSecond, primary]);
  const markerLayout = useMemo(() => {
    const groups = new Map<number, Keyframe[]>();
    for (const key of visibleKeys) {
      const items = groups.get(key.time) ?? [];
      items.push(key);
      groups.set(key.time, items);
    }
    return new Map(
      visibleKeys.map((key, i) => {
        const group = groups.get(key.time)!,
          gap = Math.min(
            i ? key.time - visibleKeys[i - 1].time : Infinity,
            i + 1 < visibleKeys.length ? visibleKeys[i + 1].time - key.time : Infinity,
          );
        return [
          key.id,
          {
            width: group.length > 1 ? 14 : Math.min(20, Math.max(4, gap * pixelsPerSecond)),
            top: group.length > 1 ? 2 + (group.indexOf(key) % 3) * 12 : 12,
          },
        ];
      }),
    );
  }, [visibleKeys, pixelsPerSecond]);
  const frameColumns = useMemo(() => {
    const columns = new Set<number>();
    // Takes are chronological; long recordings scan only samples in the visible interval.
    let lo = 0,
      hi = frames.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (frames[mid].time < viewStart) lo = mid + 1;
      else hi = mid;
    }
    for (let i = lo; i < frames.length && frames[i].time <= viewEnd; i++)
      columns.add(Math.round(x(frames[i].time) / 2) * 2);
    return columns;
  }, [frames, frames.length, viewStart, viewEnd, pixelsPerSecond]);
  const rangeBand = preview?.range
    ? preview
    : range
      ? { start: frames[bounds[0]]?.time ?? 0, end: frames[bounds[1]]?.time ?? 0 }
      : null;
  const contextItem = (
    label: string,
    action: () => void,
    disabled = false,
    icon?: string,
    shortcut?: string,
  ) => (
    <Button
      role="menuitem"
      key={label}
      icon={icon}
      disabled={disabled}
      onClick={() => {
        setMenu(null);
        action();
        focusTimeline();
      }}
    >
      {t(label)}
      {shortcut && <kbd>{shortcut}</kbd>}
    </Button>
  );
  const firstButton = (
    <Button icon="first" title="Go to start" disabled={busy} onClick={() => seek(first)} />
  );
  const lastButton = (
    <Button
      icon="last"
      title="Go to end"
      disabled={busy || !frames.length}
      onClick={() => seek(last)}
    />
  );
  const loopButton = (
    <Button
      icon="loop"
      active={loop}
      title="Loop playback"
      aria-pressed={loop}
      disabled={busy || frames.length < 2}
      onClick={() => setLoop(!loop)}
    />
  );
  const previousKeyButton = (
    <Button
      icon="previous"
      title="Previous keyframe"
      disabled={busy || !keys.length}
      onClick={() => neighborKey(-1)}
    />
  );
  const nextKeyButton = (
    <Button
      icon="next"
      title="Next keyframe"
      disabled={busy || !keys.length}
      onClick={() => neighborKey(1)}
    />
  );
  const snapshotButton = (
    <Button icon="snapshot" title="Capture snapshot" disabled={busy} onClick={onSnapshot} />
  );
  const fitButton = <Button icon="fit" title="Fit timeline" onClick={fit} />;
  const toolbarKey = (event: ReactKeyboardEvent) => {
    const globalCommand =
      (event.metaKey || event.ctrlKey) && ['z', 's', 'o', ','].includes(event.key.toLowerCase());
    if (!globalCommand) event.stopPropagation();
  };
  const speedControl = onSpeed && (
    <div className="timeline-speed-group timeline-control-group" title={t('Playback speed')}>
      <div className="timeline-speed">
        <Numeric
          value={speed}
          min={0.1}
          max={4}
          step={0.1}
          precision={2}
          onChange={onSpeed}
          label={t('Playback speed')}
          unit="×"
          disabled={busy}
        />
      </div>
    </div>
  );
  const zoomControl = (
    <div className="timeline-view-actions timeline-control-group">
      <div className="timeline-zoom-label" title={t('Timeline zoom')}>
        <div className="timeline-zoom">
          <Numeric
            label={t('Timeline zoom')}
            value={zoom * 100}
            formatValue={(value) => numericCaption(value, value < 0.1 ? 4 : 1)}
            min={(width / limits.maxSpan / 88) * 100}
            max={(width / limits.minSpan / 88) * 100}
            precision={1}
            step={10}
            unit="%"
            onChange={(value) => {
              const range = currentView();
              const ratio =
                follow === 'locked'
                  ? Math.max(0, Math.min(1, (time - range.start) / (range.end - range.start)))
                  : 0.5;
              const span = width / ((88 * value) / 100),
                anchor = range.start + ratio * (range.end - range.start);
              applyView(anchor - ratio * span, anchor + (1 - ratio) * span);
            }}
          />
        </div>
      </div>
      {fitButton}
    </div>
  );
  const rangeControls = (
    <div
      className="timeline-range-fields timeline-control-group"
      role="group"
      aria-label={t(frames.length > 1 ? 'Playback range' : 'View')}
      title={t(frames.length > 1 ? 'Playback range' : 'View')}
    >
      <span className="timeline-range-scope" aria-hidden="true">
        <span>{t(frames.length > 1 ? 'Loop' : 'View')}</span>
        <Icon name={frames.length > 1 ? 'loop' : 'frame'} />
      </span>
      <Numeric
        label={t(frames.length > 1 ? 'Loop start' : 'View start')}
        value={frames.length > 1 ? first : viewStart}
        formatValue={timelineTimeCaption}
        min={frames.length > 1 ? 0 : -Infinity}
        max={frames.length > 1 ? last : viewEnd - 0.001}
        disabled={busy}
        onChange={(value) => {
          if (frames.length > 1) {
            const index = nearest(frames, value);
            if (index < bounds[1]) setRange([index, bounds[1]]);
          } else applyView(value, viewEnd, true);
        }}
        unit="s"
      />
      <span className="timeline-range-separator" aria-hidden="true">
        –
      </span>
      <Numeric
        label={t(frames.length > 1 ? 'Loop end' : 'View end')}
        value={frames.length > 1 ? last : viewEnd}
        formatValue={timelineTimeCaption}
        min={frames.length > 1 ? first : viewStart + 0.001}
        max={frames.length > 1 ? frames.at(-1)!.time : Infinity}
        disabled={busy}
        onChange={(value) => {
          if (frames.length > 1) {
            const index = nearest(frames, value);
            if (index > bounds[0]) setRange([bounds[0], index]);
          } else applyView(viewStart, value, true);
        }}
        unit="s"
      />
    </div>
  );
  return (
    <div ref={root} className="timeline" data-testid="timeline" tabIndex={0} onKeyDown={keyboard}>
      <div
        className="timeline-toolbar"
        role="toolbar"
        aria-label={t('Timeline controls')}
        onKeyDown={toolbarKey}
      >
        <div className="timeline-primary-controls">
          <div className="transport timeline-control-group">
            {firstButton}
            <Button
              icon="stepb"
              title="Previous frame"
              disabled={busy || !frames.length || time <= first}
              onClick={() => seek(frames[previousIndex].time)}
            />
            <Button
              icon={playing ? 'pause' : 'play'}
              className="play-button"
              title={playing ? 'Pause (Space)' : 'Play (Space)'}
              disabled={busy || frames.length < 2}
              onClick={play}
            />
            <Button
              icon="stepf"
              title="Next frame"
              disabled={busy || !frames.length || time >= last}
              onClick={() => seek(frames[nextIndex].time)}
            />
            {lastButton}
            {loopButton}
          </div>
          <div className="time-readout timeline-control-group">
            <Numeric
              value={time}
              formatValue={timelineTimeCaption}
              min={0}
              disabled={busy}
              onChange={(value) => seek(value)}
              label={t('Current time')}
              unit="s"
            />
          </div>
          <div className="timeline-capture-actions timeline-control-group">
            <Button
              icon={recording ? 'stop' : 'record'}
              active={recording}
              className="record-button"
              title={
                recording
                  ? 'Stop recording'
                  : frames.length
                    ? 'Record from playhead'
                    : 'Record scene frames'
              }
              disabled={videoActive}
              onClick={onRecord}
            />
            {onExportVideo && (
              <Button
                icon={videoActive ? 'stop' : 'video'}
                className="timeline-video-button"
                active={videoActive}
                title={videoActive ? 'Stop video' : 'Export video'}
                aria-label={t(videoActive ? 'Stop video' : 'Export video')}
                disabled={!videoActive && (recording || frames.length < 2)}
                onClick={onExportVideo}
              />
            )}
            {snapshotButton}
            <Button
              icon="diamond"
              title="Capture scene pose at playhead (K)"
              disabled={busy}
              onClick={onCapture}
            />
          </div>
          <div className="timeline-key-navigation timeline-control-group">
            {previousKeyButton}
            {nextKeyButton}
          </div>
          {speedControl}
          {zoomControl}
          <div className="timeline-scrolling-range">{rangeControls}</div>
        </div>
        <div className="timeline-persistent-controls">
          <div
            className="segmented timeline-follow timeline-control-group"
            aria-label={t('Follow playhead')}
          >
            {(['off', 'page', 'locked'] as const).map((mode) => (
              <Button
                key={mode}
                icon={`follow-${mode}`}
                active={follow === mode}
                title={`Follow ${mode}`}
                aria-label={t(`Follow ${mode}`)}
                aria-pressed={follow === mode}
                onClick={() => setFollowMode(mode)}
              />
            ))}
          </div>
          <div className="timeline-pinned-range">{rangeControls}</div>
        </div>
      </div>
      <div
        className="timeline-body"
        ref={body}
        data-view-start={viewStart}
        data-view-end={viewEnd}
        data-scroll-lower={limits.lower}
        data-scroll-span={scrollSpan}
        data-scroll-travel={scrollTravel}
        onScroll={(event) => {
          const left = event.currentTarget.scrollLeft;
          if (Math.abs(left - scrollPosition.current) < 0.01 || scrollTravel <= 0) return;
          scrollPosition.current = left;
          setFollow('off');
          const start = limits.lower + (left / scrollTravel) * scrollSpan;
          updateView({ start, end: start + (viewEnd - viewStart) });
        }}
      >
        <div className="lane-labels">
          <div className="lane-header" title={doc.name}>
            {doc.name || t('Scene')}
          </div>
          <button
            className={`lane-label ${lane === 'keys' ? 'active' : ''}`}
            onClick={() => {
              setLane('keys');
              setSnapshotId(null);
              focusTimeline();
            }}
          >
            <Icon name="diamond" size={16} />
            <span>{t('Keyframes')}</span>
            <span className="lane-count">{keys.length}</span>
          </button>
          <div
            className={`lane-label take-label ${lane === 'take' ? 'active' : ''}`}
            title={activeTake ? `${activeTake.name} · ${frames.length} ${t('frames')}` : undefined}
            onContextMenu={(event) => {
              event.preventDefault();
              setLane('take');
              openMenu(event.clientX, event.clientY);
            }}
          >
            <Icon name="record" size={16} />
            {timeline.takes.length ? (
              <Select
                label={t('Take')}
                value={timeline.activeTakeId ?? ''}
                disabled={busy}
                options={timeline.takes.map((item) => ({ value: item.id, label: item.name }))}
                onChange={selectTake}
              />
            ) : (
              <span>{t('Take')}</span>
            )}
            {onCreateTake && (
              <Button
                icon="plus"
                title="New Take"
                disabled={busy}
                onClick={() => {
                  onCreateTake();
                  setLane('take');
                  setTakeSelection(null);
                }}
              />
            )}
          </div>
          <button
            className={`lane-label ${lane === 'snapshots' ? 'active' : ''}`}
            onClick={() => {
              setLane('snapshots');
              focusTimeline();
            }}
          >
            <Icon name="snapshot" size={16} />
            <span>{t('Snapshots')}</span>
            <span className="lane-count">{timeline.snapshots.length}</span>
          </button>
        </div>
        <div className="timeline-scroll" style={{ flex: 'none', width: scrollWidth }}>
          <div
            className="timeline-track"
            ref={track}
            style={{
              width: visible.width,
              minHeight: 160,
              position: 'sticky',
              left: 148,
              overflow: 'clip',
            }}
            onContextMenu={(event) => event.preventDefault()}
            {...pointerHandlers}
          >
            {rangeBand && (
              <div
                className="timeline-range-band"
                data-testid="timeline-loop-range"
                style={{
                  left: Math.max(0, x(rangeBand.start)),
                  width: Math.max(
                    0,
                    Math.min(visible.width, x(rangeBand.end)) - Math.max(0, x(rangeBand.start)),
                  ),
                }}
              />
            )}
            <div
              className="time-ruler"
              onPointerDown={(event) => startGesture(event, 'scrub', 'take')}
            >
              {Array.from(
                { length: Math.max(0, lastTick - firstTick + 1) },
                (_, index) => firstTick + index,
              ).map((i) => (
                <span key={i} className="ruler-tick" style={{ left: x(i * step) }}>
                  {Number((i * step).toFixed(4))}
                  <small>{t('s')}</small>
                </span>
              ))}
              <span className="playhead-head" style={{ left: x(time) }} aria-hidden="true" />
            </div>
            <div
              className="timeline-grid"
              style={{
                backgroundPosition: `${x(firstTick * step)}px 0`,
                backgroundSize: `${step * pixelsPerSecond}px 44px`,
              }}
            >
              <div
                className="key-lane"
                data-lane="keyframes"
                onPointerDown={(event) => startGesture(event, 'select', 'keys')}
              >
                {visibleKeys.map((key) => (
                  <button
                    key={key.id}
                    data-scene-frame={key.id}
                    className={`keyframe ${selection.has(key.id) ? 'selected' : ''}`}
                    style={{
                      left: x(keyPreview?.get(key.id) ?? key.time),
                      ...markerLayout.get(key.id),
                      padding: 0,
                    }}
                    aria-label={`${t('Keyframe')} ${key.time.toFixed(2)} ${t('seconds')} ${t('Keyframes')}`}
                    title={`${key.name || t('Keyframe')} · ${key.time.toFixed(3)} s`}
                    disabled={busy}
                    onClick={(event) => {
                      event.stopPropagation();
                      if (event.detail === 0) {
                        setSelection(new Set([key.id]));
                        setPrimary(key.id);
                        setLane('keys');
                        seek(key.time, 'keyframes');
                      }
                    }}
                    onPointerDown={(event) => startGesture(event, 'key', 'keys', key)}
                    onDoubleClick={(event) => {
                      event.stopPropagation();
                      onLoadKey?.(key.id);
                    }}
                    {...pointerHandlers}
                  >
                    <span className="primary-marker" />
                  </button>
                ))}
                {preview?.lane === 'keys' && !preview.range && (
                  <div
                    className="timeline-selection-band"
                    style={{
                      left: x(preview.start),
                      width: Math.max(1, x(preview.end) - x(preview.start)),
                    }}
                  />
                )}
              </div>
              <div
                className={`take-lane ${lane === 'take' ? 'selected' : ''}`}
                data-lane="take"
                onPointerDown={(event) => startGesture(event, 'select', 'take')}
              >
                {frames.length > 0 && (
                  <div
                    className="take-strip"
                    style={{
                      left: Math.max(0, x(frames[0].time)),
                      width: Math.max(
                        0,
                        Math.min(visible.width, x(frames.at(-1)!.time)) -
                          Math.max(0, x(frames[0].time)),
                      ),
                    }}
                  />
                )}
                {[...frameColumns].map((column) => (
                  <span className="take-frame-tick" key={column} style={{ left: column }} />
                ))}
                {takeSelection && frames[takeSelection[0]] && frames[takeSelection[1]] && (
                  <div
                    className="timeline-selection-band"
                    style={{
                      left: x(frames[takeSelection[0]].time) - 2,
                      width: Math.max(
                        4,
                        x(frames[takeSelection[1]].time) - x(frames[takeSelection[0]].time) + 4,
                      ),
                    }}
                  />
                )}
                {recording && (
                  <span className="recording-caption" style={{ left: x(time) + 8 }}>
                    {t('Recording')}
                  </span>
                )}
              </div>
              <div
                className="snapshot-lane"
                data-lane="snapshots"
                onPointerDown={(event) => startGesture(event, 'scrub', 'snapshots')}
              >
                {timeline.snapshots
                  .filter((item) => item.time >= viewStart && item.time <= viewEnd)
                  .map((item) => (
                    <button
                      key={item.id}
                      className={`snapshot-marker ${snapshotId === item.id ? 'selected' : ''}`}
                      style={{ left: x(item.time) }}
                      title={`${item.name} · ${item.time.toFixed(3)} s`}
                      aria-label={item.name}
                      disabled={busy}
                      onPointerDown={(event) => event.stopPropagation()}
                      onClick={(event) => {
                        event.stopPropagation();
                        setLane('snapshots');
                        setSnapshotId(item.id);
                        setSelection(new Set());
                        focusTimeline();
                      }}
                      onDoubleClick={(event) => {
                        event.stopPropagation();
                        onLoadSnapshot?.(item.id);
                      }}
                    >
                      <Icon name="snapshot" size={14} />
                    </button>
                  ))}
              </div>
            </div>
            <div className="playhead" style={{ left: x(time) }} aria-hidden="true">
              <i />
            </div>
          </div>
        </div>
      </div>
      {(selectedKey || selectedSnapshot || (lane === 'keys' && selection.size > 1)) && (
        <div className="timeline-details">
          <Icon
            name={lane === 'keys' ? 'diamond' : lane === 'snapshots' ? 'snapshot' : 'record'}
            size={16}
          />
          {selectedKey ? (
            <>
              <input
                className="key-name"
                aria-label={t('Keyframe name')}
                value={draftName}
                placeholder={t('Keyframe')}
                onChange={(event) => setDraftName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Escape') {
                    setDraftName(selectedKey.name ?? '');
                    setDraftTime(selectedKey.time);
                    event.stopPropagation();
                  }
                }}
              />
              <Numeric
                label={t('Keyframe time')}
                value={draftTime}
                min={0}
                unit="s"
                onChange={setDraftTime}
              />
              <Button
                title="Apply keyframe details"
                disabled={
                  busy ||
                  !draftName.trim() ||
                  (draftName === selectedKey.name && draftTime === selectedKey.time)
                }
                onClick={applyDetails}
              >
                {t('Apply')}
              </Button>
              <Button disabled={busy} onClick={() => onLoadKey?.(selectedKey.id)}>
                {t('Load')}
              </Button>
            </>
          ) : selectedSnapshot ? (
            <>
              <span className="detail-name">{selectedSnapshot.name}</span>
              <span className="detail-time">{selectedSnapshot.time.toFixed(3)} s</span>
              <Button disabled={busy} onClick={() => onLoadSnapshot?.(selectedSnapshot.id)}>
                {t('Load')}
              </Button>
            </>
          ) : (
            <span>
              {selection.size} {t('selected keyframes')}
            </span>
          )}
          <Button icon="delete" title="Delete selected item" disabled={busy} onClick={remove} />
        </div>
      )}
      {error && (
        <div className="timeline-error" role="alert">
          {t(error)}
          <Button icon="close" title="Close" onClick={() => setError('')} />
        </div>
      )}
      {menu && (
        <div
          ref={menuAnchor}
          className="timeline-context-anchor"
          style={{ left: menu.x, top: menu.y }}
        >
          <div ref={menuRef} popover="auto" className="control-popover timeline-context-host">
            <Squircle
              radius={8}
              className="timeline-context"
              role="menu"
              aria-label={t('Timeline actions')}
              onKeyDown={(event) => {
                if (event.key === 'Escape') {
                  event.stopPropagation();
                  setMenu(null);
                  focusTimeline();
                }
              }}
            >
              {contextItem('Select all', selectAll, false, undefined, '⌘ A')}
              {lane === 'keys' &&
                contextItem(
                  'Copy keyframes',
                  () => void copyKeys(),
                  !selection.size,
                  'copy',
                  '⌘ C',
                )}
              {lane === 'keys' &&
                contextItem('Paste keyframes', () => void paste(), busy, 'paste', '⌘ V')}
              {lane === 'keys' &&
                contextItem(
                  'Duplicate keyframes',
                  duplicate,
                  busy || !selection.size,
                  'duplicate',
                  '⌘ D',
                )}
              {contextItem(
                'Delete selection',
                remove,
                busy || (lane === 'keys' && !selection.size) || (lane === 'take' && !takeSelection),
                'delete',
                'Delete',
              )}
              {lane === 'take' &&
                onRange &&
                contextItem(
                  'Loop selected frames',
                  rangeFromSelection,
                  !takeSelection || takeSelection[0] === takeSelection[1],
                  'loop',
                )}
              {onRange && contextItem('Clear loop range', () => setRange(null), !range)}
              {onCreateTake && contextItem('New Take', onCreateTake, busy, 'plus')}
              {activeTake && contextItem('Clear take', clearTake, busy || !frames.length, 'delete')}
              {lane === 'take' &&
                activeTake &&
                contextItem('Delete take', deleteTake, busy, 'delete')}
              {contextItem('Fit timeline', fit, false, 'fit', 'F')}
            </Squircle>
          </div>
        </div>
      )}
    </div>
  );
}
