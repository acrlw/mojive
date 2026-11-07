import { sceneDisplayName } from './scene-labels';
import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import type {
  EntityKind,
  NativeMenuState,
  Frame,
  GeometryMode,
  Joint,
  ProjectFile,
  SceneDocument,
  SceneNode,
  Tool,
} from './contracts';
import {
  authorJointTransform,
  createPrimitive,
  canRemoveNode,
  disposeDocument,
  removeNode,
  setJoint,
} from './scene';
import {
  decodeDocument,
  encodeDocument,
  type Keyframe,
  type WorkspaceTimeline,
  type SceneSnapshot,
  takeFrameIndex,
  assertPoseCompatible,
  capturePose,
  applyPose,
  type Pose,
} from './editor';
import { sameHistoryValue } from './historyValue';
import { t, message, type LocalizedText } from './i18n';
import { SettingsDialog, useSettings } from './settings';
import { OutputPanel } from './OutputPanel';
import { JointPanel } from './JointPanel';
import { ActuatorsPanel } from './ActuatorsPanel';
import { FileDialog, type FileDialogKind, type FileDialogProps } from './FileDialog';
import { ProjectExplorer } from './ProjectExplorer';
import { ConfirmSceneDialog } from './ConfirmSceneDialog';
import { AppMenu } from './AppMenu';
import { WorkspaceMenuItems, workspaceMenus, workspaceMenuLabel } from './WorkspaceMenu';
import { commandKey, shortcut } from './platform';
import {
  actionLabel,
  actionParts,
  matchesKey,
  keyboardChord,
  focusAccelerator,
  focusAllAccelerator,
  keyLabel,
  type InputAction,
} from './input-bindings';
import { PointerHint } from './InputHints';
import { InteractionGuide } from './InteractionGuide';
import { TooltipHost } from './Tooltip';
import { InfoDialog } from './InfoDialog';
import { CamerasPanel } from './CamerasPanel';
import { CameraProperties } from './CameraProperties';
import { CameraPreview } from './CameraPreview';
import { OperationNotice, type OperationNoticeData } from './OperationNotice';
import { isViewportPickable, setViewportPickable } from './scene-selection';
import { LightProperties } from './LightProperties';
import type { SceneCameraState } from './scene-camera';
import { exportTakeVideo } from './export-video';
import { LayersPopup, LayersPanel, DEFAULT_VIEWPORT_LAYERS } from './ViewportLayers';
import type { ViewportStats } from './Viewport';
import { jointTypeLabel, JointEditor, jointIcon } from './JointEditor';
import { Splitter } from './Splitter';
import { filterSceneNodes, nodeCategory } from './scene-tree';
import { SCENE_CATALOG } from './SceneCatalog';
import {
  SearchField,
  Button,
  Empty,
  Icon,
  Numeric,
  Section,
  Segmented,
  Squircle,
  VectorFields,
  SwitchRow,
  ColorPicker,
} from './components';
import { Timeline } from './Timeline';
import { Viewport, type ViewportHandle } from './Viewport';
import { PhysicsDocument } from './physics/PhysicsDocument';
import { loadMujocoBackend } from './physics/WorkerBackend';
import { loadPhysicsWorkspace, savePhysicsWorkspace } from './physics/workspace';
import { loadModelResource } from './resources/loadModelBundle';
import type { ModelLoadOptions } from './resources/loading';
import { SceneLoadProgress, useSceneLoading } from './SceneLoadProgress';
import { EnvironmentProperties } from './physics/EnvironmentProperties';
import { SimulationToolbar } from './physics/SimulationToolbar';
import { ENVIRONMENT_ID } from './physics/SceneBinding';
import type { SimulationState } from './physics/SimulationSession';
import type { PhysicsCommand, PhysicsSnapshot } from './physics/contracts';
import { applySceneState, captureSceneState } from './physics/sceneState';
import { capturePhysicsHistory, physicsHistoryChanged, restorePhysicsHistory, type PhysicsHistoryEntry } from './physics/gestureHistory';
import { AssetBrowser } from './assets/AssetBrowser';
import { AssetPreviewDialog } from './assets/AssetPreviewDialog';
import { fileAsset, sampleAsset, readAssetLibrary, writeAssetLibrary, rememberAsset, type Asset } from './assets/catalog';
import { loadAsset, loadAssetPreview } from './assets/loadAsset';
import { browserBundles } from './resources/browserBundle';
import { droppedFiles } from './resources/dropFiles';

type TimelineSnapshot = {
  kind: 'timeline';
  pose: Pose;
  keys: Keyframe[];
  duration: number;
  timeline: WorkspaceTimeline;
  physics?: PhysicsSnapshot;
};
type DocumentSnapshot = Omit<TimelineSnapshot, 'kind' | 'pose'> & {
  kind: 'document';
  document: ReturnType<typeof encodeDocument>;
  pose?: Pose;
};
type Snapshot = DocumentSnapshot | TimelineSnapshot | PhysicsHistoryEntry;
type Log = { id: number; level: 'info' | 'warn' | 'err'; time: string; text: LocalizedText };
const kinds: Record<string, string> = {
  environment: 'env',
  world: 'world',
  geom: 'geom',
  robot: 'robot',
  link: 'body',
  joint: 'joint',
  box: 'shape-box',
  sphere: 'shape-sphere',
  cylinder: 'shape-cylinder',
  capsule: 'shape-capsule',
  cone: 'shape-cone',
  ellipsoid: 'shape-ellipsoid',
  ground: 'shape-plane',
  camera: 'camera',
  light: 'lightbulb',
};
const menus = workspaceMenus.map((menu) => menu.name);
const tools: { id: Tool; icon: string; title: string }[] = [
  { id: 'select', icon: 'select', title: 'Select' },
  { id: 'position', icon: 'move', title: 'Position' },
  { id: 'rotation', icon: 'rotate', title: 'Rotation' },
  { id: 'scale', icon: 'scale', title: 'Scale' },
  { id: 'joint', icon: 'joint', title: 'Joint' },
];
export default function App() {
  const [settings, updateSettings] = useSettings();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [layers, setLayers] = useState({ ...DEFAULT_VIEWPORT_LAYERS });
  const resolvedLayers = useMemo(() => ({
    ...layers,
    grid: settings.showGrid,
    viewCube: settings.showViewCube,
    helpers: settings.showEntityIcons,
    influence: settings.showEntityInfluence ?? true,
  }), [layers, settings.showGrid, settings.showViewCube, settings.showEntityIcons, settings.showEntityInfluence]);
  function changeLayers(next: typeof resolvedLayers) {
    setLayers(next);
    updateSettings({
      showGrid: next.grid,
      showViewCube: next.viewCube,
      showEntityIcons: next.helpers,
      showEntityInfluence: next.influence,
    });
  }
  const [layersOpen, setLayersOpen] = useState(false);
  const [sceneCamera, setSceneCamera] = useState<SceneCameraState>({ id: null, pilot: false });
  const [videoProgress, setVideoProgress] = useState<number | null>(null);
  const videoAbort = useRef<AbortController | null>(null);
  useEffect(() => () => videoAbort.current?.abort(), []);
  const [stats, setStats] = useState<ViewportStats | null>(null);
  const [pinnedId, setPinnedId] = useState<string | null>(null);
  const [typeFilter, setTypeFilter] = useState('all');
  const [searchVisible, setSearchVisible] = useState(false);
  const [snapEnabled, setSnapEnabled] = useState(false);
  const [sceneId, setSceneId] = useState('joint_gizmo');
  const [activeJoint, setActiveJoint] = useState<string | null>(null);
  const [layout, setLayout] = useState(() => {
    try {
      return {
        left: 272,
        right: 340,
        bottom: 300,
        ...JSON.parse(localStorage.getItem('mojive.layout') || '{}'),
      };
    } catch {
      return { left: 272, right: 340, bottom: 300 };
    }
  });
  const [windowSize, setWindowSize] = useState({ width: innerWidth, height: innerHeight });
  useEffect(() => {
    const resize = () => setWindowSize({ width: innerWidth, height: innerHeight });
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, []);
  useEffect(() => {
    localStorage.setItem('mojive.layout', JSON.stringify(layout));
  }, [layout]);
  const nativeMenu = window.mojive?.hasNativeMenu ?? false;
  const [replacePrompt, setReplacePrompt] = useState(false);
  const [replaceReason, setReplaceReason] = useState<'open' | 'close'>('open');
  const [about, setAbout] = useState(false);
  const [fileDialog, setFileDialog] = useState<{
    kind: FileDialogKind;
    initialPath?: string;
    extension?: '.mojive.json' | '.png' | '.webm';
    caption?: string;
    suggestedName?: string;
    onChoose: (path: string, overwrite: boolean) => Promise<void>;
    onCancel: () => void;
  } | null>(null);
  function chooseFile(
    kind: FileDialogKind,
    action: (path: string, overwrite: boolean) => Promise<void>,
    options: {
      initialPath?: string;
      extension?: '.mojive.json' | '.png' | '.webm';
      caption?: string;
      suggestedName?: string;
    } = {},
  ) {
    if (fileDialog) return Promise.resolve(false);
    return new Promise<boolean>((resolve) => {
      const dialog: FileDialogProps = {
        kind,
        ...options,
        onChoose: async (path, overwrite) => {
          await action(path, overwrite);
          setFileDialog((current) => current === dialog ? null : current);
          resolve(true);
        },
        onCancel: () => {
          setFileDialog((current) => current === dialog ? null : current);
          resolve(false);
        },
      };
      setFileDialog(dialog);
    });
  }
  async function requestClose() {
    if (closing.current) return;
    closing.current = true;
    try {
      // Abort also releases the candidate's dirty prompt before opening the close prompt.
      sceneLoading.cancel();
      await stopRecording();
      if (fileDialog || replaceAnswer.current || videoAbort.current) return;
      if (dirtyRef.current && !await requestReplacement('close')) return;
      await window.mojive?.confirmClose();
    } finally { closing.current = false; }
  }
  const closing = useRef(false);
  const replaceAnswer = useRef<((answer: boolean) => void) | null>(null);
  const dirtyRef = useRef(false);
  const [sceneEpoch, setSceneEpoch] = useState(0);
  const [doc, setDoc] = useState<SceneDocument | null>(null);
  const physics = useRef<PhysicsDocument | null>(null);
  const physicsUnsubscribe = useRef<(() => void) | null>(null);
  const [simulation, setSimulation] = useState<SimulationState | null>(null);
  const historyWork = useRef<Promise<unknown>>(Promise.resolve());
  const historyBusy = useRef(false);
  const recordingStopWork = useRef<Promise<void> | null>(null);
  useEffect(() => () => {
    replaceAnswer.current?.(false);
    physicsUnsubscribe.current?.();
    physics.current?.dispose();
    physics.current = null;
  }, []);
  const [selected, setSelected] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const [cameraPreviewEnabled, setCameraPreviewEnabled] = useState(false);
  const [tool, setTool] = useState<Tool>('select');
  const [frame, setFrame] = useState<Frame>('world');
  const [geometry, setGeometry] = useState<GeometryMode>('default');
  const [projection, setProjection] = useState<'perspective' | 'orthographic'>('perspective');
  const bindingHint = (label: string, action: InputAction) => {
    const key = actionLabel(settings.inputBindings, action);
    return `${t(label)}${key ? ` (${key})` : ''}`;
  };
  const [left, setLeft] = useState('scene');
  const [assetLibrary, setAssetLibrary] = useState(() => readAssetLibrary(localStorage));
  const [uploadedAssets, setUploadedAssets] = useState<Asset[]>([]);
  const [assetPreview, setAssetPreview] = useState<Asset | null>(null);
  useEffect(() => {
    try { writeAssetLibrary(localStorage, assetLibrary); }
    catch { /* Browsing remains available when persistent browser storage is full. */ }
  }, [assetLibrary]);
  const [bottom, setBottom] = useState('timeline');
  const [menu, setMenu] = useState<string | null>(null);
  const menuBar = useRef<HTMLElement>(null);
  const menuTrigger = useRef<HTMLElement | null>(null);
  const hoveredMenu = useRef<string | null>(null);
  const [treeContext, setTreeContext] = useState<{
    id: string;
    x: number;
    y: number;
    anchor: HTMLElement;
  } | null>(null);
  function activateMenu(name: string, anchor: HTMLElement) {
    menuTrigger.current = anchor;
    hoveredMenu.current = null;
    setMenu(name);
  }
  function switchMenu(direction: number) {
    const next = menus[(menus.indexOf(menu!) + direction + menus.length) % menus.length];
    const anchor = menuBar.current?.querySelector<HTMLButtonElement>(`[data-menu="${next}"]`);
    if (anchor) activateMenu(next, anchor);
  }
  function openTreeContext(node: SceneNode, anchor: HTMLElement, x: number, y: number) {
    setSelected(node.id);
    setTreeContext({ id: node.id, anchor, x, y });
    activateMenu('Object', anchor);
  }
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [showInspector, setShowInspector] = useState(true);
  const [showLeft, setShowLeft] = useState(true);
  const [showBottom, setShowBottom] = useState(true);
  const [inspectorTab, setInspectorTab] = useState('Properties');
  const [help, setHelp] = useState(false);
  const sceneLoading = useSceneLoading();
  const busy = sceneLoading.state !== null;
  const [notice, setNoticeState] = useState<OperationNoticeData | null>(null);
  const noticeId = useRef(0);
  const noticeActionEpoch = useRef(0);
  const [keys, setKeys] = useState<Keyframe[]>([]);
  const [timelineData, setTimelineData] = useState<WorkspaceTimeline>({
    snapshots: [],
    takes: [],
    activeTakeId: null,
  });
  const [recording, setRecording] = useState(false);
  const [takeViews, setTakeViews] = useState<
    Record<string, { cursor: number; range: [number, number] | null }>
  >({});
  const [playbackSource, setPlaybackSource] = useState<'keyframes' | 'take'>('keyframes');
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(10);
  const [playing, setPlaying] = useState(false);
  const [loop, setLoop] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [dirty, setDirty] = useState(false);
  const [project, setProject] = useState<{
    path: string;
    name: string;
    files: ProjectFile[];
  } | null>(null);
  const projectFiles = project?.files ?? [];
  const [logs, setLogs] = useState<Log[]>([]);
  const [historyTick, setHistoryTick] = useState(0);
  const viewport = useRef<ViewportHandle>(null);
  const undo = useRef<Snapshot[]>([]);
  const redo = useRef<Snapshot[]>([]);
  const pending = useRef<Snapshot | null>(null);
  const recordingJointBaseline = useRef<PhysicsHistoryEntry | null>(null);
  const logId = useRef(0);
  const recordClock = useRef<{
    start: number;
    time: number;
    id: string;
    simulationStart?: number;
  } | null>(null);
  const seekEpoch = useRef(0);
  const seekWork = useRef<Promise<unknown>>(Promise.resolve());
  const recordingStartWork = useRef<Promise<void> | null>(null);
  const state = useRef({
    doc,
    keys,
    duration,
    time,
    playing,
    loop,
    timelineData,
    recording,
    playbackSource,
    takeViews,
    playbackSpeed,
  });
  state.current = {
    doc,
    keys,
    duration,
    time,
    playing,
    loop,
    timelineData,
    recording,
    playbackSource,
    takeViews,
    playbackSpeed,
  };
  dirtyRef.current = dirty;
  function log(text: LocalizedText, level: Log['level'] = 'info') {
    setLogs((l) => [
      ...l.slice(-499),
      {
        id: ++logId.current,
        level,
        time: new Date().toLocaleTimeString('en-GB', { hour12: false }),
        text,
      },
    ]);
  }
  function invalidateNoticeAction() {
    noticeActionEpoch.current++;
    setNoticeState((current) => (current?.action ? { ...current, action: undefined } : current));
  }
  function setNotice(message: LocalizedText, level: OperationNoticeData['level'] = 'info') {
    const id = ++noticeId.current;
    setNoticeState(message ? { id, message, level } : null);
  }
  function historyNotice(message: LocalizedText, direction: 'undo' | 'redo', expected: Snapshot) {
    const id = ++noticeId.current,
      epoch = noticeActionEpoch.current;
    setNoticeState({
      id,
      message,
      action: {
        label: direction === 'undo' ? 'Undo' : 'Redo',
        onClick: () => {
          const stack = direction === 'undo' ? undo : redo;
          if (
            noticeId.current !== id ||
            noticeActionEpoch.current !== epoch ||
            pending.current ||
            state.current.recording ||
            state.current.playing ||
            sceneLoading.isActive() ||
            stack.current.at(-1) !== expected
          )
            return;
          history(direction);
        },
      },
    });
  }
  function changed() {
    invalidateNoticeAction();
    setRevision((r) => r + 1);
    setDirty(true);
  }
  function transformChanged() {
    if (doc && node && doc.joints.some((j) => j.id === node.id)) authorJointTransform(doc, node.id);
    changed();
  }
  function snapshot(compact = false): DocumentSnapshot | TimelineSnapshot | null {
    const s = state.current;
    return s.doc
      ? {
          ...(compact
            ? { kind: 'timeline' as const, pose: capturePose(s.doc) }
            : { kind: 'document' as const, document: encodeDocument(s.doc) }),
          physics: physics.current?.session.current.snapshot ?? undefined,
          keys: structuredClone(s.keys),
          duration: s.duration,
          timeline: structuredClone(s.timelineData),
        }
      : null;
  }
  function begin(pauseSimulation = true, compact = false) {
    invalidateNoticeAction();
    setPlaying(false);
    if (pauseSimulation && !state.current.recording) physics.current?.session.pause();
    // Pose-only recording history shares scene assets. Promote before the first authored edit;
    // its original pose still restores the start of the recording, not the moment of promotion.
    if (!compact && pending.current?.kind === 'timeline' && state.current.doc) {
      const baseline = pending.current;
      const promoted: DocumentSnapshot = {
        ...baseline, kind: 'document', document: encodeDocument(state.current.doc),
      };
      pending.current = promoted;
      historyWork.current = historyWork.current.then(() => {
        promoted.physics = baseline.physics;
      });
    }
    if (pending.current?.kind === 'physics') commit();
    if (!pending.current) {
      const baseline = snapshot(compact);
      pending.current = baseline;
      if (baseline && physics.current) {
        // Capture before subsequent commands in the session queue, including pending playback.
        const capture = physics.current.session.capture();
        historyWork.current = Promise.all([historyWork.current, capture]).then(([, value]) => { baseline.physics = value; }).catch(report);
      }
    }
  }
  function beginPhysics() {
    invalidateNoticeAction();
    setPlaying(false);
    // Recording already owns a document transaction that includes every captured frame.
    if (state.current.recording || pending.current || !physics.current) return;
    const baseline = capturePhysicsHistory(physics.current.session);
    pending.current = baseline;
    historyWork.current = Promise.all([historyWork.current, baseline.captured]).catch(report);
  }
  function beginJoint() {
    if (!physics.current) {
      begin(true, true);
      return;
    }
    physics.current.session.pause();
    // A recording owns the document transaction; keep its joint gesture cancellation separate.
    if (state.current.recording && !recordingJointBaseline.current) {
      const baseline = capturePhysicsHistory(physics.current.session);
      recordingJointBaseline.current = baseline;
      historyWork.current = Promise.all([historyWork.current, baseline.captured]).catch(report);
    }
    beginPhysics();
  }
  function jointChanged() {
    // Physics targets publish their applied frame asynchronously through the session.
    if (!physics.current) changed();
  }
  function cancelJoint() {
    const baseline = recordingJointBaseline.current;
    recordingJointBaseline.current = null;
    if (baseline) {
      historyWork.current = Promise.all([historyWork.current, restore(baseline)]).catch(report);
    } else {
      cancel();
    }
  }
  function cancel() {
    const backup = pending.current;
    pending.current = null;
    if (!backup) return;
    // Physics cancellation must enter the session queue before another gesture can begin.
    historyWork.current = (backup.kind === 'physics'
      ? Promise.all([historyWork.current, restore(backup)])
      : historyWork.current.then(() => restore(backup))).catch(report);
  }
  function commit(): Snapshot | null {
    recordingJointBaseline.current = null;
    if (state.current.recording) return null;
    if (pending.current?.kind === 'physics') {
      const baseline = pending.current, runtime = physics.current;
      pending.current = null;
      if (!runtime || runtime.session !== baseline.session || runtime.session.current.source !== baseline.source) return null;
      // Queue after the gesture's final command. Assets and viewer authoring did not change.
      const current = capturePhysicsHistory(runtime.session);
      historyWork.current = Promise.all([historyWork.current, physicsHistoryChanged(baseline, current)]).then(([, modified]) => {
        if (!modified || physics.current !== runtime || runtime.session.current.source !== baseline.source) return;
        undo.current.push(baseline);
        if (undo.current.length > 50) undo.current.shift();
        redo.current = [];
        setHistoryTick((tick) => tick + 1);
        setDirty(true);
      }).catch(report);
      return baseline;
    }
    if (pending.current && physics.current) {
      const baseline = pending.current, runtime = physics.current;
      pending.current = null;
      const current = snapshot(baseline.kind === 'timeline')!;
      const capture = runtime.session.capture();
      historyWork.current = Promise.all([historyWork.current, capture]).then(([, value]) => {
        if (physics.current !== runtime) return;
        current.physics = value;
        if (sameSnapshot(baseline, current)) return;
        undo.current.push(baseline);
        if (undo.current.length > 50) undo.current.shift();
        redo.current = [];
        setHistoryTick((tick) => tick + 1);
        setDirty(true);
      }).catch(report);
      return baseline;
    }
    if (pending.current) {
      if (sameSnapshot(pending.current, snapshot(pending.current.kind === 'timeline')!)) {
        pending.current = null;
        return null;
      }
      const committed = pending.current;
      undo.current.push(committed);
      if (undo.current.length > 50) undo.current.shift();
      pending.current = null;
      redo.current = [];
      setHistoryTick((t) => t + 1);
      setDirty(true);
      invalidateNoticeAction();
      return committed;
    }
    return null;
  }
  function sameSnapshot(a: DocumentSnapshot | TimelineSnapshot, b: DocumentSnapshot | TimelineSnapshot) {
    // Check inexpensive authored state before potentially large geometry arrays.
    return a.kind === b.kind && a.duration === b.duration && sameHistoryValue(a.keys, b.keys) &&
      sameHistoryValue(a.timeline, b.timeline) && sameHistoryValue(a.physics, b.physics) &&
      sameHistoryValue(a.pose, b.pose) &&
      (a.kind !== 'document' || b.kind !== 'document' || sameHistoryValue(a.document, b.document));
  }
  async function transaction(label: LocalizedText, fn: () => void | Promise<void>) {
    begin();
    try {
      const result = fn();
      if (result) await result;
      changed();
      const committed = commit();
      if (committed) historyNotice(label, 'undo', committed);
      else setNotice(label);
    } catch (error) {
      const backup = pending.current;
      pending.current = null;
      if (backup) void historyWork.current.then(() => restore(backup)).catch(report);
      report(error);
    }
  }
  async function restore(s: Snapshot) {
    seekEpoch.current++;
    if (s.kind === 'physics') {
      await restorePhysicsHistory(s, physics.current?.session);
      setPlaying(false);
      invalidateNoticeAction();
      setDirty(true);
      return;
    }
    const old = state.current.doc;
    const next = s.kind === 'timeline' ? old : decodeDocument(s.document);
    if (!next) throw new Error('The history entry requires an active scene.');
    try {
      if (s.kind === 'timeline') {
        await applySceneState(next, { pose: s.pose, simulation: s.physics }, physics.current);
      } else {
        if (s.pose) applyPose(next, s.pose);
        if (physics.current) {
          if (!s.physics) throw new Error('The history entry has no simulation state.');
          await physics.current.restore(next, s.physics);
        }
      }
    } catch (error) { if (next !== old) disposeDocument(next); throw error; }
    setDoc(next);
    state.current.doc = next;
    setKeys(s.keys);
    state.current.keys = s.keys;
    changeTimeline(s.timeline);
    setRecording(false);
    state.current.recording = false;
    recordClock.current = null;
    // Undo restores authored content, while the retained cursor and view still need their extent.
    const extent = Math.max(s.duration, state.current.duration, state.current.time);
    setDuration(extent);
    state.current.duration = extent;
    setPlaying(false);
    setSelected((id) => (next.nodes.some((n) => n.id === id) ? id : null));
    setPinnedId((id) => (next.nodes.some((n) => n.id === id) ? id : null));
    changed();
    if (old && old !== next) disposeDocument(old);
  }
  async function history(direction: 'undo' | 'redo') {
    if (historyBusy.current) return;
    historyBusy.current = true;
    try {
      physics.current?.session.pause();
      await stopRecording();
      await historyWork.current;
      const from = direction === 'undo' ? undo : redo,
        to = direction === 'undo' ? redo : undo;
      const s = from.current.at(-1);
      if (!s) return;
      const current = s.kind === 'physics' && physics.current
        ? capturePhysicsHistory(physics.current.session) : snapshot(s.kind === 'timeline');
      if (!current) return;
      if (current.kind === 'physics') await current.captured;
      else if (physics.current) current.physics = await physics.current.session.capture();
      await restore(s);
      from.current.pop();
      to.current.push(current);
      setHistoryTick((t) => t + 1);
      historyNotice(
        direction === 'undo' ? 'Edit undone' : 'Edit restored',
        direction === 'undo' ? 'redo' : 'undo',
        current,
      );
    } catch (error) {
      report(error);
    } finally { historyBusy.current = false; }
  }

  function report(error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    log(message, 'err');
    setNotice(message, 'error');
    setBottom('output');
    setShowBottom(true);
  }
  function installDocument(next: SceneDocument, runtime: PhysicsDocument | null = null) {
    seekEpoch.current++;
    physicsUnsubscribe.current?.();
    physics.current?.dispose();
    physics.current = runtime;
    setSimulation(runtime?.session.current ?? null);
    let appliedFrame = runtime?.session.current.frame;
    physicsUnsubscribe.current = runtime?.session.subscribe((value) => {
      setSimulation(value);
      if (value.frame !== appliedFrame) {
        appliedFrame = value.frame;
        setRevision((revision) => revision + 1);
      }
      if (value.error) report(value.error);
    }) ?? null;
    runtime?.setErrorHandler(report);
    invalidateNoticeAction();
    setNotice('');
    setSceneEpoch((v) => v + 1);
    const old = state.current.doc;
    setDoc(next);
    state.current.doc = next;
    setSelected(null);
    setPinnedId(null);
    setActiveJoint(null);
    setTypeFilter('all');
    setSearch('');
    setExpanded(
      new Set(
        next.nodes
          .filter(
            (n) =>
              n.kind === 'world' ||
              n.kind === 'robot' ||
              (n.kind === 'link' &&
                !next.joints.some((j) => j.child === n.id || j.bodyId === n.id)),
          )
          .map((n) => n.id),
      ),
    );
    setKeys([]);
    state.current.keys = [];
    changeTimeline({ snapshots: [], takes: [], activeTakeId: null });
    setTakeViews({});
    state.current.takeViews = {};
    state.current.time = 0;
    setRecording(false);
    state.current.recording = false;
    setPlaybackSource('keyframes');
    state.current.playbackSource = 'keyframes';
    setTime(0);
    setPlaying(false);
    setDirty(false);
    undo.current = [];
    redo.current = [];
    pending.current = null;
    recordingJointBaseline.current = null;
    setHistoryTick((t) => t + 1);
    setRevision((r) => r + 1);
    for (const w of next.warnings) log(w, 'warn');
    log(
      message('Loaded {{name}} · {{objects}} objects · {{joints}} joints', {
        name: sceneDisplayName(next),
        objects: next.nodes.length,
        joints: next.joints.length,
      }),
    );
    if (old) disposeDocument(old);
  }
  function answerReplacement(answer: boolean) {
    replaceAnswer.current?.(answer);
  }
  function requestReplacement(reason: 'open' | 'close', signal?: AbortSignal): Promise<boolean> {
    if (signal?.aborted) return Promise.resolve(false);
    return new Promise((resolve) => {
      replaceAnswer.current?.(false);
      const answer = (allowed: boolean) => {
        signal?.removeEventListener('abort', abort);
        // A superseded prompt must never clear the new prompt's owner or UI.
        if (replaceAnswer.current === answer) {
          replaceAnswer.current = null;
          setReplacePrompt(false);
        }
        resolve(allowed);
      };
      const abort = () => answer(false);
      replaceAnswer.current = answer;
      signal?.addEventListener('abort', abort, { once: true });
      setReplaceReason(reason);
      setPlaying(false);
      setReplacePrompt(true);
    });
  }
  async function replaceDocument(
    next: SceneDocument,
    nextKeys: Keyframe[] = [],
    nextDuration = 10,
    nextTimeline: WorkspaceTimeline = { snapshots: [], takes: [], activeTakeId: null },
    runtime: PhysicsDocument | null = null,
    signal?: AbortSignal,
  ) {
    let installed = false;
    try {
      await stopRecording();
      if (signal?.aborted) return false;
      if (dirtyRef.current && !await requestReplacement('open', signal)) return false;
      if (signal?.aborted) return false;
      installDocument(next, runtime);
      installed = true;
      setKeys(nextKeys);
      state.current.keys = nextKeys;
      setDuration(nextDuration);
      state.current.duration = nextDuration;
      changeTimeline(nextTimeline);
      return true;
    } finally {
      if (!installed) { runtime?.dispose(); disposeDocument(next); }
    }
  }
  async function runSceneLoad(label: string, task: (options: ModelLoadOptions) => Promise<void>) {
    try {
      await sceneLoading.run(label, async (options) => {
        await stopRecording();
        physics.current?.session.pause();
        setPlaying(false);
        options.signal?.throwIfAborted();
        await task(options);
      });
    } catch (error) { report(error); }
  }
  async function openSimulation() {
    setMenu(null);
    if (!window.mojive) { setNotice('Drop an MJCF model and its resources into the viewport.'); return; }
    await chooseFile('model', async (path) => {
      setFileDialog(null);
      await runSceneLoad(path, async (options) => {
        const file = await window.mojive!.openModelPath(path);
        const bundle = await loadModelResource(file.name, file.resourceBase, options);
        const runtime = await PhysicsDocument.load(bundle, loadMujocoBackend, options);
        await replaceDocument(runtime.document, runtime.keyframes, 10, undefined, runtime, options.signal);
      });
    });
  }
  async function openAsset(asset: Asset) {
    if (sceneLoading.isActive()) return;
    setAssetPreview(null);
    await runSceneLoad(asset.name, async (options) => {
      const next = await loadAsset(asset, options);
      if (await replaceDocument(next.document, next.runtime?.keyframes ?? [], 10, undefined, next.runtime, options.signal)) {
        setAssetLibrary((library) => rememberAsset(library, asset));
        setSceneId(asset.source.kind === 'sample' ? asset.source.scene : '');
      }
    });
  }
  function toggleFavorite(asset: Asset) {
    setAssetLibrary((library) => ({ ...library, favorites: library.favorites.some((item) => item.id === asset.id)
      ? library.favorites.filter((item) => item.id !== asset.id) : [...library.favorites, asset] }));
  }
  function simulationCommand(command: PhysicsCommand) {
    const runtime = physics.current;
    if (!runtime) return;
    void runtime.session.execute(command).catch(report);
  }
  function simulationTransaction(command: PhysicsCommand) {
    begin(); simulationCommand(command); commit();
  }
  function toggleSimulation() {
    const runtime = physics.current;
    if (!runtime || sceneLoading.isActive() || state.current.recording) return;
    setPlaying(false);
    runtime.session.current.playing ? runtime.session.pause() : runtime.session.play();
  }
  function previousSimulationFrame() {
    const runtime = physics.current;
    if (!runtime || sceneLoading.isActive() || state.current.recording || runtime.session.current.playing || !runtime.session.current.previousFrames) return;
    begin();
    void runtime.session.stepBack().catch(report);
    commit();
  }
  function stepSimulationFrame() {
    const runtime = physics.current;
    if (!runtime || sceneLoading.isActive() || state.current.recording || runtime.session.current.playing) return;
    simulationTransaction({ type: 'step', count: 1 });
  }
  async function openUrdf() {
    setMenu(null);
    if (!window.mojive) {
      setNotice('Drag a URDF file into the viewport.');
      return;
    }
    await chooseFile('urdf', async (path) => {
      setFileDialog(null);
      await runSceneLoad(path, async (options) => {
        const next = await loadAsset(fileAsset(path), options);
        await replaceDocument(next.document, [], 10, undefined, next.runtime, options.signal);
      });
    });
  }

  async function openProject() {
    setMenu(null);
    if (!window.mojive) {
      setNotice('Open project folders in the desktop app.');
      return;
    }
    await chooseFile(
      'project',
      async (path) => {
        const next = await window.mojive!.openProjectPath(path);
        setProject(next);
        setSearch('');
        setSearchVisible(false);
        setLeft('project');
        setShowLeft(true);
        log(message('Opened project folder {{name}}', { name: next.name }));
      },
      { initialPath: project?.path },
    );
  }
  function closeProject() {
    setMenu(null);
    if (!project) return;
    setProject(null);
    setSearch('');
    setSearchVisible(false);
    log('Closed project folder');
  }
  async function loadCatalog(id: string) {
    setMenu(null);
    await openAsset(sampleAsset(id));
  }
  async function openSaved() {
    setMenu(null);
    if (!window.mojive) {
      setNotice('Drag a scene file into the viewport.');
      return;
    }
    await chooseFile('scene', async (path) => {
      setFileDialog(null);
      await runSceneLoad(path, async (options) => {
        const r = await window.mojive!.openScenePath(path);
        options.signal?.throwIfAborted();
        const loaded = await loadPhysicsWorkspace(r.content, physicsBackend, options);
        if (await replaceDocument(loaded.doc, loaded.keys, loaded.duration, loaded.timeline, loaded.runtime, options.signal))
          log(message('Opened {{path}}', { path: r.path }));
      });
    });
  }

  function physicsBackend(id: string) {
    if (id !== 'mujoco') throw new Error(`Unsupported physics backend: ${id}`);
    return loadMujocoBackend;
  }
  async function save() {
    await stopRecording();
    setMenu(null);
    if (!doc) return false;
    if (!window.mojive) {
      setNotice('Save scenes in the desktop app.');
      return false;
    }
    return chooseFile('save', async (path, overwrite) => {
      setPlaying(false);
      physics.current?.session.pause();
      await historyWork.current;
      const current = state.current;
      const saved = await window.mojive!.saveScenePath(
        path,
        await savePhysicsWorkspace(
          current.doc!,
          current.keys,
          current.duration,
          current.timelineData,
          physics.current,
        ),
        overwrite,
      );
      log(message('Saved {{path}}', { path: saved }));
      setDirty(false);
      dirtyRef.current = false;
      setNotice('Project saved');
    });
  }
  function useCamera(id: string | null, pilot = false) {
    try {
      viewport.current?.useSceneCamera(id, pilot);
    } catch (error) {
      report(error);
    }
  }
  function alignCamera(id: string) {
    transaction('Camera aligned to view', () => viewport.current?.alignCameraToView(id));
  }
  function createCamera() {
    if (state.current.recording) {
      setNotice('Stop recording before changing scene structure.');
      return;
    }
    transaction('Camera created', () => {
      const id = viewport.current?.createCameraFromView();
      if (id) setSelected(id);
    });
  }
  function deleteCamera(id: string) {
    if (!state.current.doc || state.current.recording) return;
    transaction('Camera deleted', () => {
      if (sceneCamera.id === id) useCamera(null);
      removeNode(state.current.doc!, id);
      setSelected((selected) => (selected === id ? null : selected));
      setPinnedId((pinned) => (pinned === id ? null : pinned));
    });
  }
  async function captureCamera(id?: string) {
    if (!window.mojive) {
      setNotice('Save images in the desktop app.');
      return;
    }
    try {
      const capture = await viewport.current?.captureCameraView(id);
      if (!capture) return;
      const bytes = Uint8Array.from(
        atob(capture.dataUrl.slice(capture.dataUrl.indexOf(',') + 1)),
        (character) => character.charCodeAt(0),
      );
      const name = doc?.nodes.find((node) => node.id === id)?.name ?? doc?.name ?? 'Scene';
      await chooseFile(
        'save',
        async (path, overwrite) => {
          const saved = await window.mojive!.saveBinaryFile(path, bytes, overwrite);
          log(message('Saved {{path}}', { path: saved }));
        },
        { extension: '.png', caption: t(id ? 'Save camera image' : 'Save scene image'), suggestedName: `${name}.png` },
      );
    } catch (error) {
      report(error);
    }
  }
  async function exportVideo() {
    const current = state.current,
      take = current.timelineData.takes.find(
        (item) => item.id === current.timelineData.activeTakeId,
      );
    if (!current.doc || !take?.frames.length || current.recording || videoAbort.current) return;
    if (!window.mojive) {
      setNotice('Export video in the desktop app.');
      return;
    }
    const view = viewport.current?.getCameraView();
    if (!view) return;
    current.playing = false;
    setPlaying(false);
    const abort = new AbortController();
    videoAbort.current = abort;
    setVideoProgress(0);
    try {
      const blob = await exportTakeVideo(current.doc, take, view, {
        geometry,
        grid: resolvedLayers.grid,
        range: current.takeViews[take.id]?.range ?? null,
        signal: abort.signal,
        onProgress: setVideoProgress,
      });
      setVideoProgress(null);
      const bytes = new Uint8Array(await blob.arrayBuffer());
      await chooseFile(
        'save',
        async (path, overwrite) => {
          const saved = await window.mojive!.saveBinaryFile(path, bytes, overwrite);
          log(message('Saved {{path}}', { path: saved }));
        },
        { extension: '.webm', caption: t('Export video'), suggestedName: `${take.name}.webm` },
      );
    } catch (error) {
      if (!abort.signal.aborted) report(error);
    } finally {
      videoAbort.current = null;
      setVideoProgress(null);
    }
  }
  async function fileOpen(file: ProjectFile) {
    if (file.kind === 'directory') {
      setSearch('');
      return;
    }
    if (/\.(urdf|xml)$/i.test(file.name) && window.mojive) {
      await openAsset(fileAsset(file.path));
    } else
      setNotice(
        file.name.toLowerCase().endsWith('.urdf')
          ? 'Use Open URDF to load a local file.'
          : 'Mesh resources are loaded through their URDF model.',
      );
  }
  function add(kind: EntityKind) {
    if (sceneLoading.isActive() || closing.current || recordingStartWork.current || recordingStopWork.current) return;
    if (state.current.recording) {
      setNotice('Stop recording before changing scene structure.');
      return;
    }
    if (!doc) return;
    setMenu(null);
    transaction(message('Added {{kind}}', { kind: message(kind) }), () => {
      const n = createPrimitive(kind);
      const view = viewport.current?.getCameraView();
      if (view && n.object instanceof THREE.Mesh) {
        doc.root.updateWorldMatrix(true, false);
        n.object.position.copy(doc.root.worldToLocal(new THREE.Vector3().fromArray(view.target)));
      }
      doc.root.add(n.object);
      doc.nodes.push(n);
      setSelected(n.id);
    });
    log(message('Added {{kind}}', { kind: message(kind) }));
  }
  function remove() {
    if (sceneLoading.isActive() || closing.current || recordingStartWork.current || recordingStopWork.current) return;
    if (state.current.recording) {
      setNotice('Stop recording before changing scene structure.');
      return;
    }
    if (!doc || !selected) return;
    const node = doc.nodes.find((n) => n.id === selected);
    if (!node) return;
    transaction(message('Deleted {{name}}', { name: node.name }), () => {
      removeNode(doc, node.id);
      setSelected(null);
      setPinnedId((id) => (doc.nodes.some((n) => n.id === id) ? id : null));
    });
  }
  // The saved extent sizes the timeline; it never bounds recording or authored time.
  function extendDuration(time: number) {
    if (!Number.isFinite(time) || time <= state.current.duration) return;
    const extent = Math.max(time, Math.ceil(time / 10) * 10);
    state.current.duration = extent;
    setDuration(extent);
  }
  async function seek(t: number, source?: 'keyframes' | 'take', mode: 'nearest' | 'floor' = 'nearest') {
    const current = state.current;
    if (current.recording || !Number.isFinite(t)) return false;
    const epoch = ++seekEpoch.current;
    const runtime = physics.current;
    const nextSource = source ?? current.playbackSource;
    const value = Math.max(0, t);
    const take = current.timelineData.takes.find(
      (item) => item.id === current.timelineData.activeTakeId,
    );
    try {
      if (current.doc && nextSource === 'take' && take) {
        const index = takeFrameIndex(take, value, mode);
        if (index >= 0) {
          const work = seekWork.current.catch(() => undefined).then(async () => {
            if (epoch !== seekEpoch.current || state.current.doc !== current.doc) return;
            await applySceneState(current.doc!, take.frames[index], runtime);
          });
          seekWork.current = work;
          await work;
        }
      }
      if (epoch !== seekEpoch.current || state.current.doc !== current.doc) return false;
    } catch (error) {
      if (epoch !== seekEpoch.current || state.current.doc !== current.doc) return false;
      current.playing = false;
      setPlaying(false);
      report(error);
      return false;
    }
    // Publish cursor and selection only after compatibility and the complete pose succeed.
    extendDuration(value);
    state.current.time = value;
    setTime(value);
    state.current.playbackSource = nextSource;
    setPlaybackSource(nextSource);
    if (nextSource === 'take' && take) {
      const views = {
        ...current.takeViews,
        [take.id]: { cursor: value, range: current.takeViews[take.id]?.range ?? null },
      };
      state.current.takeViews = views;
      setTakeViews(views);
    }
    invalidateNoticeAction();
    setRevision((revision) => revision + 1);
    return true;
  }
  function capture() {
    const s = state.current;
    if (!s.doc) return;
    transaction('Keyframe captured', async () => {
      await physics.current?.session.capture();
      const k: Keyframe = {
        id: crypto.randomUUID(),
        name: `${t('Keyframe')} ${s.keys.length + 1}`,
        time: s.time,
        ...captureSceneState(s.doc!, physics.current),
      };
      const next = [...s.keys, k].sort((a, b) => a.time - b.time);
      setKeys(next);
      state.current.keys = next;
    });
  }
  function changeTimeline(next: WorkspaceTimeline) {
    state.current.timelineData = next;
    setTimelineData(next);
  }
  async function selectTake(id: string) {
    const current = state.current;
    if (current.recording || !current.doc) return;
    const take = current.timelineData.takes.find((item) => item.id === id);
    if (!take) return;
    const cursor = current.takeViews[id]?.cursor ?? take.frames[0]?.time ?? 0;
    try {
      const index = takeFrameIndex(take, cursor);
      if (index >= 0) await applySceneState(current.doc, take.frames[index], physics.current);
    } catch (error) {
      report(error);
      return;
    }
    current.playing = false;
    setPlaying(false);
    if (state.current.doc !== current.doc) return;
    changeTimeline({ ...state.current.timelineData, activeTakeId: id });
    state.current.time = cursor; setTime(cursor);
    state.current.playbackSource = 'take'; setPlaybackSource('take');
    setRevision((revision) => revision + 1);
  }
  function createTake() {
    if (state.current.recording) return;
    transaction('Take created', () => {
      const current = state.current,
        take = {
          id: crypto.randomUUID(),
          name: `${t('Take')} ${current.timelineData.takes.length + 1}`,
          frames: [],
        };
      changeTimeline({
        ...current.timelineData,
        takes: [...current.timelineData.takes, take],
        activeTakeId: take.id,
      });
      setTime(0);
      current.time = 0;
    });
  }
  function deleteTake(id: string) {
    if (state.current.recording) return;
    transaction('Take deleted', () => {
      const current = state.current,
        takes = current.timelineData.takes.filter((item) => item.id !== id);
      const activeTakeId =
        current.timelineData.activeTakeId === id
          ? (takes[0]?.id ?? null)
          : current.timelineData.activeTakeId;
      changeTimeline({ ...current.timelineData, takes, activeTakeId });
      if (activeTakeId) selectTake(activeTakeId);
    });
  }
  function changeTakeRange(range: [number, number] | null) {
    const current = state.current,
      take = current.timelineData.takes.find(
        (item) => item.id === current.timelineData.activeTakeId,
      );
    if (!take || current.recording) return;
    const last = Math.max(0, take.frames.length - 1);
    const bounded: [number, number] | null = range
      ? [
          Math.max(0, Math.min(last, Math.min(...range))),
          Math.max(0, Math.min(last, Math.max(...range))),
        ]
      : null;
    const views = {
      ...current.takeViews,
      [take.id]: { cursor: current.takeViews[take.id]?.cursor ?? current.time, range: bounded },
    };
    current.takeViews = views;
    setTakeViews(views);
  }
  function captureSnapshot() {
    transaction('Snapshot captured', async () => {
      await physics.current?.session.capture();
      const current = state.current;
      const item: SceneSnapshot = {
        id: crypto.randomUUID(),
        name: `${t('Snapshot')} ${current.timelineData.snapshots.length + 1}`,
        time: current.time,
        ...captureSceneState(current.doc!, physics.current),
      };
      changeTimeline({
        ...current.timelineData,
        snapshots: [...current.timelineData.snapshots, item].sort((a, b) => a.time - b.time),
      });
    });
  }
  function loadSnapshot(id: string) {
    const item = state.current.timelineData.snapshots.find((item) => item.id === id);
    if (!item || !state.current.doc || state.current.recording) return;
    try {
      assertPoseCompatible(state.current.doc!, item.pose);
    } catch (error) {
      report(error);
      return;
    }
    transaction('Snapshot loaded', async () => {
      await applySceneState(state.current.doc!, item, physics.current);
      setTime(item.time);
      state.current.time = item.time;
    });
  }
  function recordingTime(now: number) {
    const clock = recordClock.current!;
    return clock.time + (clock.simulationStart === undefined
      ? (now - clock.start) / 1000
      : Math.max(0, (physics.current?.session.current.frame?.time ?? clock.simulationStart) - clock.simulationStart));
  }
  async function stopRecording() {
    if (recordingStopWork.current) return recordingStopWork.current;
    const finish = async () => {
      // Starting may be awaiting a Worker restore/capture while recording is still false.
      await recordingStartWork.current;
      if (!state.current.recording) return;
      const runtime = physics.current;
      runtime?.session.pause();
      try {
        // Pausing cancels future ticks; the last Worker step must settle before
        // the terminal frame and history entry are captured.
        await runtime?.session.capture();
        const current = state.current, clock = recordClock.current;
        const take = current.timelineData.takes.find((item) => item.id === clock?.id);
        if (take?.frames.length && current.doc && clock) {
          const time = recordingTime(performance.now());
          if (time > take.frames[take.frames.length - 1].time)
            take.frames.push({ time, ...captureSceneState(current.doc, runtime) });
          setTime(time);
          current.time = time;
          extendDuration(time);
          changeTimeline({ ...current.timelineData });
        }
      } finally {
        // A failed barrier must stop the recording clock too. Keep the frames
        // and pending transaction intact, and propagate the error to the caller.
        recordClock.current = null;
        state.current.recording = false;
        setRecording(false);
      }
      commit();
      await historyWork.current;
    };
    recordingStopWork.current = finish();
    try { await recordingStopWork.current; }
    finally { recordingStopWork.current = null; }
  }
  async function toggleRecording() {
    if (sceneLoading.isActive() || closing.current || recordingStartWork.current || recordingStopWork.current) return;
    if (state.current.recording) {
      await stopRecording();
      return;
    }
    const current = state.current, document = current.doc, runtime = physics.current;
    if (!document) return;
    const previous = current.timelineData.takes.find(
      (item) => item.id === current.timelineData.activeTakeId,
    );
    const resume = previous?.frames.reduce<(typeof previous.frames)[number] | null>(
      (nearest, frame) =>
        !nearest || Math.abs(frame.time - current.time) < Math.abs(nearest.time - current.time)
          ? frame
          : nearest,
      null,
    );
    try {
      if (resume) assertPoseCompatible(document, resume.pose);
    } catch (error) {
      report(error);
      return;
    }
    begin(true, true);
    const start = resume?.time ?? current.time;
    const baseline = pending.current;
    const ownsDocument = () => state.current.doc === document && physics.current === runtime;
    const rollbackStart = async () => {
      if (pending.current !== baseline || !ownsDocument()) return;
      pending.current = null;
      await historyWork.current;
      if (resume && baseline && ownsDocument()) await restore(baseline);
    };
    const startRecording = async () => {
      try {
        if (resume) await applySceneState(document, resume, runtime);
        await runtime?.session.capture();
        if (!ownsDocument() || sceneLoading.isActive() || closing.current) {
          await rollbackStart();
          return;
        }
        setTime(start);
        state.current.time = start;
        extendDuration(start);
        const take = {
          id: previous?.id ?? crypto.randomUUID(),
          name: previous?.name ?? `${t('Take')} ${current.timelineData.takes.length + 1}`,
          frames: [
            ...(previous?.frames.filter((frame) => frame.time < start) ?? []),
            { time: start, ...captureSceneState(document, runtime) },
          ],
        };
        changeTimeline({
          ...current.timelineData,
          takes: previous
            ? current.timelineData.takes.map((item) => (item.id === take.id ? take : item))
            : [...current.timelineData.takes, take],
          activeTakeId: take.id,
        });
        const views = { ...current.takeViews, [take.id]: { cursor: start, range: null } };
        current.takeViews = views;
        setTakeViews(views);
        recordClock.current = {
          start: performance.now(),
          time: start,
          id: take.id,
          simulationStart: runtime?.session.current.frame?.time,
        };
        runtime?.session.play();
        state.current.recording = true;
        setRecording(true);
        state.current.playing = false;
        setPlaybackSource('take');
        state.current.playbackSource = 'take';
        setDirty(true);
      } catch (error) {
        await rollbackStart();
        throw error;
      }
    };
    const work = startRecording();
    recordingStartWork.current = work;
    try { await work; }
    catch (error) { report(error); }
    finally { if (recordingStartWork.current === work) recordingStartWork.current = null; }
  }
  function loadKey(id: string) {
    const key = state.current.keys.find((key) => key.id === id),
      current = state.current;
    if (!key || !current.doc || current.recording) return;
    try {
      assertPoseCompatible(current.doc, key.pose);
    } catch (error) {
      report(error);
      return;
    }
    transaction('Keyframe loaded', async () => {
      await applySceneState(current.doc!, key, physics.current);
      setTime(key.time);
      current.time = key.time;
    });
  }
  function changeKeys(next: Keyframe[]) {
    extendDuration(next.reduce((end, key) => Math.max(end, key.time), 0));
    setKeys(next);
    state.current.keys = next;
    setDirty(true);
  }
  function jointChange(joint: Joint, value: number) {
    if (!doc) return;
    setJoint(doc, joint.id, value);
    changed();
  }
  function toggleExpanded(id: string) {
    setExpanded((prev) => {
      const n = new Set(prev);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  }
  useEffect(() => {
    void runSceneLoad(sampleAsset('joint_gizmo').name, async (options) => {
      const next = await loadAsset(sampleAsset('joint_gizmo'), options);
      await replaceDocument(next.document, next.runtime?.keyframes ?? [], 10, undefined, next.runtime, options.signal);
    });
    return sceneLoading.cancel;
  }, []);
  useEffect(() => {
    if (playing || recording || busy) invalidateNoticeAction();
  }, [playing, recording, busy]);
  useEffect(() => {
    if (!playing && !recording) return;
    let last = performance.now(),
      raf = 0,
      cursorTime = state.current.time;
    let cancelled = false;
    const tick = async (now: number) => {
      const current = state.current;
      if (!current.playing && !current.recording) return;
      const previous = last;
      last = now;
      if (current.recording) {
        const next = recordingTime(now);
        const take = current.timelineData.takes.find(
          (item) => item.id === current.timelineData.activeTakeId,
        );
        if (take && current.doc && next > take.frames[take.frames.length - 1].time) {
          take.frames.push({ time: next, ...captureSceneState(current.doc, physics.current) });
          changeTimeline({ ...current.timelineData });
        }
        setTime(next);
        current.time = next;
        extendDuration(next);
      } else {
        const take = current.timelineData.takes.find(
          (item) => item.id === current.timelineData.activeTakeId,
        );
        if (!take || take.frames.length < 2) {
          setPlaying(false);
          return;
        }
        const frames = take.frames,
          range = current.takeViews[take.id]?.range;
        const firstIndex = Math.min(frames.length - 1, range?.[0] ?? 0),
          lastIndex = Math.min(frames.length - 1, range?.[1] ?? frames.length - 1);
        const start = frames[firstIndex].time,
          end = frames[lastIndex].time;
        if (lastIndex <= firstIndex) {
          await seek(start, 'take', 'floor');
          setPlaying(false);
          return;
        }
        const tail = end - frames[lastIndex - 1].time,
          period = end - start + tail;
        const desired = cursorTime + ((now - previous) / 1000) * current.playbackSpeed;
        let cursor = Math.max(start, desired);
        if (cursor >= end + tail) {
          if (current.loop) cursor = start + ((cursor - start) % period);
          else {
            await seek(end, 'take', 'floor');
            setPlaying(false);
            return;
          }
        }
        cursorTime = cursor;
        await seek(cursor, 'take', 'floor');
      }
      if (!cancelled) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { cancelled = true; cancelAnimationFrame(raf); };
  }, [playing, recording]);
  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      if (e.defaultPrevented) return;
      if (replacePrompt) {
        if (e.key === 'Escape') answerReplacement(false);
        return;
      }
      if (sceneLoading.isActive() || closing.current || recordingStartWork.current || recordingStopWork.current || settingsOpen || fileDialog || about || help || assetPreview || videoProgress !== null) return;
      const target = e.target as HTMLElement;
      if (
        target.closest(
          'input,textarea,[contenteditable],[role=dialog],[role=listbox],[role=combobox],[role=menu]',
        )
      )
        return;
      if (target.closest('button') && (e.key === ' ' || e.key === 'Enter')) return;
      const command = commandKey(e) && e.key !== 'Control';
      const k = e.key.toLowerCase();
      if (nativeMenu && command && ['o', 's', 'z', ','].includes(k)) return;
      if (command && k === ',') {
        e.preventDefault();
        setSettingsOpen(true);
      } else if (command && k === 'o') {
        e.preventDefault();
        openSaved();
      } else if (command && e.shiftKey && k === 's') {
        e.preventDefault();
        save();
      } else if (command && k === 'z') {
        e.preventDefault();
        history(e.shiftKey ? 'redo' : 'undo');
      } else if (
        matchesKey(settings.inputBindings, 'focus', e) ||
        keyboardChord(e) === focusAllAccelerator(settings.inputBindings)
      ) {
        e.preventDefault();
        viewport.current?.focus(
          keyboardChord(e) === focusAllAccelerator(settings.inputBindings)
            ? undefined
            : (selected ?? undefined),
        );
      } else if (matchesKey(settings.inputBindings, 'playback', e)) {
        e.preventDefault();
        if (e.repeat) return;
        if (physics.current) {
          toggleSimulation();
          return;
        }
        if (
          timelineData.takes.some(
            (take) => take.id === timelineData.activeTakeId && take.frames.length > 1,
          ) &&
          !recording
        ) {
          setPlaybackSource('take');
          state.current.playbackSource = 'take';
          setPlaying((p) => !p);
        }
      } else if (matchesKey(settings.inputBindings, 'previousFrame', e)) {
        if (!physics.current) return;
        e.preventDefault();
        if (!e.repeat) previousSimulationFrame();
      } else if (matchesKey(settings.inputBindings, 'stepFrame', e)) {
        if (!physics.current) return;
        e.preventDefault();
        if (!e.repeat) stepSimulationFrame();
      } else if (matchesKey(settings.inputBindings, 'capture', e)) {
        capture();
      } else if (k === 'delete' || k === 'backspace') {
        e.preventDefault();
        remove();
      } else if (k === 'escape') {
        setMenu(null);
        setHelp(false);
        setSettingsOpen(false);
        if (pending.current) {
          const s = pending.current;
          pending.current = null;
          void historyWork.current.then(() => restore(s)).catch(report);
        }
      } else {
        if (matchesKey(settings.inputBindings, 'frame', e))
          setFrame((value) => (value === 'world' ? 'body' : 'world'));
        const t = tools.find((t) => matchesKey(settings.inputBindings, t.id, e));
        if (t) setTool(t.id);
      }
    };
    window.addEventListener('keydown', listener);
    return () => window.removeEventListener('keydown', listener);
  });
  const menuBlocked = Boolean(busy || fileDialog || replacePrompt || settingsOpen || about || help || assetPreview || videoProgress !== null);
  function dispatchMenuAction(action: string) {
    if ((sceneLoading.isActive() || closing.current || recordingStartWork.current || recordingStopWork.current || menuBlocked) && action !== 'app.requestClose') return;
    if (action.startsWith('scene.')) { loadCatalog(action.slice(6)); return; }
    if (action.startsWith('add.')) { add(action.slice(4) as EntityKind); return; }
    if (action.startsWith('tool.')) { setTool(action.slice(5) as Tool); return; }
    if (action.startsWith('frame.')) { setFrame(action.slice(6) as Frame); return; }
    if (action.startsWith('simulation.speed.')) {
      physics.current?.session.setSpeed(Number(action.slice('simulation.speed.'.length)));
      return;
    }
    if (action.startsWith('panel.')) {
      const panel = action.slice(6);
      if (panel === 'output' || panel === 'timeline') {
        setShowBottom(bottom === panel ? !showBottom : true);
        setBottom(panel);
      } else { setLeft(panel); setShowLeft(true); }
      return;
    }
    if (action.startsWith('theme.')) {
      updateSettings({ appearance: action.slice(6) as 'dark' | 'light' }); return;
    }
    if (action.startsWith('locale.')) {
      updateSettings({ language: action === 'locale.en' ? 'en' : 'zh' }); return;
    }
    const actions: Record<string, () => void> = {
      'file.openUrdf': openUrdf,
      'file.openSimulation': openSimulation,
      'file.openProject': openProject,
      'file.closeProject': closeProject,
      'file.openScene': openSaved,
      'file.save': save,
      'edit.undo': () => history('undo'),
      'edit.redo': () => history('redo'),
      'edit.delete': remove,
      'edit.visibility': () => {
        const node = doc?.nodes.find((node) => node.id === selected);
        if (!node) return;
        transaction(message(node.visible ? 'Hid {{name}}' : 'Showed {{name}}', { name: node.name }), () => {
          node.visible = !node.visible;
          node.object.visible = node.visible;
        });
      },
      'app.settings': () => setSettingsOpen(true),
      'app.requestClose': requestClose,
      'app.about': () => setAbout(true),
      'settings.open': () => setSettingsOpen(true),
      'help.interaction': () => setHelp(true),
      'help.documentation': () => { window.open('https://github.com/acrlw/mojive#readme', '_blank', 'noopener,noreferrer'); },
      'view.focusSelection': () => viewport.current?.focus(selected ?? undefined),
      'view.focusAll': () => viewport.current?.focus(),
      'view.captureScene': () => { void captureCamera(); },
      'view.perspective': () => { setProjection('perspective'); viewport.current?.view('perspective'); },
      'view.orthographic': () => setProjection('orthographic'),
      'view.toggleProjection': () => setProjection((p) => p === 'perspective' ? 'orthographic' : 'perspective'),
      'view.toggleLeft': () => setShowLeft((v) => !v),
      'view.toggleInspector': () => setShowInspector((v) => !v),
      'view.toggleTimeline': () => { setBottom('timeline'); setShowBottom(bottom === 'timeline' ? !showBottom : true); },
      'view.toggleGizmos': () => changeLayers({ ...resolvedLayers, gizmos: !resolvedLayers.gizmos }),
      'simulation.toggle': toggleSimulation,
      'simulation.previous': previousSimulationFrame,
      'simulation.step': stepSimulationFrame,
      'simulation.reset': () => simulationTransaction({ type: 'reset' }),
      'simulation.record': toggleRecording,
      'simulation.environment': () => { setSelected(ENVIRONMENT_ID); setPinnedId(null); setInspectorTab('Properties'); setShowInspector(true); },
    };
    if (actions[action]) actions[action]();
    else if (action.startsWith('view.')) viewport.current?.view(action.slice(5));
  }
  useEffect(() => window.mojive?.onMenuAction(dispatchMenuAction));
  const removableSelection = useMemo(() => !!doc && !!selected && canRemoveNode(doc, selected), [doc, selected, revision]);
  const structureBlocked = busy || recording || !!recordingStartWork.current || !!recordingStopWork.current || closing.current;
  const menuState: NativeMenuState = {
    locale: settings.language === 'zh' ? 'zh-CN' : 'en',
    theme: document.documentElement.dataset.theme === 'light' ? 'light' : 'dark',
    blocked: menuBlocked,
    canUndo: undo.current.length > 0,
    canRedo: redo.current.length > 0,
    hasDocument: !!doc,
    canAdd: !!doc && !structureBlocked,
    canDelete: removableSelection && !structureBlocked,
    hasSelection: !!selected,
    selectionVisible: doc?.nodes.find((node) => node.id === selected)?.visible ?? false,
    hasProject: !!project,
    showLeft, showInspector,
    showTimeline: showBottom && bottom === 'timeline',
    showOutput: showBottom && bottom === 'output',
    panel: showLeft ? left : '',
    projection, tool, frame,
    showGizmos: resolvedLayers.gizmos,
    hasSimulation: !!simulation?.source,
    simulationPlaying: !!simulation?.playing,
    simulationRecording: recording,
    simulationSpeed: simulation?.speed ?? 1,
    canToggleSimulation: !!simulation?.source && !busy && !recording,
    canSimulationPrevious: !!simulation?.source && !busy && !recording && !simulation.playing && simulation.previousFrames > 0,
    canSimulationStep: !!simulation?.source && !busy && !recording && !simulation.playing,
    sceneId,
    focusKey: focusAccelerator(settings.inputBindings),
    focusAllKey: focusAllAccelerator(settings.inputBindings),
  };
  const menuStateKey = JSON.stringify(menuState);
  useEffect(() => {
    const update = () => window.mojive?.updateMenuState({
      ...menuState,
      textInput: !!document.activeElement?.matches('input,textarea,[contenteditable]'),
      scopedKeyboard: !!document.activeElement?.closest('.timeline'),
    }).catch(report);
    update();
    window.addEventListener('focusin', update);
    window.addEventListener('focusout', update);
    return () => {
      window.removeEventListener('focusin', update);
      window.removeEventListener('focusout', update);
    };
  }, [menuStateKey]);
  useEffect(() => {
    const before = (e: BeforeUnloadEvent) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', before);
    return () => window.removeEventListener('beforeunload', before);
  }, [dirty]);
  useEffect(() => {
    if (!doc || !selected) return;
    const parents: string[] = [];
    let n = doc.nodes.find((n) => n.id === selected);
    while (n?.parentId) {
      parents.push(n.parentId);
      n = doc.nodes.find((p) => p.id === n!.parentId);
    }
    setExpanded((current) => new Set([...current, ...parents]));
  }, [selected, doc]);
  const selectedNode = doc?.nodes.find((n) => n.id === selected);
  const node = doc?.nodes.find((n) => n.id === (pinnedId ?? selected));
  const rotation = node ? new THREE.Euler().setFromQuaternion(node.object.quaternion, 'ZYX') : null;
  const joint = doc?.joints.find((j) => j.id === node?.id);
  const movable = doc?.joints.filter((j) => j.type !== 'fixed') ?? [];
  const rootNodes = doc?.nodes.filter((n) => !n.parentId) ?? [];
  const filtering = !!search.trim() || typeFilter !== 'all';
  const filteredNodes = filterSceneNodes(doc?.nodes ?? [], search, typeFilter);
  const categoryIcons: Record<string, string> = {
    all: 'tree',
    robot: 'robot',
    link: 'body',
    geom: 'geom',
    joint: 'joint',
    camera: 'camera',
    light: 'light',
  };
  const availableTypes = ['all', 'robot', 'link', 'geom', 'joint', 'camera', 'light']
    .map((kind) => ({
      kind,
      count:
        kind === 'all'
          ? (doc?.nodes.length ?? 0)
          : (doc?.nodes.filter((n) => nodeCategory(n) === kind).length ?? 0),
    }))
    .filter((f) => f.kind === 'all' || f.count > 0);
  const logicalWidth = windowSize.width / settings.uiScale;
  const logicalHeight = windowSize.height / settings.uiScale;
  const leftMaximum = Math.min(600, logicalWidth * 0.32);
  const rightMaximum = Math.min(650, logicalWidth * 0.36);
  const bottomMaximum = logicalHeight * 0.55;
  const leftWidth = Math.min(layout.left, leftMaximum);
  const rightWidth = Math.min(layout.right, rightMaximum);
  const bottomHeight = Math.min(layout.bottom, bottomMaximum);

  function treeNode(n: SceneNode, depth = 0): React.ReactNode {
    const children = filtering ? [] : doc!.nodes.filter((c) => c.parentId === n.id);
    const open = expanded.has(n.id);
    return (
      <div key={n.id}>
        <div
          className={`tree-row type-${nodeCategory(n)} ${selected === n.id ? 'selected' : ''}`}
          style={{ paddingLeft: filtering ? 8 : 8 + depth * 12 }}
          role="treeitem"
          aria-selected={selected === n.id}
          aria-expanded={children.length ? open : undefined}
          tabIndex={selected === n.id ? 0 : -1}
          onContextMenu={(event) => {
            event.preventDefault();
            event.stopPropagation();
            openTreeContext(n, event.currentTarget, event.clientX, event.clientY);
          }}
          onKeyDown={(event) => {
            if (event.key === 'ContextMenu' || (event.shiftKey && event.key === 'F10')) {
              event.preventDefault();
              event.stopPropagation();
              const box = event.currentTarget.getBoundingClientRect();
              openTreeContext(n, event.currentTarget, box.left + 16, box.bottom);
            }
          }}
          onClick={() => setSelected(n.id)}
          onDoubleClick={() => viewport.current?.focus(n.id)}
          data-tooltip={n.name}
          data-tooltip-overflow=".tree-label"
          aria-label={n.name}
        >
          {children.length > 0 ? (
            <button
              className="tree-chevron"
              aria-label={t(open ? 'Collapse {{name}}' : 'Expand {{name}}', { name: n.name })}
              onClick={(e) => {
                e.stopPropagation();
                toggleExpanded(n.id);
              }}
            >
              <Icon name={open ? 'down' : 'right'} size={14} />
            </button>
          ) : (
            <span className="tree-chevron" />
          )}
          <Icon
            name={
              n.kind === 'joint'
                ? jointIcon(doc!.joints.find((j) => j.id === n.id)?.type ?? '')
                : (kinds[n.kind] ?? 'cube')
            }
            size={16}
          />
          <span className={`tree-label ${n.visible ? '' : 'dim'}`}>{n.name}</span>
          <button
            className="tree-visibility"
            data-tooltip={t(n.visible ? 'Hide object' : 'Show object')}
            aria-label={t(n.visible ? 'Hide object' : 'Show object')}
            aria-pressed={!n.visible}
            onDoubleClick={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation();
              transaction(
                message(n.visible ? 'Hid {{name}}' : 'Showed {{name}}', { name: n.name }),
                () => {
                  n.visible = !n.visible;
                  n.object.visible = n.visible;
                },
              );
            }}
          >
            <Icon name={n.visible ? 'eye' : 'eye-off'} size={16} />
          </button>
        </div>
        {children.length > 0 &&
          open &&
          !filtering &&
          children.map((c) => treeNode(c, depth + (n.kind === 'joint' ? 0 : 1)))}
      </div>
    );
  }
  function menuItem(
    label: string,
    action: () => void,
    icon?: string,
    shortcut?: string,
    disabled = false,
  ) {
    return (
      <button
        role="menuitem"
        aria-label={t(label)}
        aria-description={shortcut}
        disabled={disabled}
        onClick={() => {
          setMenu(null);
          action();
        }}
      >
        {icon ? <Icon name={icon} size={16} /> : <span className="menu-icon-space" />}
        <span>{t(label)}</span>
        {shortcut && <kbd>{shortcut}</kbd>}
      </button>
    );
  }
  const leftTitle =
    left === 'scene'
      ? 'Scene'
      : left === 'joints'
        ? 'Joints'
        : left === 'actuators'
          ? 'Actuators'
          : left === 'layers'
            ? 'Layers'
            : left === 'cameras'
              ? 'Cameras'
              : 'Project';
  return (
    <div
      className="app-shell"
      onDragOver={(event) => {
        if (event.dataTransfer.types.includes('Files')) event.preventDefault();
      }}
      onDrop={async (event) => {
        if (!event.dataTransfer.files.length) return;
        event.preventDefault();
        if (fileDialog || replacePrompt || videoAbort.current || assetPreview || sceneLoading.isActive()) return;
        const file = event.dataTransfer.files[0];
        const resources = file.name.toLowerCase().endsWith('.json') ? null : droppedFiles(event.dataTransfer);
        await runSceneLoad(file.name, async (options) => {
          if (file.name.toLowerCase().endsWith('.json')) {
            const text = await file.text();
            const scene = await loadPhysicsWorkspace(text, physicsBackend, options);
            await replaceDocument(scene.doc, scene.keys, scene.duration, scene.timeline, scene.runtime, options.signal);
          } else {
            const bundles = await browserBundles(await resources!);
            options.signal?.throwIfAborted();
            const assets: Asset[] = bundles.map((bundle) => ({ id: `upload:${bundle.id}`, name: bundle.name,
              format: bundle.entry.toLowerCase().endsWith('.urdf') ? 'URDF' : 'MJCF', source: { kind: 'bundle', bundle } }));
            setUploadedAssets((previous) => [...new Map([...previous, ...assets].map((asset) => [asset.id, asset])).values()]);
            if (assets.length === 1) {
              const next = await loadAsset(assets[0], options);
              if (await replaceDocument(next.document, next.runtime?.keyframes ?? [], 10, undefined, next.runtime, options.signal))
                setAssetLibrary((library) => rememberAsset(library, assets[0]));
            } else { setLeft('assets'); setShowLeft(true); }
          }
        });
      }}
      style={
        {
          zoom: settings.uiScale,
          width: logicalWidth,
          height: logicalHeight,
          '--left-width': showLeft ? `${leftWidth}px` : '0px',
          gridTemplateColumns: `48px ${showLeft ? leftWidth + 'px' : '0px'} minmax(120px,1fr) ${showInspector ? rightWidth + 'px' : '0px'}`,
          gridTemplateRows: `36px minmax(120px,1fr) ${showBottom ? bottomHeight + 'px' : '0px'} 28px`,
        } as React.CSSProperties
      }
    >
      <header className={`titlebar ${nativeMenu ? 'native-titlebar' : ''}`}>
        {nativeMenu && <div className="window-traffic-spacer" />}
        {!nativeMenu && (
          <nav
            ref={menuBar}
            aria-label={t('Application menu')}
            onPointerLeave={() => {
              hoveredMenu.current = null;
            }}
          >
            {menus.map((m) => (
              <button
                key={m}
                className={menu === m ? 'active' : ''}
                data-menu={m}
                aria-haspopup="menu"
                aria-expanded={menu === m}
                onPointerEnter={(event) => {
                  if (menu && menus.includes(menu) && menu !== m) {
                    activateMenu(m, event.currentTarget);
                    hoveredMenu.current = m;
                  }
                }}
                onClick={(event) => {
                  menuTrigger.current = event.currentTarget;
                  setMenu(menu === m && hoveredMenu.current !== m ? null : m);
                  hoveredMenu.current = null;
                }}
                onKeyDown={(event) => {
                  if (['ArrowDown', 'ArrowUp'].includes(event.key)) {
                    event.preventDefault();
                    activateMenu(m, event.currentTarget);
                  } else if (['ArrowLeft', 'ArrowRight'].includes(event.key)) {
                    event.preventDefault();
                    const next =
                      menus[
                        (menus.indexOf(m) + (event.key === 'ArrowLeft' ? -1 : 1) + menus.length) %
                          menus.length
                      ];
                    menuBar.current
                      ?.querySelector<HTMLButtonElement>(`[data-menu="${next}"]`)
                      ?.focus();
                  }
                }}
              >
                {workspaceMenuLabel(workspaceMenus.find((item) => item.name === m)!.label, menuState.locale)}
              </button>
            ))}
          </nav>
        )}
        <div className="document-title">
          <Icon name="file" size={16} />
          {doc ? t(sceneDisplayName(doc)) : t('Untitled scene')}
          {dirty && <span className="dirty-dot" />}
        </div>
        <div className="title-actions">
          <Button
            icon={document.documentElement.dataset.theme === 'light' ? 'moon' : 'sun'}
            title={t('Toggle theme')}
            onClick={() =>
              updateSettings({
                appearance: document.documentElement.dataset.theme === 'light' ? 'dark' : 'light',
              })
            }
          />
          <Button icon="settings" title={t('Settings')} onClick={() => setSettingsOpen(true)} />

          <Button
            icon="sidebar-left"
            title={t('Toggle left panel')}
            active={showLeft}
            onClick={() => setShowLeft(!showLeft)}
          />
          <Button
            icon="panel-right"
            title={t('Toggle Inspector')}
            active={showInspector}
            onClick={() => setShowInspector(!showInspector)}
          />
        </div>
      </header>
      <aside className="activity-rail">
        {[
          { id: 'scene', icon: 'scene', title: 'Scene' },
          { id: 'project', icon: 'folder', title: 'Project files' },
          { id: 'assets', icon: 'asset-library', title: 'Assets' },
          { id: 'joints', icon: 'pose', title: 'Joints' },
          { id: 'actuators', icon: 'control', title: 'Actuators' },
          { id: 'cameras', icon: 'camera', title: 'Cameras' },
          { id: 'layers', icon: 'layers', title: 'Layers' },
        ].map((item) => (
          <Button
            key={item.id}
            icon={item.icon}
            title={item.title}
            active={left === item.id && showLeft}
            activityIndicator={left === item.id && showLeft}
            onClick={() => {
              setLeft(item.id);
              setShowLeft(true);
              setSearch('');
            }}
          />
        ))}
        <span className="flex-spacer" />
        <Button
          icon="key"
          title={t('Timeline')}
          active={showBottom && bottom === 'timeline'}
          activityIndicator={showBottom && bottom === 'timeline'}
          onClick={() => {
            setBottom('timeline');
            setShowBottom(!(showBottom && bottom === 'timeline'));
          }}
        />
        <Button
          icon="terminal"
          title={t('Output')}
          active={showBottom && bottom === 'output'}
          activityIndicator={showBottom && bottom === 'output'}
          onClick={() => {
            setBottom('output');
            setShowBottom(true);
          }}
        />
        <Button icon="settings" title={t('Settings')} onClick={() => setSettingsOpen(true)} />
      </aside>
      <aside className="left-panel panel" style={{ visibility: showLeft ? 'visible' : 'hidden' }}>
        {!['joints', 'cameras', 'actuators', 'assets'].includes(left) && (
          <div className="panel-header">
            <strong>{t(leftTitle)}</strong>
            <span className="flex-spacer" />
            {left === 'layers' ? (
              <Button
                icon="reset"
                title={t('Reset viewport layers')}
                onClick={() => changeLayers({ ...DEFAULT_VIEWPORT_LAYERS })}
              />
            ) : left === 'project' ? (
              <>
                <Button icon="folder" title={t('Open project folder')} onClick={openProject} />
                <Button
                  icon="close"
                  title={t('Close project folder')}
                  disabled={!project}
                  onClick={closeProject}
                />
              </>
            ) : (
              <>
                <Button
                  icon="search"
                  title={t('Search')}
                  active={searchVisible}
                  onClick={() => {
                    setSearchVisible(!searchVisible);
                    setSearch('');
                  }}
                />
                <Button
                  icon="plus"
                  title={t('Add object')}
                  onClick={(event) => {
                    menuTrigger.current = event.currentTarget;
                    setMenu(menu === 'Add' ? null : 'Add');
                  }}
                />
              </>
            )}
          </div>
        )}
        {!['joints', 'cameras', 'actuators', 'assets'].includes(left) &&
          (left === 'project' ? !!project : searchVisible) && (
            <SearchField
              className="panel-search"
              label={t(`Search ${leftTitle}`)}
              value={search}
              onChange={setSearch}
            />
          )}
        {left === 'scene' && (
          <div className="scene-controls">
            <div className="type-filters" aria-label={t('Object types')}>
              {availableTypes.map(({ kind, count }) => (
                <button
                  key={kind}
                  data-tooltip={t(
                    kind === 'light' ? 'Light source' : kind[0].toUpperCase() + kind.slice(1),
                  )}
                  aria-label={t(
                    kind === 'light' ? 'Light source' : kind[0].toUpperCase() + kind.slice(1),
                  )}
                  aria-pressed={typeFilter === kind}
                  className={`filter-chip type-${kind} ${typeFilter === kind ? 'active' : ''}`}
                  data-capsule-layout=""
                  onClick={() => setTypeFilter(kind)}
                >
                  <Icon name={categoryIcons[kind]} size={16} />
                  <span>{count}</span>
                </button>
              ))}
            </div>
          </div>
        )}
        <div
          className={`left-content ${['joints', 'cameras', 'actuators'].includes(left) ? 'joint-panel-host' : ''}`}
        >
          {left === 'scene' && (
            <>
              <div role="tree" aria-label={t('Scene hierarchy')}>
                {(filtering ? filteredNodes : rootNodes).map((n) => treeNode(n))}
              </div>
              {doc && !doc.nodes.length ? (
                <Empty
                  title={t('Your scene is empty')}
                  action={
                    <Button
                      icon="plus"
                      onClick={(event) => activateMenu('Add', event.currentTarget)}
                    >
                      {t('Add object')}
                    </Button>
                  }
                />
              ) : filtering && !filteredNodes.length ? (
                <Empty
                  title={t('No matching objects')}
                  action={
                    <Button
                      onClick={() => {
                        setSearch('');
                        setTypeFilter('all');
                      }}
                    >
                      {t('Clear filters')}
                    </Button>
                  }
                />
              ) : null}
            </>
          )}
          {left === 'assets' && <AssetBrowser library={assetLibrary} uploads={uploadedAssets} project={project}
            onPreview={setAssetPreview} onOpen={openAsset} onFavorite={toggleFavorite} onOpenProject={openProject} />}
          {left === 'project' && (
            <>
              {project && (
                <div className="group-eyebrow" data-tooltip={project.path}>
                  {project.name}
                </div>
              )}
              {project ? (
                projectFiles.length ? (
                  <ProjectExplorer
                    key={project.path}
                    project={project}
                    query={search}
                    onOpen={fileOpen}
                    onClearSearch={() => setSearch('')}
                  />
                ) : (
                  <Empty title={t('This folder is empty')} />
                )
              ) : (
                <Empty
                  title={t('No folder open')}
                  action={
                    <Button icon="folder" onClick={openProject}>
                      {t('Open folder')}
                    </Button>
                  }
                />
              )}
            </>
          )}
          {left === 'joints' && doc && (
            <JointPanel
              doc={doc}
              selectedId={selected}
              activeJointId={activeJoint}
              onSelect={setSelected}
              onFocus={(id) => viewport.current?.focus(id)}
              onBegin={beginJoint}
              onCommit={commit}
              onCancel={cancelJoint}
              onChange={jointChanged}
            />
          )}
          {left === 'cameras' && (
            <CamerasPanel
              doc={doc}
              selectedId={selected}
              activeCameraId={sceneCamera.id}
              piloting={sceneCamera.pilot}
              onSelect={setSelected}
              onView={useCamera}
              onFocus={(id) => viewport.current?.focus(id)}
              onAlign={alignCamera}
              onCreate={createCamera}
              onDelete={deleteCamera}
              onCapture={captureCamera}
              onPreset={(preset) => viewport.current?.view(preset)}
            />
          )}
          {left === 'layers' && (
            <LayersPanel value={resolvedLayers} onChange={changeLayers} translate={t} />
          )}
          {left === 'actuators' && (
            <ActuatorsPanel
              doc={doc}
              selectedId={selected}
              onSelect={setSelected}
              onFocus={(id) => viewport.current?.focus(id)}
              controls={simulation?.source?.capabilities.includes('controls') && simulation.frame ? {
                actuators: simulation.source.actuators,
                values: simulation.frame.controls,
                onBegin: beginPhysics,
                onCommit: commit,
                onChange: (id, value) => simulationCommand({ type: 'controls', values: [{ id, value }] }),
              } : undefined}
            />
          )}
        </div>
      </aside>
      <main className="viewport-panel">
        <div className={`viewport-topbar${simulation?.source ? ' has-simulation' : ''}`}>
          <div className="viewport-navigation">
            <Button
              icon={projection}
              title={t('Projection')}
              onClick={() => {
                setLayersOpen(false);
                setMenu(menu === 'Projection' ? null : 'Projection');
              }}
            >
              <span className="projection-caption">
                {t(projection === 'perspective' ? 'Perspective' : 'Orthographic')}
              </span>
              <Icon name="down" size={14} />
            </Button>
            <Button icon="fit" title={t('Frame all')} onClick={() => viewport.current?.focus()} />
            <Button
              icon="layers"
              title={t('Layers')}
              active={layersOpen}
              onClick={() => {
                setMenu(null);
                setLayersOpen(!layersOpen);
              }}
            />
          </div>
          {simulation?.source && <SimulationToolbar state={simulation} recording={recording} bindings={settings.inputBindings}
          onToggle={toggleSimulation}
          onPrevious={previousSimulationFrame}
          onStep={stepSimulationFrame}
          onReset={() => simulationTransaction({ type: 'reset' })}
          onSpeed={(speed) => physics.current?.session.setSpeed(speed)}
          onEnvironment={() => { setSelected(ENVIRONMENT_ID); setPinnedId(null); setInspectorTab('Properties'); setShowInspector(true); }} />}
          {!simulation?.source && <span className="flex-spacer" />}
          <Segmented className="geometry-selector" aria-label={t('Geometry display')}>
            {(['default', 'visual', 'collision', 'both'] as const).map((mode) => (
              <button
                key={mode}
                data-tooltip={t(mode[0].toUpperCase() + mode.slice(1))}
                aria-label={t(mode[0].toUpperCase() + mode.slice(1))}
                aria-pressed={geometry === mode}
                className={geometry === mode ? 'active' : ''}
                onClick={() => setGeometry(mode)}
              >
                <Icon name={mode === 'default' ? 'geometry-default' : mode} size={16} />
                <span>{t(mode[0].toUpperCase() + mode.slice(1))}</span>
              </button>
            ))}
          </Segmented>
        </div>
        <div className={`viewport-content ${sceneCamera.id ? 'has-scene-camera' : ''}`}>
          {doc && (
            <CameraPreview
              doc={doc}
              selectedId={selected}
              enabled={cameraPreviewEnabled}
              revision={revision}
              geometry={geometry}
              onClose={() => setCameraPreviewEnabled(false)}
            />
          )}
          {doc && (
            <Viewport
              ref={viewport}
              document={doc}
              cameraResetKey={sceneEpoch}
              selected={selected}
              tool={tool}
              frame={frame}
              translate={t}
              layers={resolvedLayers}
              onStats={setStats}
              settings={settings}
              onSceneCameraChange={setSceneCamera}
              geometry={geometry}
              projection={projection}
              snapEnabled={snapEnabled}
              translationSnap={settings.translationSnap}
              rotationSnap={settings.rotationSnap}
              activeJoint={activeJoint}
              onActiveJoint={setActiveJoint}
              revision={revision}
              onSelect={setSelected}
              onProjectionChange={setProjection}
              onBegin={begin}
              onJointBegin={beginJoint}
              onJointChange={jointChanged}
              onJointCancel={cancelJoint}
              onPerturbBegin={beginPhysics}
              onPerturbCancel={() => { if (!state.current.recording) cancel(); }}
              onChange={changed}
              onCommit={commit}
            />
          )}
          <Squircle className="tool-palette" radius={8}>
            {tools.map((item) => (
              <Button
                key={item.id}
                icon={item.icon}
                active={tool === item.id}
                variant={tool === item.id ? 'primary' : undefined}
                aria-pressed={tool === item.id}
                title={bindingHint(item.title, item.id)}
                hintText={item.title}
                hintKeys={
                  settings.inputBindings.keys[item.id]
                    ? actionParts(settings.inputBindings, item.id)
                    : undefined
                }
                onClick={() => setTool(item.id)}
              />
            ))}
            <div className="palette-divider" />
            <Button
              icon={frame === 'world' ? 'world' : 'body-frame'}
              title={bindingHint(`${frame === 'world' ? 'World' : 'Body'} frame`, 'frame')}
              hintText={`${frame === 'world' ? 'World' : 'Body'} frame`}
              hintKeys={
                settings.inputBindings.keys.frame
                  ? actionParts(settings.inputBindings, 'frame')
                  : undefined
              }
              onClick={() => setFrame(frame === 'world' ? 'body' : 'world')}
            />
            <Button
              icon="snap"
              title={t('Snapping')}
              hintKeys={
                settings.inputBindings.keys.snap
                  ? actionParts(settings.inputBindings, 'snap')
                  : undefined
              }
              active={snapEnabled}
              variant={snapEnabled ? 'primary' : undefined}
              aria-pressed={snapEnabled}
              onClick={() => setSnapEnabled(!snapEnabled)}
            />
          </Squircle>
          {layersOpen && (
            <div className="viewport-layers-host">
              <LayersPopup
                value={resolvedLayers}
                onChange={changeLayers}
                onClose={() => setLayersOpen(false)}
                translate={t}
              />
            </div>
          )}
          {sceneCamera.id && (
            <div className="scene-camera-return">
              <Button
                icon="previous"
                title="Return to Editor Camera"
                onClick={() => useCamera(null)}
              >
                {t('Editor Camera')}
              </Button>
            </div>
          )}
          {notice && (
            <OperationNotice
              notice={notice}
              onDismiss={(id) => setNoticeState((current) => (current?.id === id ? null : current))}
            />
          )}
        </div>
      </main>
      <aside
        className="inspector panel"
        style={{ visibility: showInspector ? 'visible' : 'hidden' }}
      >
        <div
          className="panel-header inspector-heading"
          data-tooltip={node?.name}
          data-tooltip-overflow="strong"
        >
          <Icon name={node ? (kinds[node.kind] ?? 'cube') : 'inspect'} size={20} />
          <strong>{node?.name ?? t('Inspector')}</strong>
          <span className="flex-spacer" />
          <Button
            icon="pin"
            title={t('Pin Inspector')}
            active={!!pinnedId}
            disabled={!node}
            onClick={() => setPinnedId(pinnedId ? null : node!.id)}
          />
          <Button
            icon="close"
            title={t('Close Inspector')}
            onClick={() => setShowInspector(false)}
          />
        </div>
        {node ? (
          <>
            <div className="inspector-tabs">
              {(node.kind === 'environment' ? ['Properties'] : ['Properties', 'Material']).map((tab) => (
                <button
                  key={tab}
                  className={inspectorTab === tab ? 'active' : ''}
                  data-tooltip={t(tab)}
                  aria-label={t(tab)}
                  onClick={() => setInspectorTab(tab)}
                >
                  {t(tab)}
                </button>
              ))}
            </div>
            <div className="inspector-content">
              {node.kind === 'environment' && simulation?.source ? (
                <EnvironmentProperties fields={simulation.source.environment} values={simulation.environment}
                  onBegin={begin} onCommit={commit} onCancel={cancel}
                  onChange={(values) => simulationCommand({ type: 'environment', values })}
                  onReset={() => simulationTransaction({ type: 'resetEnvironment' })} />
              ) : inspectorTab === 'Properties' && (
                <>
                  {!node.object.userData.physicsOwned && <Section
                    title={t('Transform')}
                    action={
                      <Button
                        icon="reset"
                        title={t('Reset transform')}
                        onClick={() =>
                          transaction('Reset transform', () => {
                            if (joint) setJoint(doc!, joint.id, 0);
                            else {
                              node.object.position.set(0, 0, 0);
                              node.object.rotation.set(0, 0, 0);
                              node.object.scale.set(1, 1, 1);
                            }
                          })
                        }
                      />
                    }
                  >
                    <VectorFields
                      label={t('Position')}
                      values={node.object.position.toArray()}
                      unit="m"
                      onBegin={begin}
                      onCommit={commit}
                      onChange={(i, v) => {
                        node.object.position.setComponent(i, v);
                        transformChanged();
                      }}
                    />
                    <VectorFields
                      label={t('Rotation')}
                      values={[rotation!.x, rotation!.y, rotation!.z].map(THREE.MathUtils.radToDeg)}
                      unit="°"
                      onBegin={begin}
                      onCommit={commit}
                      onChange={(i, v) => {
                        const e = new THREE.Euler().setFromQuaternion(
                          node.object.quaternion,
                          'ZYX',
                        );
                        const a = [e.x, e.y, e.z];
                        a[i] = THREE.MathUtils.degToRad(v);
                        node.object.quaternion.setFromEuler(e.set(a[0], a[1], a[2], 'ZYX'));
                        transformChanged();
                      }}
                    />
                    <VectorFields
                      label={t('Scale')}
                      resetValues={[1, 1, 1]}
                      values={node.object.scale.toArray()}
                      onBegin={begin}
                      onCommit={commit}
                      onChange={(i, v) => {
                        if (v > 0) {
                          node.object.scale.setComponent(i, v);
                          transformChanged();
                        }
                      }}
                    />
                  </Section>}
                  {joint && (
                    <Section title={t('Joint')}>
                      <Property label={t('Type')} value={jointTypeLabel(joint.type)} />
                      <Property label={t('Parent')} value={joint.parent} />
                      <Property label={t('Child')} value={joint.child} />
                      <Property label={t('Axis')} value={joint.axis.toArray().join(', ')} />
                      <JointEditor
                        doc={doc!}
                        joint={joint}
                        showScalarLabel
                        onBegin={beginJoint}
                        onCommit={commit}
                        onCancel={cancelJoint}
                        onChange={jointChanged}
                      />
                    </Section>
                  )}
                  <Section title={t('Display')}>
                    {node.kind !== 'light' && (
                      <SwitchRow
                        label={t('Visible')}
                        checked={node.visible}
                        onChange={(visible) =>
                          transaction('Updated visibility', () => {
                            node.visible = visible;
                            node.object.visible = visible;
                          })
                        }
                      />
                    )}
                    <SwitchRow
                      label="Select in viewport"
                      title={t(
                        node.object.parent && !isViewportPickable(node.object.parent)
                          ? 'Selection disabled by parent'
                          : 'Disabled objects remain editable in the scene tree',
                      )}
                      checked={isViewportPickable(node.object)}
                      disabled={!!node.object.parent && !isViewportPickable(node.object.parent)}
                      onChange={(enabled) =>
                        transaction('Updated viewport selection', () =>
                          setViewportPickable(node.object, enabled),
                        )
                      }
                    />
                    <SwitchRow
                      label={t('Transform handles')}
                      checked={node.object.userData.gizmoEnabled !== false}
                      disabled={!!node.object.userData.physicsOwned &&
                        (!simulation?.source?.capabilities.includes('jointPose') || !['joint', 'link', 'geom'].includes(node.kind))}
                      onChange={(value) =>
                        transaction('Updated transform handles', () => {
                          node.object.userData.gizmoEnabled = value;
                        })
                      }
                    />
                    {node.kind === 'camera' && (
                      <SwitchRow
                        label="Preview"
                        checked={cameraPreviewEnabled}
                        onChange={setCameraPreviewEnabled}
                      />
                    )}
                  </Section>
                  <Section title={t('Identity')} initial={false}>
                    <Property label={t('Name')} value={node.name} />
                    <Property label={t('Type')} value={t(node.kind)} />
                    {Object.entries(node.properties ?? {})
                      .filter(
                        ([key]) =>
                          !['name', 'type'].includes(key.toLowerCase()) &&
                          (!joint || !['axis', 'parent', 'child'].includes(key.toLowerCase())),
                      )
                      .map(([key, value]) => (
                        <Property
                          key={key}
                          label={key.replaceAll('_', ' ')}
                          value={
                            ['shape', 'role', 'projection'].includes(key.toLowerCase())
                              ? t(String(value))
                              : String(value)
                          }
                        />
                      ))}
                  </Section>
                  {node.kind === 'camera' && (
                    <CameraProperties
                      key={node.id}
                      doc={doc!}
                      node={node}
                      active={sceneCamera.id === node.id}
                      piloting={sceneCamera.pilot}
                      onBegin={begin}
                      onCommit={commit}
                      onChange={changed}
                      onView={(pilot) => useCamera(node.id, pilot)}
                      onExit={() => useCamera(null)}
                      onAlign={() => alignCamera(node.id)}
                      onCapture={() => captureCamera(node.id)}
                    />
                  )}
                  {node.kind === 'light' && (
                    <LightProperties
                      key={node.id}
                      doc={doc!}
                      node={node}
                      onBegin={begin}
                      onCommit={commit}
                      onChange={changed}
                    />
                  )}
                </>
              )}
              {inspectorTab === 'Material' && (
                <MaterialPanel
                  key={node.id}
                  node={node}
                  begin={begin}
                  commit={commit}
                  change={changed}
                />
              )}

            </div>
          </>
        ) : (
          <div className="inspector-empty">
            {t('Select an object in the viewport or scene tree.')}
          </div>
        )}
      </aside>
      <section className="bottom-panel" style={{ visibility: showBottom ? 'visible' : 'hidden' }}>
        <div className="bottom-tabs">
          <button
            className={bottom === 'timeline' ? 'active' : ''}
            onClick={() => setBottom('timeline')}
          >
            <Icon name="key" size={16} />
            {t('Timeline')}
          </button>
          <button
            className={bottom === 'output' ? 'active' : ''}
            onClick={() => setBottom('output')}
          >
            <Icon name="terminal" size={16} />
            {t('Output')}
            <span className="count">{logs.length}</span>
          </button>
          <span className="flex-spacer" />
          <Button
            icon="close"
            title={t('Hide bottom panel')}
            onClick={() => setShowBottom(false)}
          />
        </div>
        {bottom === 'timeline' && doc && (
          <Timeline
            inputBindings={settings.inputBindings}
            doc={doc}
            navigationResetKey={sceneEpoch}
            keys={keys}
            timeline={timelineData}
            range={
              timelineData.activeTakeId
                ? (takeViews[timelineData.activeTakeId]?.range ?? null)
                : null
            }
            onRange={changeTakeRange}
            onSelectTake={selectTake}
            onCreateTake={createTake}
            onDeleteTake={deleteTake}
            recording={recording}
            speed={playbackSpeed}
            onSpeed={setPlaybackSpeed}
            onExportVideo={exportVideo}
            videoActive={videoProgress !== null}
            playbackSource={playbackSource}
            onTimeline={changeTimeline}
            onRecord={toggleRecording}
            onSnapshot={captureSnapshot}
            onLoadSnapshot={loadSnapshot}
            onLoadKey={loadKey}
            time={time}
            duration={duration}
            playing={playing}
            loop={loop}
            setLoop={setLoop}
            setTime={seek}
            setDuration={extendDuration}
            setPlaying={setPlaying}
            onCapture={capture}
            onKeys={changeKeys}
            onBegin={begin}
            onCommit={commit}
            onCancel={cancel}
          />
        )}
        {bottom === 'output' && <OutputPanel entries={logs} onClear={() => setLogs([])} />}
      </section>
      {showLeft && (
        <Splitter
          axis="x"
          label={t('Resize left panel')}
          value={leftWidth}
          min={220}
          max={leftMaximum}
          onChange={(left) => setLayout({ ...layout, left })}
          onReset={() => setLayout({ ...layout, left: 272 })}
          style={{ gridColumn: 2, gridRow: '2/4', justifySelf: 'end' }}
        />
      )}
      {showInspector && (
        <Splitter
          axis="x"
          label={t('Resize Inspector')}
          reverse
          value={rightWidth}
          min={280}
          max={rightMaximum}
          onChange={(right) => setLayout({ ...layout, right })}
          onReset={() => setLayout({ ...layout, right: 340 })}
          style={{ gridColumn: 4, gridRow: '2/4', justifySelf: 'start' }}
        />
      )}
      {showBottom && (
        <Splitter
          axis="y"
          label={t('Resize bottom panel')}
          reverse
          value={bottomHeight}
          min={140}
          max={bottomMaximum}
          onChange={(bottom) => setLayout({ ...layout, bottom })}
          onReset={() => setLayout({ ...layout, bottom: 300 })}
          style={{ gridColumn: 3, gridRow: 3, alignSelf: 'start' }}
        />
      )}
      <footer className="statusbar">
        <span className="status-state">
          <span className="status-dot" />
          {t(busy ? 'Loading' : playing || simulation?.playing ? 'Playing' : 'Ready')}
        </span>
        <span className="status-divider" />
        <span className="status-hint">
          <PointerHint
            chord={settings.inputBindings.pointer.orbit[0]}
            bindings={settings.inputBindings}
            size={20}
          />
          {t('Orbit')}
        </span>
        <span className="status-hint">
          <PointerHint
            chord={settings.inputBindings.pointer.pan[0]}
            bindings={settings.inputBindings}
            size={20}
          />
          {t('Pan')}
        </span>
        <span className="status-hint">
          <PointerHint
            chord={settings.inputBindings.pointer.zoom[0]}
            bindings={settings.inputBindings}
            size={20}
          />
          {t('Zoom')}
        </span>
        <span className="status-hint">
          <kbd>{actionLabel(settings.inputBindings, 'focus') || t('Unbound')}</kbd>
          {t('Focus')}
        </span>
        <span className="flex-spacer" />
        <div className="render-stats" aria-label={t('Render statistics')}>
          <span className="renderer-name" data-tooltip={stats?.backend ?? 'Three.js · WebGL 2'}>
            WebGL 2
          </span>
          <span className="status-divider" />
          <span className="status-time" aria-label={t(simulation?.source ? 'Simulation time' : 'Time')} data-tooltip={t(simulation?.source ? 'Simulation time' : 'Time')}>
            {(simulation?.frame?.time ?? time).toFixed(3)} s
          </span>
          <span className="status-divider" />
          <span
            className="stat-fps"
            data-tooltip={t(stats?.idle ? 'Idle viewport' : 'Rendered frames per second')}
          >
            {stats?.idle ? t('Idle') : `${stats?.fps.toFixed(1) ?? '0.0'} FPS`}
          </span>
        </div>
      </footer>
      <TooltipHost />
      <SettingsDialog
        open={settingsOpen}
        settings={settings}
        onChange={updateSettings}
        onClose={() => setSettingsOpen(false)}
      />
      {menu && (
        <>
          <AppMenu
            key={`${menu}:${menu === 'Object' ? `${treeContext?.id}:${treeContext?.x}:${treeContext?.y}` : ''}`}
            name={menu}
            anchor={
              menu === 'Object'
                ? treeContext?.anchor
                : menus.includes(menu)
                  ? menuTrigger.current
                  : undefined
            }
            point={menu === 'Object' ? (treeContext ?? undefined) : undefined}
            onNavigate={menus.includes(menu) ? switchMenu : undefined}
            onClose={() => setMenu((current) => (current === menu ? null : current))}
          >
            {menu === 'Projection' && (
              <>
                {['perspective', 'orthographic'].map((v) => (
                  <div key={v}>
                    {menuItem(
                      v === 'perspective' ? 'Perspective' : 'Orthographic',
                      () => setProjection(v as typeof projection),
                      v,
                    )}
                  </div>
                ))}
                <hr />
                {['front', 'back', 'left', 'right', 'top', 'bottom'].map((v) => (
                  <div key={v}>
                    {menuItem(
                      v[0].toUpperCase() + v.slice(1),
                      () => {
                        if (v === 'perspective' || v === 'orthographic') setProjection(v);
                        viewport.current?.view(v);
                      },
                      `view-${v}`,
                    )}
                  </div>
                ))}
              </>
            )}
            {menus.includes(menu) && <WorkspaceMenuItems name={menu} state={menuState} onAction={dispatchMenuAction} onClose={() => setMenu(null)} />}
            {menu === 'Object' && (
              <>
                {menuItem(
                  'Focus object',
                  () => viewport.current?.focus(selected ?? undefined),
                  'target',
                  actionLabel(settings.inputBindings, 'focus'),
                )}
                {menuItem(
                  'Copy name',
                  () => {
                    if (!selectedNode) return;
                    const write =
                      window.mojive?.writeClipboardText(selectedNode.name) ??
                      navigator.clipboard.writeText(selectedNode.name);
                    void write.catch(report);
                  },
                  'copy',
                )}
                <hr />
                {selectedNode &&
                  menuItem(
                    selectedNode.visible ? 'Hide object' : 'Show object',
                    () => {
                      transaction(
                        message(selectedNode.visible ? 'Hid {{name}}' : 'Showed {{name}}', {
                          name: selectedNode.name,
                        }),
                        () => {
                          selectedNode.visible = !selectedNode.visible;
                          selectedNode.object.visible = selectedNode.visible;
                        },
                      );
                    },
                    selectedNode.visible ? 'eye-off' : 'eye',
                  )}
                {selectedNode &&
                  menuItem(
                    isViewportPickable(selectedNode.object)
                      ? 'Disable viewport picking'
                      : 'Enable viewport picking',
                    () => {
                      transaction(
                        message('Changed viewport picking for {{name}}', {
                          name: selectedNode.name,
                        }),
                        () => {
                          setViewportPickable(
                            selectedNode.object,
                            !isViewportPickable(selectedNode.object),
                          );
                        },
                      );
                    },
                    'select',
                  )}
                <hr />
                {menuItem('Delete object', remove, 'delete', shortcut('Delete'), recording)}
              </>
            )}
          </AppMenu>
        </>
      )}
      {videoProgress !== null && (
        <VideoProgress progress={videoProgress} onCancel={() => videoAbort.current?.abort()} />
      )}
      {sceneLoading.state && !replacePrompt && !fileDialog && (
        <SceneLoadProgress state={sceneLoading.state} onCancel={sceneLoading.cancel} />
      )}
      {fileDialog && (
        <FileDialog
          kind={fileDialog.kind}
          initialPath={fileDialog.initialPath}
          extension={fileDialog.extension}
          caption={fileDialog.caption}
          suggestedName={fileDialog.suggestedName ?? `${doc?.name ?? 'Untitled'}.mojive.json`}
          onChoose={fileDialog.onChoose}
          onCancel={fileDialog.onCancel}
        />
      )}
      {replacePrompt && !fileDialog && (
        <ConfirmSceneDialog
          filename={doc ? t(sceneDisplayName(doc)) : t('Untitled scene')}
          reason={replaceReason}
          onCancel={() => answerReplacement(false)}
          onDiscard={() => answerReplacement(true)}
          onSave={async () => {
            const answer = replaceAnswer.current;
            const saved = await save();
            const current = replaceAnswer.current === answer;
            if (saved && current) answer?.(true);
            return saved && current;
          }}
        />
      )}
      {assetPreview && <AssetPreviewDialog asset={assetPreview} load={loadAssetPreview} settings={settings}
        onClose={() => setAssetPreview(null)} onOpen={openAsset} />}
      {about && (
        <InfoDialog
          label={t('About Mojive')}
          onClose={() => setAbout(false)}
          className="about-dialog"
        >
          <div className="dialog-heading">
            <strong>Mojive</strong>
            <span className="flex-spacer" />
            <Button icon="close" title="Close" onClick={() => setAbout(false)} />
          </div>
          <p>0.1.0</p>
        </InfoDialog>
      )}
      {help && (
        <InfoDialog label={t('Interaction guide')} onClose={() => setHelp(false)}>
          <div className="dialog-heading">
            <Icon name="keyboard" />
            <strong>{t('Interaction guide')}</strong>
            <span className="flex-spacer" />
            <Button icon="close" title={t('Close guide')} onClick={() => setHelp(false)} />
          </div>
          <InteractionGuide bindings={settings.inputBindings} />
        </InfoDialog>
      )}
    </div>
  );
}
function Property({ label, value }: { label: string; value: string }) {
  return (
    <div className="property-row">
      <span>{t(label)}</span>
      <span className="property-value" data-tooltip={value}>
        {value}
      </span>
    </div>
  );
}
function MaterialPanel({
  node,
  begin,
  commit,
  change,
}: {
  node: SceneNode;
  begin: () => void;
  commit: () => void;
  change: () => void;
}) {
  const materials: THREE.MeshStandardMaterial[] = [];
  node.object.traverse((o) => {
    if (o instanceof THREE.Mesh && o.userData.role !== 'collision') {
      for (const m of Array.isArray(o.material) ? o.material : [o.material])
        if (m instanceof THREE.MeshStandardMaterial && !materials.includes(m)) materials.push(m);
    }
  });
  if (!materials.length)
    return (
      <Empty icon="palette" title={t('No surface material')}>
        {t('Select a mesh or a link with visual geometry.')}
      </Empty>
    );
  return (
    <>
      {materials.map((m, i) => (
        <Section title={m.name || t('Material {{index}}', { index: i + 1 })} key={m.uuid}>
          <div className="property-row material-property">
            <label>{t('Base color')}</label>
            <ColorPicker
              label={`${t('Material')} ${i + 1} ${t('Color')}`}
              value={`#${m.color.getHexString()}`}
              onBegin={begin}
              onCommit={commit}
              onChange={(value) => {
                m.color.set(value);
                change();
              }}
            />
          </div>
          <div className="property-row material-property">
            <span>{t('Roughness')}</span>
            <Numeric
              label={t('Material {{index}} roughness', { index: i + 1 })}
              value={m.roughness}
              onBegin={begin}
              onCommit={commit}
              onChange={(v) => {
                m.roughness = THREE.MathUtils.clamp(v, 0, 1);
                change();
              }}
            />
          </div>
          <div className="property-row material-property">
            <span>{t('Metallic')}</span>
            <Numeric
              label={t('Material {{index}} metallic', { index: i + 1 })}
              value={m.metalness}
              onBegin={begin}
              onCommit={commit}
              onChange={(v) => {
                m.metalness = THREE.MathUtils.clamp(v, 0, 1);
                change();
              }}
            />
          </div>
        </Section>
      ))}
    </>
  );
}

function VideoProgress({ progress, onCancel }: { progress: number; onCancel: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    dialog.current?.showModal();
    return () => dialog.current?.close();
  }, []);
  return (
    <dialog
      ref={dialog}
      className="video-export-dialog"
      aria-label={t('Export video')}
      onCancel={(event) => {
        event.preventDefault();
        onCancel();
      }}
    >
      <h2>{t('Export video')}</h2>
      <progress max={1} value={progress} aria-label={t('Export progress')} />
      <div>
        <span className="mono">{Math.round(progress * 100)}%</span>
        <Button onClick={onCancel}>{t('Cancel')}</Button>
      </div>
    </dialog>
  );
}
