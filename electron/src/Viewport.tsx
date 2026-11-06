import {
  DEFAULT_INPUT_BINDINGS,
  FLY_ACTIONS,
  matchesKey,
  keyboardChord,
  focusAllAccelerator,
  matchesPointer,
  keyHeld,
  eventKey,
  type InputBindings,
} from './input-bindings';
import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import * as THREE from 'three';
import type { SceneDocument, Tool, Frame, GeometryMode } from './contracts';
import { createPrimitive } from './scene';
import { isViewportPickable } from './scene-selection';
import { sceneCommands } from './scene-commands';
import {
  cameraNode,
  readCameraView,
  updateCameraView,
  renderSceneCamera,
  cameraViewFromRender,
  type CameraViewState,
  type SceneCameraState,
  type SceneCamera,
} from './scene-camera';
import { GizmoOverlay, type GizmoSettings } from './viewport/gizmo';
import { createViewportScene } from './viewport/scene-environment';
import { ViewportPipeline } from './viewport/pipeline';
import { ViewCubeOverlay } from './viewport/viewcube';
import { SceneEntityInfluence } from './viewport/influence';
import { SceneEntityMarkers } from './viewport/entities';
import {
  CameraViewTransition,
  cameraForTransition,
  type TransitionCameraView,
} from './viewport/camera-transition';
import { CameraFocusTransition } from './viewport/camera-focus';
import { sceneFocusBounds, sceneNodeBounds } from './viewport/focus-bounds';
import { jointFocusPlan } from './viewport/joint-focus';
import { PreciseInputPopover } from './viewport/precise-input';
import { JointPicker } from './viewport/joint-picker';
import { PerturbationGesture } from './viewport/perturbation';
import { PerturbationFeedback } from './viewport/perturbation-feedback';
import {
  DEFAULT_VIEWPORT_LAYERS,
  objectGizmosEnabled,
  type ViewportLayers,
} from './viewport/layers';
import { ViewportMetrics, type ViewportStats } from './viewport/metrics';
export type { ViewportStats, ViewportLayers };
export interface ViewportHandle {
  useSceneCamera: (id: string | null, pilot?: boolean) => void;
  alignCameraToView: (id: string) => void;
  createCameraFromView: (name?: string) => string;
  captureCameraView: (id?: string) => Promise<{ dataUrl: string; width: number; height: number }>;
  getCameraView: () => CameraViewState;
  focus: (id?: string) => void;
  view: (preset: string) => void;
  capture: () => string;
  setProjection: (mode: 'perspective' | 'orthographic') => void;
}
export interface ViewportSettings {
  inputBindings?: InputBindings;
  uiScale?: number;
  showGrid?: boolean;
  showViewCube?: boolean;
  showEntityIcons?: boolean;
  showEntityInfluence?: boolean;
  selectionOutline?: boolean;
  selectionHighlight?: boolean;
  outlineWidth?: number;
  cameraFov?: number;
  orbitSpeed?: number;
  panSpeed?: number;
  zoomSpeed?: number;
  focusMargin?: number;
  transitionDuration?: number;
  rotationTickScale?: number;
  translationSnap?: number;
  rotationSnap?: number;
}
export interface ViewportProps {
  onSceneCameraChange?: (state: SceneCameraState) => void;
  layers?: ViewportLayers;
  onStats?: (stats: ViewportStats) => void;
  translate?: (text: string) => string;
  snapEnabled?: boolean;
  translationSnap?: number;
  rotationSnap?: number;
  activeJoint?: string | null;
  onActiveJoint?: (id: string) => void;
  projection?: 'perspective' | 'orthographic';
  settings?: ViewportSettings;
  cameraResetKey?: number;
  document: SceneDocument;
  selected: string | null;
  tool: Tool;
  frame: Frame;
  geometry: GeometryMode;
  revision: number;
  onSelect: (id: string | null) => void;
  onChange: () => void;
  onBegin?: () => void;
  onJointBegin?: () => void;
  onJointChange?: () => void;
  onJointCancel?: () => void;
  onPerturbBegin?: () => void;
  onPerturbCancel?: () => void;
  onCommit?: () => void;
  onProjectionChange?: (mode: 'perspective' | 'orthographic') => void;
}
export const Viewport = forwardRef<ViewportHandle, ViewportProps>(function Viewport(props, ref) {
  const host = useRef<HTMLDivElement>(null),
    latest = useRef(props),
    api = useRef<ViewportHandle>({
      useSceneCamera: () => {},
      alignCameraToView: () => {},
      createCameraFromView: () => '',
      captureCameraView: async () => ({ dataUrl: '', width: 0, height: 0 }),
      getCameraView: () => {
        throw Error('Viewport is not ready.');
      },
      focus: () => {},
      view: () => {},
      capture: () => '',
      setProjection: () => {},
    });
  latest.current = props;
  const invalidateRef = useRef<() => void>(() => {});
  useImperativeHandle(ref, () => ({
    useSceneCamera: (id, pilot) => api.current.useSceneCamera(id, pilot),
    alignCameraToView: (id) => api.current.alignCameraToView(id),
    createCameraFromView: (name) => api.current.createCameraFromView(name),
    captureCameraView: (id) => api.current.captureCameraView(id),
    getCameraView: () => api.current.getCameraView(),
    focus: (id) => api.current.focus(id),
    view: (p) => api.current.view(p),
    capture: () => api.current.capture(),
    setProjection: (mode) => api.current.setProjection(mode),
  }));
  useEffect(
    () => invalidateRef.current(),
    [
      props.document,
      props.selected,
      props.tool,
      props.frame,
      props.geometry,
      props.revision,
      props.cameraResetKey,
      props.snapEnabled,
      props.translationSnap,
      props.rotationSnap,
      props.projection,
      props.activeJoint,
      props.settings,
      props.layers,
    ],
  );
  useEffect(() => {
    const container = host.current!;
    let width = 1,
      height = 1,
      queued = 0,
      disposed = false,
      attached: SceneDocument | null = null,
      resetKey = latest.current.cameraResetKey;
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });
    renderer.info.autoReset = false;
    const metrics = new ViewportMetrics();
    let lastStatsAt = -Infinity;
    const publishStats = () => {
      const now = performance.now();
      if (now - lastStatsAt >= 250) {
        lastStatsAt = now;
        latest.current.onStats?.(metrics.sample(now));
      }
    };
    const statsTimer = window.setInterval(publishStats, 250);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.domElement.style.cssText =
      'display:block;width:100%;height:100%;touch-action:none;outline:none;';
    renderer.domElement.tabIndex = 0;
    renderer.domElement.setAttribute('aria-label', '3D scene viewport');
    renderer.domElement.dataset.testid = 'viewport-canvas';
    const overlay = document.createElement('canvas');
    overlay.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;';
    const cube = document.createElement('canvas');
    cube.style.cssText =
      'position:absolute;right:12px;top:12px;width:90px;height:90px;touch-action:none;';
    cube.setAttribute('aria-label', 'View cube');
    cube.dataset.testid = 'view-cube';
    container.append(renderer.domElement, overlay, cube);
    const cameraGate = document.createElement('div');
    cameraGate.style.cssText =
      'position:absolute;pointer-events:none;border:1px solid #ffffff55;box-shadow:0 0 0 100vmax #1118;display:none;';
    const cameraCaption = document.createElement('span');
    cameraCaption.style.cssText =
      'position:absolute;left:8px;top:8px;color:var(--text-color);font:12px var(--font-ui);background:var(--ui-child);padding:4px 8px;border-radius:4px;';
    cameraGate.append(cameraCaption);
    container.insertBefore(cameraGate, overlay);
    const precise = new PreciseInputPopover(
      container,
      () => renderer.domElement.focus(),
      (text) => latest.current.translate?.(text) ?? text,
    );
    const jointPicker = new JointPicker(container, {
      select: (id) => latest.current.onActiveJoint?.(id),
      focus: (id) => focus(id),
      restoreFocus: () => renderer.domElement.focus(),
      translate: (text) => latest.current.translate?.(text) ?? text,
    });
    let styledSettings: ViewportSettings | undefined;
    let styledTheme: string | undefined;
    let readoutStyle: NonNullable<GizmoSettings['readoutStyle']> = {
      font: '"JetBrains Mono",monospace',
      size: 16,
      surface: '#252a2f',
      ink: '#e6e8eb',
    };
    const environment = createViewportScene();
    const { scene, grid, dispose: disposeEnvironment } = environment;
    const perspective = new THREE.PerspectiveCamera(42, 1, 0.001, 10000),
      orthographic = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.001, 10000);
    let camera: THREE.Camera = perspective;
    let activeCamera: string | null = null,
      piloting = false,
      sceneRenderCamera: SceneCamera | undefined;
    function selectedGizmosEnabled() {
      const p = latest.current;
      const object = p.document.nodes.find((node) => node.id === p.selected)?.object;
      // Native viewing_selected_camera yields both paint and input ownership.
      return (
        !perturbationHeld() &&
        !perturbation.active &&
        (!activeCamera || p.selected !== activeCamera) &&
        p.layers?.gizmos !== false &&
        (!object?.userData.physicsOwned || p.tool === 'joint') &&
        objectGizmosEnabled(object)
      );
    }
    const navigationUp = new THREE.Vector3(0, 0, 1);
    let editorFov = latest.current.settings?.cameraFov ?? 42;
    let observedFov = editorFov;
    let editorOrthoRatio = 2 * Math.tan(THREE.MathUtils.degToRad(editorFov / 2));
    const sceneBounds = new THREE.Sphere(new THREE.Vector3(), 1);
    let boundsRevision = -1;
    let boundsDocument: SceneDocument | null = null;
    let pilotStart: {
      id: string;
      camera: SceneCamera;
      properties: Record<string, string | number> | undefined;
    } | null = null;
    let pilotWheelTimer = 0;
    camera.up.set(0, 0, 1);
    orthographic.up.set(0, 0, 1);
    let yaw = -135,
      pitch = 25,
      distance = 4.5,
      ortho = latest.current.projection === 'orthographic',
      projectionBlend = ortho ? 1 : 0;
    let observedProjection = latest.current.projection;
    let pendingProjection: 'perspective' | 'orthographic' | null = null;
    function acknowledgeProjection() {
      if (latest.current.projection === pendingProjection) pendingProjection = null;
    }
    function publishProjection(mode: 'perspective' | 'orthographic') {
      acknowledgeProjection();
      // React may not commit a controlled-prop echo before the next imperative
      // command. Keep one pending notification until that echo is acknowledged.
      if (pendingProjection === mode) return;
      pendingProjection = mode;
      latest.current.onProjectionChange?.(mode);
    }
    type ProjectionAnimation = { start: number; from: number; to: number };
    let projectionAnimation: ProjectionAnimation | null = null;
    const target = new THREE.Vector3(0, 0, 0.6);
    type NavigationAnimation = {
      start: number;
      fromTarget: THREE.Vector3;
      toTarget: THREE.Vector3;
      fromYaw: number;
      toYaw: number;
      fromPitch: number;
      toPitch: number;
      fromDistance: number;
      toDistance: number;
      duration: number;
      easeOut: boolean;
    };
    let animation: NavigationAnimation | null = null;
    let handoff: { curve: CameraViewTransition; start: number; pauseEditor: boolean } | null = null;
    let handoffCamera: THREE.Camera | undefined;
    let renderedView: TransitionCameraView | null = null;
    let editorExactView: TransitionCameraView | null = null;
    let editorExactCamera: THREE.Camera | undefined;
    let focusMotion: { curve: CameraFocusTransition; start: number } | null = null;
    function stopHandoff() {
      if (handoff?.pauseEditor) {
        const paused = performance.now() - handoff.start;
        if (animation) animation.start += paused;
        if (focusMotion) focusMotion.start += paused;
        if (projectionAnimation) projectionAnimation.start += paused;
      }
      handoff = null;
    }
    // Scene cameras own their view; piloting may borrow the orbit controls, never the editor state.
    let editorNavigation: {
      yaw: number;
      pitch: number;
      distance: number;
      target: THREE.Vector3;
      up: THREE.Vector3;
      fov: number;
      orthoRatio: number;
      ortho: boolean;
      blend: number;
      animation: NavigationAnimation | null;
      projectionAnimation: ProjectionAnimation | null;
      pausedAt: number;
      exactView: TransitionCameraView | null;
      focusMotion: { curve: CameraFocusTransition; start: number } | null;
    } | null = null;
    function saveEditorNavigation() {
      editorNavigation = {
        yaw,
        pitch,
        distance,
        target: target.clone(),
        up: navigationUp.clone(),
        fov: editorFov,
        orthoRatio: editorOrthoRatio,
        ortho,
        blend: projectionBlend,
        animation,
        projectionAnimation,
        pausedAt: performance.now(),
        exactView: editorExactView,
        focusMotion,
      };
    }
    function restoreEditorNavigation() {
      const saved = editorNavigation;
      if (!saved) return;
      yaw = saved.yaw;
      pitch = saved.pitch;
      distance = saved.distance;
      target.copy(saved.target);
      navigationUp.copy(saved.up);
      editorFov = saved.fov;
      editorOrthoRatio = saved.orthoRatio;
      ortho = saved.ortho;
      projectionBlend = saved.blend;
      const paused = performance.now() - saved.pausedAt;
      animation = saved.animation
        ? { ...saved.animation, start: saved.animation.start + paused }
        : null;
      projectionAnimation = saved.projectionAnimation
        ? { ...saved.projectionAnimation, start: saved.projectionAnimation.start + paused }
        : null;
      editorExactView = saved.exactView;
      focusMotion = saved.focusMotion
        ? { ...saved.focusMotion, start: saved.focusMotion.start + paused }
        : null;
      editorNavigation = null;
    }
    const pipeline = new ViewportPipeline(renderer);
    const markers = new SceneEntityMarkers();
    const influence = new SceneEntityInfluence();
    function invalidate() {
      if (!queued && !disposed) queued = requestAnimationFrame(draw);
    }
    invalidateRef.current = invalidate;
    let capturedPointerId: number | null = null;
    function capturePointer(id: number) {
      capturedPointerId = id;
      renderer.domElement.setPointerCapture(id);
    }
    function releasePointer() {
      const id = capturedPointerId;
      capturedPointerId = null;
      if (id !== null && renderer.domElement.hasPointerCapture(id))
        renderer.domElement.releasePointerCapture(id);
      renderer.domElement.style.cursor = 'default';
    }
    let jointGesture = false;
    const gizmo = new GizmoOverlay(overlay, {
      onBegin: () => {
        const current = latest.current;
        jointGesture = current.tool === 'joint';
        (jointGesture ? current.onJointBegin ?? current.onBegin : current.onBegin)?.();
      },
      onChange: () => {
        const current = latest.current;
        (jointGesture ? current.onJointChange ?? current.onChange : current.onChange)();
      },
      onCommit: () => {
        jointGesture = false;
        releasePointer();
        latest.current.onCommit?.();
      },
      onCancel: () => {
        const current = latest.current,
          cancel = jointGesture ? current.onJointCancel ?? current.onCommit : current.onCommit;
        jointGesture = false;
        releasePointer();
        cancel?.();
      },
      invalidate,
    });
    const perturbation = new PerturbationGesture();
    const perturbationFeedback = new PerturbationFeedback();
    function finishPerturbation(cancelled = false) {
      if (!perturbation.end()) return;
      if (cancelled) latest.current.onPerturbCancel?.();
      else latest.current.onCommit?.();
      renderer.domElement.style.cursor = 'default';
      invalidate();
    }
    function beginPilot() {
      if (pilotStart || !activeCamera || !piloting) return;
      const node = cameraNode(latest.current.document, activeCamera);
      const interrupted = handoff ? currentRenderedView() : null;
      pilotStart = {
        id: activeCamera,
        camera: (node.object as SceneCamera).clone(false),
        properties: node.properties ? { ...node.properties } : undefined,
      };
      latest.current.onBegin?.();
      stopHandoff();
      // Inspector edits and an interrupted handoff both seed the next gesture.
      syncOrbit(interrupted ?? readCameraView(node));
      if (interrupted) {
        const authored = readCameraView(node),
          gateY = Math.min(1, width / height / (authored.resolution[0] / authored.resolution[1]));
        updateCameraView(latest.current.document, activeCamera, {
          ...interrupted,
          ...(interrupted.projectionBlend === 0 && interrupted.normalizedFocal
            ? {
                fov: THREE.MathUtils.radToDeg(
                  2 * Math.atan(gateY / (2 * interrupted.normalizedFocal[1])),
                ),
              }
            : {}),
          orthoHeight: interrupted.orthoHeight * gateY,
          resolution: authored.resolution,
        });
        latest.current.onChange();
      }
    }
    function finishPilot(cancel = false) {
      if (pilotWheelTimer) {
        clearTimeout(pilotWheelTimer);
        pilotWheelTimer = 0;
      }
      if (!pilotStart) return;
      const start = pilotStart;
      pilotStart = null;
      if (cancel) {
        const node = cameraNode(latest.current.document, start.id),
          object = node.object;
        if (
          object instanceof THREE.PerspectiveCamera &&
          start.camera instanceof THREE.PerspectiveCamera
        )
          object.copy(start.camera, false);
        else if (
          object instanceof THREE.OrthographicCamera &&
          start.camera instanceof THREE.OrthographicCamera
        )
          object.copy(start.camera, false);
        node.properties = start.properties;
        object.updateMatrixWorld(true);
        latest.current.onChange();
        syncOrbit(readCameraView(node));
      }
      latest.current.onCommit?.();
    }
    function syncOrbit(view: CameraViewState) {
      target.fromArray(view.target);
      const eye = new THREE.Vector3().fromArray(view.position),
        delta = eye.sub(target);
      distance = Math.max(0.001, delta.length());
      const direction = delta.multiplyScalar(1 / distance);
      if (Math.hypot(direction.x, direction.y) > 1e-9)
        yaw = THREE.MathUtils.radToDeg(Math.atan2(direction.y, direction.x));
      else {
        // Native _adopted_yaw uses screen-up only to recover heading at a Z pole.
        const screenUp = new THREE.Vector3().fromArray(view.up);
        screenUp.addScaledVector(direction, -screenUp.dot(direction));
        const horizontal = Math.hypot(screenUp.x, screenUp.y),
          sign = direction.z >= 0 ? -1 : 1;
        yaw =
          Number.isFinite(horizontal) && horizontal > 1e-9
            ? THREE.MathUtils.radToDeg(Math.atan2(sign * screenUp.y, sign * screenUp.x))
            : -90;
      }
      pitch = THREE.MathUtils.clamp(
        THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(direction.z, -1, 1))),
        -89.9,
        89.9,
      );
      // Screen-up encodes the authored view, not the orbit pole. Native gestures
      // clear an exact/rolled view and reconstruct yaw/pitch around world Z.
      navigationUp.set(0, 0, 1);
      editorFov = view.fov;
      editorOrthoRatio = view.orthoHeight / distance;
      ortho = view.projection === 'orthographic';
      projectionBlend = ortho ? 1 : 0;
      projectionAnimation = null;
      animation = null;
    }
    function useSceneCamera(id: string | null, pilot = false, animate = true) {
      const previousId = activeCamera,
        previousPilot = piloting,
        previousProjection = ortho;
      const node = id ? cameraNode(latest.current.document, id) : null;
      if (pilot && node?.object.userData.physicsOwned)
        throw new Error('The simulation backend owns this camera pose and projection type.');
      if (id === previousId) {
        finishPilot();
        piloting = Boolean(id && pilot);
        if (previousPilot !== piloting)
          latest.current.onSceneCameraChange?.({ id, pilot: piloting });
        invalidate();
        return;
      }
      const start = animate ? currentRenderedView() : null;
      finishPilot();
      stopHandoff();
      if (node) {
        if (!activeCamera) saveEditorNavigation();
        editorExactView = null;
        focusMotion = null;
        syncOrbit(readCameraView(node));
      } else if (activeCamera) restoreEditorNavigation();
      activeCamera = id;
      piloting = Boolean(id && pilot);
      cameraGate.style.display = id ? 'block' : 'none';
      if (previousId !== id || previousPilot !== piloting)
        latest.current.onSceneCameraChange?.({ id, pilot: piloting });
      if (previousProjection !== ortho) publishProjection(ortho ? 'orthographic' : 'perspective');
      observedProjection = latest.current.projection;
      if (start)
        handoff = {
          curve: new CameraViewTransition(start),
          start: performance.now(),
          pauseEditor: !id,
        };
      invalidate();
    }
    function leaveSceneView() {
      if ((!activeCamera && !handoff && !focusMotion && !editorExactView) || piloting) return;
      const view = currentRenderedView(),
        previousProjection = ortho;
      stopHandoff();
      focusMotion = null;
      editorExactView = null;
      if (activeCamera) {
        activeCamera = null;
        editorNavigation = null;
        cameraGate.style.display = 'none';
        latest.current.onSceneCameraChange?.({ id: null, pilot: false });
      }
      syncOrbit({
        ...view,
        ...(view.projectionBlend === 0 && view.normalizedFocal
          ? { fov: THREE.MathUtils.radToDeg(2 * Math.atan(0.5 / view.normalizedFocal[1])) }
          : {}),
      });
      if (previousProjection !== ortho) publishProjection(ortho ? 'orthographic' : 'perspective');
      observedProjection = latest.current.projection;
    }
    function getCameraView(): CameraViewState {
      cameraUpdate();
      if ((handoff || editorExactView) && renderedView)
        return {
          ...renderedView,
          up: new THREE.Vector3(0, 1, 0)
            .applyQuaternion(camera.getWorldQuaternion(new THREE.Quaternion()))
            .toArray(),
        };
      return activeCamera
        ? readCameraView(cameraNode(latest.current.document, activeCamera))
        : editorRenderedView();
    }
    function editorRenderedView(): CameraViewState {
      const view = cameraViewFromRender(camera, target, [
        Math.max(1, Math.round(width)),
        Math.max(1, Math.round(height)),
      ]);
      // OrthographicCamera has no FOV property, but the orbit controller retains
      // that optical state for projection changes and native focus framing.
      return {
        ...view,
        fov: editorFov,
        focalLength:
          view.sensorWidth /
          Math.max(width / height, 1) /
          (2 * Math.tan(THREE.MathUtils.degToRad(editorFov / 2))),
      };
    }
    function updatePilotedCamera() {
      if (!activeCamera || !piloting) return;
      cameraUpdate(true);
      const previous = readCameraView(cameraNode(latest.current.document, activeCamera)),
        view = cameraViewFromRender(camera, target, previous.resolution);
      updateCameraView(latest.current.document, activeCamera, {
        position: view.position,
        target: view.target,
        up: view.up,
        ...(ortho ? { orthoHeight: view.orthoHeight } : {}),
      });
      latest.current.onChange();
    }
    function currentRenderedView(): TransitionCameraView {
      cameraUpdate();
      return structuredClone(renderedView!);
    }
    function cameraUpdate(editorOnly = false) {
      if (!activeCamera && focusMotion && !handoff?.pauseEditor) {
        const elapsed = (performance.now() - focusMotion.start) / 1000,
          sampled = focusMotion.curve.sample(elapsed);
        syncOrbit(sampled);
        editorExactView = sampled;
        if (elapsed >= focusMotion.curve.duration) focusMotion = null;
        else invalidate();
      }
      updateGoalCamera(editorOnly);
      if (editorOnly) return;
      const source = activeCamera
        ? readCameraView(cameraNode(latest.current.document, activeCamera))
        : (editorExactView ?? editorRenderedView());
      const aspect = width / height,
        blend = Number(
          camera.userData.projectionBlend ?? (source.projection === 'orthographic' ? 1 : 0),
        ),
        m = camera.projectionMatrix.elements,
        focal = 0.5 / Math.tan(THREE.MathUtils.degToRad(source.fov / 2));
      const goal: TransitionCameraView = {
        ...source,
        resolution: [Math.max(1, Math.round(width)), Math.max(1, Math.round(height))],
        projectionBlend: blend,
        viewportRelativeFocal: activeCamera
          ? false
          : ((source as TransitionCameraView).viewportRelativeFocal ?? true),
        normalizedFocal: blend > 0 ? [focal / aspect, focal] : [m[0] / 2, m[5] / 2],
        principalOffset: blend > 0 ? [0, 0] : [m[8] / 2, -m[9] / 2],
        orthoHeight: blend >= 1 ? 2 / m[5] : source.orthoHeight,
      };
      renderedView = goal;
      if (handoff) {
        const elapsed = (performance.now() - handoff.start) / 1000;
        renderedView = handoff.curve.sample(goal, elapsed);
        if (elapsed >= 0.28) stopHandoff();
        else {
          camera = handoffCamera = cameraForTransition(renderedView, aspect, handoffCamera);
          invalidate();
        }
      }
    }
    function updateGoalCamera(editorOnly = false) {
      if (activeCamera && !editorOnly) {
        const node = latest.current.document.nodes.find(
          (node) => node.id === activeCamera && node.kind === 'camera',
        );
        if (!node) {
          useSceneCamera(null, false, false);
          updateGoalCamera(editorOnly);
          return;
        } else {
          const authoredOrtho = node.object instanceof THREE.OrthographicCamera;
          if (authoredOrtho !== ortho) {
            syncOrbit(readCameraView(node));
            publishProjection(authoredOrtho ? 'orthographic' : 'perspective');
          }
          sceneRenderCamera = renderSceneCamera(node, width / height, sceneRenderCamera);
          camera = sceneRenderCamera;
          const view = readCameraView(node),
            aspect = view.resolution[0] / view.resolution[1],
            gateWidth = Math.min(width, height * aspect),
            gateHeight = gateWidth / aspect;
          cameraGate.style.display = 'block';
          cameraGate.style.left = (width - gateWidth) / 2 + 'px';
          cameraGate.style.top = (height - gateHeight) / 2 + 'px';
          cameraGate.style.width = gateWidth + 'px';
          cameraGate.style.height = gateHeight + 'px';
          cameraCaption.textContent = `${node.name} · ${view.resolution[0]} × ${view.resolution[1]}${piloting ? ' · ' + (latest.current.translate?.('Pilot camera') ?? 'Pilot camera') : ''}`;
          return;
        }
      }
      if (editorExactView && !editorOnly) {
        const previousAspect = editorExactView.resolution[0] / editorExactView.resolution[1],
          aspect = width / height;
        if (
          editorExactView.viewportRelativeFocal &&
          previousAspect !== aspect &&
          editorExactView.normalizedFocal
        )
          editorExactView = {
            ...editorExactView,
            resolution: [width, height],
            normalizedFocal: [
              (editorExactView.normalizedFocal[0] * previousAspect) / aspect,
              editorExactView.normalizedFocal[1],
            ],
          };
        camera = editorExactCamera = cameraForTransition(
          editorExactView,
          width / height,
          editorExactCamera,
        );
        return;
      }
      const a = THREE.MathUtils.degToRad(yaw),
        b = THREE.MathUtils.degToRad(pitch);
      camera = projectionBlend >= 1 ? orthographic : perspective;
      camera.position.set(
        target.x + distance * Math.cos(b) * Math.cos(a),
        target.y + distance * Math.cos(b) * Math.sin(a),
        target.z + distance * Math.sin(b),
      );
      camera.up.copy(navigationUp);
      camera.lookAt(target);
      const optics = activeCamera
        ? readCameraView(cameraNode(latest.current.document, activeCamera))
        : null;
      const span =
        optics?.projection === 'orthographic'
          ? optics.orthoHeight / 2
          : ortho && !projectionAnimation
            ? (distance * editorOrthoRatio) / 2
            : distance * Math.tan(THREE.MathUtils.degToRad((optics?.fov ?? editorFov) / 2));
      orthographic.left = (-span * width) / height;
      orthographic.right = (span * width) / height;
      orthographic.top = span;
      orthographic.bottom = -span;
      // Native close-up near plane scales down with distance; a fixed minimum clips tiny models.
      const near = Math.max(1e-8, distance * 0.0025);
      const far = Math.max(
        near * 100,
        distance * 2,
        camera.position.distanceTo(sceneBounds.center) + sceneBounds.radius * 2,
        1,
      );
      orthographic.near = near;
      orthographic.far = far;
      orthographic.updateProjectionMatrix();
      perspective.fov = optics?.fov ?? editorFov;
      perspective.aspect = width / height;
      perspective.near = near;
      perspective.far = far;
      perspective.updateProjectionMatrix();
      if (projectionBlend > 0 && projectionBlend < 1) {
        const a = perspective.projectionMatrix.elements,
          b = orthographic.projectionMatrix.elements;
        for (let i = 0; i < 16; i++)
          a[i] = (a[i] / distance) * (1 - projectionBlend) + b[i] * projectionBlend;
        perspective.projectionMatrixInverse.copy(perspective.projectionMatrix).invert();
      }
      camera.userData.projectionBlend = projectionBlend;
      camera.updateMatrixWorld();
    }
    function animate(
      toTarget = target.clone(),
      toDistance = distance,
      toYaw = yaw,
      toPitch = pitch,
      duration = 0.36,
    ) {
      let dy = toYaw - yaw;
      dy = ((((dy + 180) % 360) + 360) % 360) - 180;
      animation = {
        start: performance.now(),
        fromTarget: target.clone(),
        toTarget,
        fromYaw: yaw,
        toYaw: yaw + dy,
        fromPitch: pitch,
        toPitch,
        fromDistance: distance,
        toDistance,
        duration: duration * ((latest.current.settings?.transitionDuration ?? 0.36) / 0.36),
        easeOut: duration === 0.35,
      };
      invalidate();
    }
    function focus(
      id?: string,
      instant = false,
      margin = latest.current.settings?.focusMargin ?? 1.15,
      orientation?: [number, number],
    ) {
      const doc = latest.current.document;
      const node = id === undefined ? undefined : doc.nodes.find((node) => node.id === id);
      if (id !== undefined && !node) return;
      // Environment is scene-wide: use the complete Frame All transition and camera exit.
      if (node?.kind === 'environment') id = undefined;
      // Native node/joint focus requests cannot replace an active scene camera.
      // Frame All and ViewCube alignment have separate navigation entry points.
      if (activeCamera && id !== undefined && orientation === undefined) return;
      const start = currentRenderedView(),
        previousProjection = ortho;
      finishPilot();
      stopHandoff();
      activeCamera = null;
      piloting = false;
      editorNavigation = null;
      focusMotion = null;
      cameraGate.style.display = 'none';
      latest.current.onSceneCameraChange?.({ id: null, pilot: false });
      const joint = !orientation && id ? jointFocusPlan(doc, id, start) : null,
        bounds = joint ?? sceneFocusBounds(doc, id),
        angles = orientation?.map(THREE.MathUtils.degToRad),
        direction = angles
          ? [
              Math.cos(angles[1]) * Math.cos(angles[0]),
              Math.cos(angles[1]) * Math.sin(angles[0]),
              Math.sin(angles[1]),
            ]
          : joint?.direction,
        curve = new CameraFocusTransition(start, bounds, {
          margin: orientation ? 1.15 : margin,
          ...(id === undefined ? { frameScene: true } : {}),
          duration: instant
            ? 0
            : (orientation || id === undefined ? 0.35 / 0.36 : 1) *
              (latest.current.settings?.transitionDuration ?? 0.36),
          ...(direction ? { eyeDirection: direction } : {}),
          ...(orientation
            ? {
                sphereRadius: Math.hypot(...bounds.half),
                sphereMargin: margin,
                easing: 'ease_out_quart',
              }
            : joint
              ? { sphereRadius: joint.radius }
              : {}),
        });
      if (instant) {
        editorExactView = curve.end;
        syncOrbit(curve.end);
      } else {
        editorExactView = start;
        syncOrbit(start);
        focusMotion = { curve, start: performance.now() };
      }
      if (previousProjection !== ortho) publishProjection(ortho ? 'orthographic' : 'perspective');
      observedProjection = latest.current.projection;
      invalidate();
    }
    function setProjection(mode: 'perspective' | 'orthographic', fromProps = false) {
      if (activeCamera) {
        if (fromProps) return;
        useSceneCamera(null);
      }
      const changed = (mode === 'orthographic') !== ortho;
      if (!fromProps) observedProjection = latest.current.projection;
      if (
        changed ||
        (!fromProps &&
          latest.current.projection !== undefined &&
          latest.current.projection !== mode)
      )
        publishProjection(mode);
      if (!changed) return;
      if (!activeCamera && (editorExactView || focusMotion)) leaveSceneView();
      ortho = mode === 'orthographic';
      editorOrthoRatio = 2 * Math.tan(THREE.MathUtils.degToRad(editorFov / 2));
      projectionAnimation = { start: performance.now(), from: projectionBlend, to: ortho ? 1 : 0 };
      invalidate();
    }
    function toggleProjection() {
      setProjection(ortho ? 'perspective' : 'orthographic');
    }
    function view(preset: string) {
      if (activeCamera) useSceneCamera(null);
      else leaveSceneView();
      navigationUp.set(0, 0, 1);
      if (preset === 'orthographic') {
        setProjection('orthographic');
        return;
      }
      const presets: Record<string, [number, number]> = {
        front: [-90, 0],
        back: [90, 0],
        right: [0, 0],
        left: [180, 0],
        top: [-90, 89.9],
        bottom: [-90, -89.9],
        perspective: [-135, 30],
        iso: [-135, 30],
      };
      const angles = presets[preset];
      if (!angles) return;
      if (preset === 'perspective' || preset === 'iso') setProjection('perspective');
      animate(target.clone(), distance, angles[0], angles[1], 0.35);
    }
    const viewCube = new ViewCubeOverlay(cube, {
      onAxis: (axis, sign) => {
        if (activeCamera) useSceneCamera(null);
        navigationUp.set(0, 0, 1);
        const angles: Record<string, [number, number]> = {
          '0:1': [0, 0],
          '0:-1': [180, 0],
          '1:1': [90, 0],
          '1:-1': [-90, 0],
          '2:1': [-90, 89.9],
          '2:-1': [-90, -89.9],
        };
        const a = angles[`${axis}:${sign}`];
        if (
          latest.current.selected &&
          sceneNodeBounds(latest.current.document, latest.current.selected)
        ) {
          focus(latest.current.selected, false, 1.2, a);
          return;
        }
        leaveSceneView();
        animate(
          animation?.toTarget ?? target.clone(),
          animation?.toDistance ?? distance,
          a[0],
          a[1],
          0.35,
        );
      },
      onProjectionToggle: toggleProjection,
      onOrbit: (dx, dy) => {
        if (activeCamera) useSceneCamera(null, false, false);
        leaveSceneView();
        animation = null;
        yaw -= dx * (latest.current.settings?.orbitSpeed ?? 0.35);
        pitch = Math.max(
          -89.9,
          Math.min(89.9, pitch + dy * (latest.current.settings?.orbitSpeed ?? 0.35)),
        );
        invalidate();
      },
      invalidate,
    });
    function draw() {
      queued = 0;
      if (disposed) return;
      const p = latest.current,
        frameStarted = performance.now(),
        layers = p.layers ?? DEFAULT_VIEWPORT_LAYERS;
      renderer.info.reset();
      acknowledgeProjection();
      if (p.projection !== observedProjection) {
        observedProjection = p.projection;
        if (p.projection) setProjection(p.projection, true);
      }
      const settings = p.settings;
      const theme = document.documentElement.dataset.theme ?? '';
      if (styledSettings !== settings || styledTheme !== theme) {
        styledSettings = settings;
        styledTheme = theme;
        const style = getComputedStyle(document.documentElement);
        readoutStyle = {
          font: style.getPropertyValue('--font-mono').trim() || readoutStyle.font,
          size: parseFloat(style.getPropertyValue('--numeric-font-size')) || 16,
          surface: style.getPropertyValue('--floating-color').trim() || '#252a2f',
          ink: style.getPropertyValue('--text-color').trim() || '#e6e8eb',
          controls: {
            primary: style.getPropertyValue('--sage-color').trim() || '#9cbf8d',
            bright: style.getPropertyValue('--ui-primary-bright').trim() || '#b8d2ac',
            dim: style.getPropertyValue('--ui-primary-dim').trim() || '#67875a',
            field: style.getPropertyValue('--input-color').trim() || '#2b2f34',
            hovered: style.getPropertyValue('--ui-field-hovered').trim() || '#363b41',
            active: style.getPropertyValue('--ui-field-active').trim() || '#40474e',
            disabled: style.getPropertyValue('--dim-color').trim() || '#8e98a5',
          },
        };
      }
      grid.visible = layers.grid && settings?.showGrid !== false;
      cube.style.display = !layers.viewCube || settings?.showViewCube === false ? 'none' : '';
      jointPicker.update(
        p.document,
        p.selected,
        p.activeJoint,
        p.tool !== 'select' && selectedGizmosEnabled(),
      );
      gizmo.configure({
        snapEnabled: p.snapEnabled,
        translationSnap: p.translationSnap ?? settings?.translationSnap,
        rotationSnap: p.rotationSnap ?? settings?.rotationSnap,
        rotationTickScale: settings?.rotationTickScale,
        activeJoint: p.activeJoint,
        translate: p.translate,
        readoutStyle,
      });
      if (attached !== p.document) {
        perturbation.end();
        const reset = !attached || resetKey !== p.cameraResetKey;
        if (attached) scene.remove(attached.root);
        attached = p.document;
        scene.add(attached.root);
        environment.configure(attached.root);
        resetKey = p.cameraResetKey;
        if (reset) {
          useSceneCamera(null, false, false);
          stopHandoff();
          navigationUp.set(0, 0, 1);
          focus(undefined, true);
        }
      }
      if (animation && !handoff?.pauseEditor) {
        const t = Math.min(1, (performance.now() - animation.start) / (animation.duration * 1000)),
          e = animation.easeOut ? t * (2 - t) : t * t * (3 - 2 * t);
        target.lerpVectors(animation.fromTarget, animation.toTarget, e);
        distance = THREE.MathUtils.lerp(animation.fromDistance, animation.toDistance, e);
        yaw = THREE.MathUtils.lerp(animation.fromYaw, animation.toYaw, e);
        pitch = THREE.MathUtils.lerp(animation.fromPitch, animation.toPitch, e);
        if (t >= 1) animation = null;
        else invalidate();
      }
      if (projectionAnimation && !handoff?.pauseEditor) {
        const t = Math.min(
            1,
            (performance.now() - projectionAnimation.start) /
              (280 * ((settings?.transitionDuration ?? 0.36) / 0.36)),
          ),
          e = t * t * (3 - 2 * t);
        projectionBlend = THREE.MathUtils.lerp(projectionAnimation.from, projectionAnimation.to, e);
        if (t >= 1) projectionAnimation = null;
        else invalidate();
      }
      if (observedFov !== (settings?.cameraFov ?? 42)) {
        observedFov = settings?.cameraFov ?? 42;
        editorFov = observedFov;
        editorOrthoRatio = 2 * Math.tan(THREE.MathUtils.degToRad(editorFov / 2));
        if (editorNavigation) {
          editorNavigation.fov = editorFov;
          editorNavigation.orthoRatio = editorOrthoRatio;
        }
      }
      if (boundsDocument !== p.document || boundsRevision !== p.revision) {
        p.document.root.updateWorldMatrix(true, true);
        const bounds = new THREE.Box3().setFromObject(p.document.root);
        if (!bounds.isEmpty()) bounds.getBoundingSphere(sceneBounds);
        boundsDocument = p.document;
        boundsRevision = p.revision;
      }
      advanceFly(frameStarted);
      cameraUpdate();
      scene.updateMatrixWorld(true);
      influence.update(
        p.document.nodes.find((node) => node.id === p.selected),
        camera,
        width,
        height,
        !activeCamera &&
          layers.helpers &&
          layers.influence !== false &&
          settings?.showEntityInfluence !== false,
      );
      environment.update(camera);
      pipeline.render(scene, camera, p.document, layers.selection ? p.selected : null, p.geometry, {
        ...settings,
        geometryVisible: layers.geometry,
        renderHelpers:
          !activeCamera && layers.helpers && layers.influence !== false
            ? () => influence.render(renderer, camera)
            : undefined,
      });
      if (!activeCamera && layers.helpers && settings?.showEntityIcons !== false)
        markers.render(
          renderer,
          camera,
          p.document,
          p.selected,
          width,
          height,
          settings?.uiScale ?? 1,
        );
      gizmo.update(
        camera,
        p.document,
        p.selected,
        selectedGizmosEnabled() && !perturbation.active ? p.tool : 'select',
        p.frame,
        width,
        height,
      );
      perturbationFeedback.draw(
        overlay.getContext('2d')!,
        perturbation,
        camera,
        width,
        height,
        Math.min(window.devicePixelRatio, 2),
      );
      precise.validate();
      viewCube.update(camera);
      metrics.record(frameStarted, performance.now(), renderer.info.render);
      publishStats();
    }
    const resize = () => {
      const rect = container.getBoundingClientRect();
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      renderer.setSize(width, height, false);
      pipeline.resize(
        Math.round(width * renderer.getPixelRatio()),
        Math.round(height * renderer.getPixelRatio()),
      );
      invalidate();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    const themeObserver = new MutationObserver(() => {
      styledTheme = undefined;
      invalidate();
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['style', 'data-theme'],
    });
    const point = (e: PointerEvent | MouseEvent): [number, number] => {
      const r = container.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    let navigation: {
      id: number;
      button: number;
      start: [number, number];
      last: [number, number];
      moved: boolean;
      kind: 'orbit' | 'pan' | 'dolly' | null;
      select: boolean;
    } | null = null;
    const pickHit = (p: [number, number]): { id: string; point?: THREE.Vector3 } | null => {
      const helper =
        !activeCamera &&
        latest.current.layers?.helpers !== false &&
        latest.current.settings?.showEntityIcons !== false
          ? markers.pick(p)
          : null;
      if (helper) return { id: helper };
      if (latest.current.layers?.geometry === false) return null;
      const ray = new THREE.Raycaster();
      ray.setFromCamera(new THREE.Vector2((p[0] / width) * 2 - 1, 1 - (p[1] / height) * 2), camera);
      const hits = ray.intersectObject(latest.current.document.root, true);
      for (const hit of hits) {
        if (!isViewportPickable(hit.object)) continue;
        if (hit.object instanceof THREE.Mesh) {
          const material = hit.object.material;
          if (
            Array.isArray(material) ? material.every((m) => m.opacity <= 0) : material.opacity <= 0
          )
            continue;
        }
        let visible = true;
        for (let o: THREE.Object3D | null = hit.object; o; o = o.parent) {
          if (
            !o.visible &&
            (latest.current.geometry === 'default' || o.userData.roleGroup !== true)
          )
            visible = false;
        }
        const role = pipeline.role(hit.object);
        if (
          !visible ||
          (latest.current.geometry === 'visual' && role === 'collision') ||
          (latest.current.geometry === 'collision' && role === 'visual')
        )
          continue;
        for (let o: THREE.Object3D | null = hit.object; o; o = o.parent) {
          if (o.userData.nodeId) return { id: String(o.userData.nodeId), point: hit.point };
        }
      }
      return null;
    };
    const pick = (p: [number, number]) => pickHit(p)?.id ?? null;
    const heldKeys = new Set<string>();
    const bindings = () => latest.current.settings?.inputBindings ?? DEFAULT_INPUT_BINDINGS;
    let flyModifiers = { ctrlKey: false, altKey: false, shiftKey: false, metaKey: false };
    function perturbationHeld() {
      return (
        !!sceneCommands(latest.current.document)?.perturbation &&
        keyHeld(bindings().keys.perturb, flyModifiers, heldKeys)
      );
    }
    const updateModifiers = (e: MouseEvent | KeyboardEvent) => {
      const wasPerturbing = perturbationHeld();
      flyModifiers = {
        ctrlKey: e.ctrlKey,
        altKey: e.altKey,
        shiftKey: e.shiftKey,
        metaKey: e.metaKey,
      };
      if (wasPerturbing !== perturbationHeld()) {
        gizmo.hoverAt([-9999, -9999]);
        invalidate();
      }
    };
    let flyTime: number | null = null;
    let flyPilot = false;
    let flyEngaged = false;
    const flyActions = Object.keys(FLY_ACTIONS) as (keyof typeof FLY_ACTIONS)[];
    const flyForward = new THREE.Vector3(),
      flyRight = new THREE.Vector3();
    function stopFly(cancel = false) {
      flyTime = null;
      flyEngaged = false;
      for (const action of flyActions) {
        const key = bindings().keys[action]?.split('+').at(-1);
        if (key) heldKeys.delete(key);
      }
      if (flyPilot && (!navigation || cancel)) finishPilot(cancel);
      flyPilot = false;
    }
    function advanceFly(now: number) {
      if (!flyEngaged) return;
      if (
        document.activeElement !== renderer.domElement ||
        document.hidden ||
        precise.active ||
        gizmo.active ||
        (navigation && !navigation.kind) ||
        (activeCamera && !piloting)
      ) {
        stopFly();
        return;
      }
      const down = (action: keyof typeof FLY_ACTIONS) => {
        const chord = bindings().keys[action];
        const key = chord?.split('+').at(-1);
        return Boolean(
          key &&
            keyHeld(chord, flyModifiers, heldKeys) &&
            matchesKey(bindings(), action, { ...flyModifiers, key }),
        );
      };
      const forward = Number(down('flyForward')) - Number(down('flyBack'));
      const right = Number(down('flyRight')) - Number(down('flyLeft'));
      const up = Number(down('flyUp')) - Number(down('flyDown'));
      if (!forward && !right && !up) {
        flyTime = null;
        return;
      }
      const dt = flyTime === null ? 0 : Math.min(0.05, Math.max(0, (now - flyTime) / 1000));
      flyTime = now;
      leaveSceneView();
      beginPilot();
      flyPilot = piloting;
      animation = null;
      cameraUpdate();
      camera.getWorldDirection(flyForward);
      flyRight.setFromMatrixColumn(camera.matrixWorld, 0);
      const speed = distance * 1.6 * dt;
      target.addScaledVector(flyForward, forward * speed).addScaledVector(flyRight, right * speed);
      target.z += up * speed;
      if (piloting && speed) updatePilotedCamera();
      invalidate();
    }
    const down = (e: PointerEvent) => {
      if (precise.active || e.defaultPrevented || navigation || gizmo.active || perturbation.active)
        return;
      renderer.domElement.focus();
      const input = bindings();
      updateModifiers(e);
      if (perturbationHeld()) {
        const hit = pickHit(point(e));
        // Perturb the selection, including drags beginning over empty space or another body.
        if (
          perturbation.begin(
            latest.current.document,
            latest.current.selected,
            camera,
            point(e),
            width,
            height,
            e.pointerId,
            e.button,
            () => latest.current.onPerturbBegin?.(),
            hit?.point ? { id: hit.id, point: hit.point } : undefined,
          )
        ) {
          stopFly();
          animation = null;
          capturePointer(e.pointerId);
          renderer.domElement.style.cursor = 'grabbing';
        }
        e.preventDefault();
        invalidate();
        return;
      }
      const kind =
        (['pan', 'dolly', 'orbit'] as const).find((action) =>
          matchesPointer(input, action, e, heldKeys),
        ) ?? null;
      const snap = keyHeld(input.keys.snap, e, heldKeys);
      const primary =
        e.button === 0 &&
        (!e.ctrlKey || (input.keys.snap?.split('+').includes('Control') ?? false)) &&
        (!e.altKey || (input.keys.snap?.split('+').includes('Alt') ?? false)) &&
        !e.metaKey;
      if (primary && selectedGizmosEnabled() && gizmo.begin(point(e))) {
        gizmo.setSnapping(snap);
        capturePointer(e.pointerId);
        e.preventDefault();
        return;
      }
      if (!kind && !primary) return;
      navigation = {
        id: e.pointerId,
        button: e.button,
        start: point(e),
        last: point(e),
        moved: false,
        kind,
        select: primary && !e.ctrlKey && !e.altKey && !e.metaKey && !e.shiftKey,
      };
      renderer.domElement.focus();
      capturePointer(e.pointerId);
      e.preventDefault();
    };
    const move = (e: PointerEvent) => {
      const p = point(e);
      updateModifiers(e);
      if (perturbation.active) {
        if (perturbation.owns(e.pointerId)) {
          perturbation.move(p);
          invalidate();
        }
        return;
      }
      if (gizmo.active) {
        if (capturedPointerId === e.pointerId)
          gizmo.move(p, false, keyHeld(bindings().keys.snap, e, heldKeys));
        return;
      }
      if (!navigation) {
        renderer.domElement.style.cursor =
          (perturbationHeld() &&
            sceneCommands(latest.current.document)?.perturbation?.bodyAt(
              latest.current.selected ?? '',
            )) ||
          (selectedGizmosEnabled() && gizmo.hoverAt(p))
            ? 'grab'
            : 'default';
        return;
      }
      if (navigation.id !== e.pointerId) return;
      const dx = p[0] - navigation.last[0],
        dy = p[1] - navigation.last[1];
      if (Math.hypot(p[0] - navigation.start[0], p[1] - navigation.start[1]) > 4)
        navigation.moved = true;
      if (navigation.moved && navigation.kind === 'dolly') {
        zoomBy(dy * 0.01 * (latest.current.settings?.zoomSpeed ?? 1), false);
        navigation.last = p;
        return;
      }
      if (navigation.moved && navigation.kind) {
        leaveSceneView();
        beginPilot();
        animation = null;
        if (navigation.kind === 'pan') {
          const projectedTarget = new THREE.Vector4(target.x, target.y, target.z, 1)
            .applyMatrix4(camera.matrixWorldInverse)
            .applyMatrix4(camera.projectionMatrix);
          const unit =
              ((2 * Math.abs(projectedTarget.w)) /
                (height * Math.abs(camera.projectionMatrix.elements[5]))) *
              (latest.current.settings?.panSpeed ?? 1),
            right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0),
            up = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
          target.addScaledVector(right, -dx * unit).addScaledVector(up, dy * unit);
        } else {
          yaw -= dx * (latest.current.settings?.orbitSpeed ?? 0.35);
          pitch = Math.max(
            -89.9,
            Math.min(89.9, pitch + dy * (latest.current.settings?.orbitSpeed ?? 0.35)),
          );
        }
        if (piloting) updatePilotedCamera();
        invalidate();
      }
      navigation.last = p;
    };
    const up = (e: PointerEvent) => {
      if (perturbation.active) {
        if (!perturbation.releases(e.pointerId, e.button)) return;
        finishPerturbation();
        releasePointer();
        return;
      }
      if (capturedPointerId !== e.pointerId) return;
      if ((navigation && e.button !== navigation.button) || (gizmo.active && e.button !== 0))
        return;
      if (gizmo.active) {
        gizmo.hoverAt(point(e));
        gizmo.end();
      } else if (navigation?.id === e.pointerId) {
        if (!navigation.moved && navigation.select) {
          const id = pick(point(e));
          jointPicker.selectedAt(id, point(e));
          latest.current.onSelect(id);
        }
        navigation = null;
        if (!flyEngaged) finishPilot();
      }
      releasePointer();
    };
    const cancel = () => {
      finishPerturbation(true);
      stopFly(true);
      heldKeys.clear();
      flyModifiers = { ctrlKey: false, altKey: false, shiftKey: false, metaKey: false };
      gizmo.setSnapping(false);
      finishPilot(true);
      gizmo.end(true);
      navigation = null;
      releasePointer();
    };
    const zoomBy = (delta: number, settle = true) => {
      if (!Number.isFinite(delta)) return;
      const exactDolly =
        !activeCamera && (editorExactView || focusMotion) ? currentRenderedView() : null;
      leaveSceneView();
      animation = null;
      beginPilot();
      distance = Math.exp(
        THREE.MathUtils.clamp(Math.log(distance) + delta, Math.log(0.001), Math.log(1e6)),
      );
      if (exactDolly) {
        editorExactView = {
          ...exactDolly,
          position: new THREE.Vector3()
            .fromArray(exactDolly.position)
            .sub(target)
            .normalize()
            .multiplyScalar(distance)
            .add(target)
            .toArray(),
          target: target.toArray(),
          near: Math.min(exactDolly.near, Math.max(1e-8, distance * 0.0025)),
          far: Math.max(exactDolly.far, distance * 2),
          orthoHeight: ortho
            ? THREE.MathUtils.clamp(exactDolly.orthoHeight * Math.exp(delta), 0.0001, 1e6)
            : exactDolly.orthoHeight,
        };
        syncOrbit(editorExactView);
      }
      if (piloting && activeCamera) {
        // Native dolly retains an exact authored view, including roll and poles.
        // Only orbit/pan reconstruct the level world-Z navigation frame.
        const view = readCameraView(cameraNode(latest.current.document, activeCamera)),
          offset = new THREE.Vector3()
            .fromArray(view.position)
            .sub(target)
            .normalize()
            .multiplyScalar(distance);
        updateCameraView(latest.current.document, activeCamera, {
          position: offset.add(target).toArray(),
          target: target.toArray(),
          up: view.up,
          ...(ortho
            ? {
                orthoHeight: Math.exp(
                  THREE.MathUtils.clamp(
                    Math.log(view.orthoHeight) + delta,
                    Math.log(0.0001),
                    Math.log(1e6),
                  ),
                ),
              }
            : {}),
        });
        latest.current.onChange();
        if (pilotWheelTimer) clearTimeout(pilotWheelTimer);
        if (settle) pilotWheelTimer = window.setTimeout(() => finishPilot(), 180);
      }
      invalidate();
    };
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      if (perturbation.active) return;
      if (!matchesPointer(bindings(), 'zoom', e, heldKeys, true)) return;
      zoomBy(
        e.deltaY *
          (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? height : 1) *
          0.0012 *
          (latest.current.settings?.zoomSpeed ?? 1),
      );
    };
    const dbl = (e: MouseEvent) => {
      if (perturbation.active || keyHeld(bindings().keys.perturb, e, heldKeys)) return;
      const target = selectedGizmosEnabled() ? gizmo.preciseInput(point(e)) : null;
      if (target) {
        e.preventDefault();
        e.stopPropagation();
        navigation = null;
        precise.show(target, point(e));
        return;
      }
      const id = pick(point(e));
      if (id) {
        latest.current.onSelect(id);
        jointPicker.dismiss();
        focus(id);
      }
    };
    const keydown = (e: KeyboardEvent) => {
      updateModifiers(e);
      heldKeys.add(eventKey(e));
      flyModifiers = {
        ctrlKey: e.ctrlKey,
        altKey: e.altKey,
        shiftKey: e.shiftKey,
        metaKey: e.metaKey,
      };
      if (
        Object.keys(FLY_ACTIONS).some((action) =>
          matchesKey(bindings(), action as keyof typeof FLY_ACTIONS, e),
        )
      ) {
        e.preventDefault();
        e.stopPropagation();
        if (
          !precise.active &&
          !gizmo.active &&
          !perturbation.active &&
          (!navigation || navigation.kind) &&
          (!activeCamera || piloting)
        ) {
          flyEngaged = true;
          flyTime ??= performance.now();
          invalidate();
        }
        return;
      }
      gizmo.setSnapping(keyHeld(bindings().keys.snap, e, heldKeys));
      invalidate();
      if (e.key === 'Escape') {
        if (perturbation.active) {
          cancel();
          e.preventDefault();
          e.stopPropagation();
          return;
        }
        if (flyTime !== null || flyPilot) {
          stopFly(true);
          e.preventDefault();
          e.stopPropagation();
          return;
        }
        if (pilotStart) {
          cancel();
          e.preventDefault();
          e.stopPropagation();
          return;
        }
        if (gizmo.active || navigation) {
          cancel();
          e.preventDefault();
          e.stopPropagation();
          return;
        }
        latest.current.onSelect(null);
      }
      if (
        !perturbation.active &&
        (matchesKey(bindings(), 'focus', e) || keyboardChord(e) === focusAllAccelerator(bindings()))
      ) {
        focus(
          keyboardChord(e) === focusAllAccelerator(bindings())
            ? undefined
            : (latest.current.selected ?? undefined),
        );
        e.preventDefault();
      }
    };
    const keyup = (e: KeyboardEvent) => {
      heldKeys.delete(eventKey(e));
      flyModifiers = {
        ctrlKey: e.ctrlKey,
        altKey: e.altKey,
        shiftKey: e.shiftKey,
        metaKey: e.metaKey,
      };
      if (
        flyEngaged &&
        !flyActions.some((action) => {
          const key = bindings().keys[action]?.split('+').at(-1);
          return key && heldKeys.has(key);
        })
      )
        stopFly();
      if (perturbation.active && !perturbationHeld()) finishPerturbation();
      if (document.activeElement === renderer.domElement) invalidate();
      gizmo.setSnapping(keyHeld(bindings().keys.snap, e, heldKeys));
    };
    const context = (e: Event) => e.preventDefault();
    container.addEventListener('pointerdown', down);
    container.addEventListener('pointermove', (e) => {
      if (e.target === cube && !e.defaultPrevented) move(e);
    });
    renderer.domElement.addEventListener('pointermove', move);
    renderer.domElement.addEventListener('pointerleave', () => {
      if (!gizmo.active) gizmo.hoverAt([-9999, -9999]);
    });
    renderer.domElement.addEventListener('pointerup', up);
    renderer.domElement.addEventListener('pointercancel', cancel);
    renderer.domElement.addEventListener('lostpointercapture', () => {
      if (navigation || gizmo.active || perturbation.active) cancel();
    });
    window.addEventListener('keydown', updateModifiers);
    window.addEventListener('blur', cancel);
    renderer.domElement.addEventListener('wheel', wheel, { passive: false });
    renderer.domElement.addEventListener('dblclick', dbl);
    renderer.domElement.addEventListener('keydown', keydown);
    renderer.domElement.addEventListener('blur', cancel);
    const visibilityChanged = () => {
      if (document.hidden) cancel();
    };
    document.addEventListener('visibilitychange', visibilityChanged);
    window.addEventListener('keyup', keyup);
    renderer.domElement.addEventListener('contextmenu', context);
    api.current = {
      useSceneCamera,
      getCameraView,
      alignCameraToView: (id) => {
        const view = getCameraView();
        latest.current.onBegin?.();
        updateCameraView(latest.current.document, id, view);
        latest.current.onChange();
        latest.current.onCommit?.();
        invalidate();
      },
      createCameraFromView: (name) => {
        const view = getCameraView();
        latest.current.onBegin?.();
        const node = createPrimitive('camera', name);
        node.parentId = undefined;
        latest.current.document.root.add(node.object);
        latest.current.document.nodes.push(node);
        updateCameraView(latest.current.document, node.id, view);
        latest.current.onSelect(node.id);
        latest.current.onChange();
        latest.current.onCommit?.();
        invalidate();
        return node.id;
      },
      captureCameraView: async (id) => {
        const node = id
          ? cameraNode(latest.current.document, id)
          : activeCamera
            ? cameraNode(latest.current.document, activeCamera)
            : null;
        const view = node ? readCameraView(node) : getCameraView(),
          captureCamera = node ? renderSceneCamera(node) : camera.clone(),
          [w, h] = view.resolution,
          ratio = renderer.getPixelRatio(),
          gridVisible = grid.visible;
        if (w > renderer.capabilities.maxTextureSize || h > renderer.capabilities.maxTextureSize)
          throw Error('Camera resolution exceeds this GPU limit.');
        try {
          grid.visible = false;
          renderer.setPixelRatio(1);
          renderer.setSize(w, h, false);
          pipeline.resize(w, h);
          scene.updateMatrixWorld(true);
          pipeline.render(
            scene,
            captureCamera,
            latest.current.document,
            null,
            latest.current.geometry,
            {
              ...latest.current.settings,
              geometryVisible: latest.current.layers?.geometry !== false,
              selectionOutline: false,
              selectionHighlight: false,
            },
          );
          return { dataUrl: renderer.domElement.toDataURL('image/png'), width: w, height: h };
        } finally {
          grid.visible = gridVisible;
          renderer.setPixelRatio(ratio);
          renderer.setSize(width, height, false);
          pipeline.resize(Math.round(width * ratio), Math.round(height * ratio));
          invalidate();
        }
      },
      focus,
      view,
      setProjection,
      capture: () => {
        draw();
        const result = document.createElement('canvas');
        result.width = renderer.domElement.width;
        result.height = renderer.domElement.height;
        const ctx = result.getContext('2d')!;
        ctx.drawImage(renderer.domElement, 0, 0);
        ctx.drawImage(overlay, 0, 0);
        const ratio = renderer.getPixelRatio();
        if (
          latest.current.layers?.viewCube !== false &&
          latest.current.settings?.showViewCube !== false
        )
          ctx.drawImage(cube, (width - 102) * ratio, 12 * ratio, 90 * ratio, 90 * ratio);
        return result.toDataURL('image/png');
      },
    };
    resize();
    invalidate();
    return () => {
      window.removeEventListener('keydown', updateModifiers);
      window.clearInterval(statsTimer);
      if (pilotWheelTimer) clearTimeout(pilotWheelTimer);
      window.removeEventListener('blur', cancel);
      window.removeEventListener('keyup', keyup);
      document.removeEventListener('visibilitychange', visibilityChanged);
      container.removeEventListener('pointerdown', down);
      disposed = true;
      cancelAnimationFrame(queued);
      observer.disconnect();
      themeObserver.disconnect();
      perturbation.end();
      perturbationFeedback.dispose();
      precise.dispose();
      jointPicker.dispose();
      viewCube.dispose();
      gizmo.dispose();
      markers.dispose();
      influence.dispose();
      pipeline.dispose();
      renderer.dispose();
      disposeEnvironment();
      container.replaceChildren();
      invalidateRef.current = () => {};
    };
  }, []);
  return (
    <div
      ref={host}
      className="viewport-stage"
      style={{ position: 'absolute', inset: 0, overflow: 'hidden' }}
    />
  );
});
export default Viewport;
