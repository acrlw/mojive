import * as THREE from 'three';
import type { GeometryMode, SceneDocument } from './contracts';
import {
  assertPoseCompatible,
  decodeDocument,
  encodeDocument,
  sampleTake,
  type SceneTake,
} from './editor';
import { disposeDocument } from './scene';
import { cameraFromView, type CameraViewState } from './scene-camera';
import { ViewportPipeline } from './viewport/pipeline';
import { createViewportScene } from './viewport/scene-environment';

/** Export a complete recorded take without seeking or modifying the editor document. */
export async function exportTakeVideo(
  document: SceneDocument,
  take: SceneTake,
  view: CameraViewState,
  options: {
    geometry?: GeometryMode;
    grid?: boolean;
    range?: [number, number] | null;
    fps?: number;
    signal?: AbortSignal;
    onProgress?: (progress: number) => void;
  } = {},
): Promise<Blob> {
  if (!take.frames.length) throw Error('Record a take before exporting video.');
  const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find((type) =>
    MediaRecorder.isTypeSupported(type),
  );
  if (!mimeType) throw Error('WebM recording is unavailable.');
  const fps = Math.max(1, Math.min(60, Math.round(options.fps ?? 30)));
  const first = Math.max(0, Math.min(take.frames.length - 1, options.range?.[0] ?? 0));
  const last = Math.max(
    first,
    Math.min(take.frames.length - 1, options.range?.[1] ?? take.frames.length - 1),
  );
  for (const frame of take.frames.slice(first, last + 1))
    assertPoseCompatible(document, frame.pose);
  options.signal?.throwIfAborted();
  const camera = cameraFromView(view);
  const [width, height] = view.resolution;
  const sceneDocument = decodeDocument(encodeDocument(document));
  let renderer: THREE.WebGLRenderer | undefined;
  let pipeline: ViewportPipeline | undefined;
  let disposeEnvironment: (() => void) | undefined;
  let stream: MediaStream | undefined;
  let recorder: MediaRecorder | undefined;
  try {
    // ObjectLoader.parse restores images asynchronously even for embedded PNGs.
    // Decode before the first sample; a one-frame take has no later frame to recover.
    await waitForTextureImages(sceneDocument.root, options.signal);
    options.signal?.throwIfAborted();
    const { scene, grid, dispose } = createViewportScene();
    disposeEnvironment = dispose;
    scene.add(sceneDocument.root);
    grid.visible = options.grid !== false;
    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    pipeline = new ViewportPipeline(renderer);
    if (
      width > renderer.capabilities.maxTextureSize ||
      height > renderer.capabilities.maxTextureSize
    )
      throw Error('Camera resolution exceeds this GPU limit.');
    renderer.setPixelRatio(1);
    renderer.setSize(width, height, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    pipeline.resize(width, height);
    const render = (time: number) => {
      sampleTake(sceneDocument, take, time, 'floor');
      scene.updateMatrixWorld(true);
      pipeline!.render(scene, camera, sceneDocument, null, options.geometry ?? 'default', {
        selectionHighlight: false,
        selectionOutline: false,
      });
    };
    const start = take.frames[first].time;
    const tail = last > first ? take.frames[last].time - take.frames[last - 1].time : 1 / fps;
    const duration = take.frames[last].time - start + Math.max(tail, 1 / fps);
    const frames = Math.max(1, Math.ceil(duration * fps));
    render(start);
    // MediaRecorder timestamps use wall time. Pace exact scene samples instead of
    // encoding a tight loop, which produces a nearly zero-duration file.
    stream = renderer.domElement.captureStream(0);
    const track = stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack;
    const chunks: Blob[] = [];
    let bytes = 0;
    recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 8_000_000 });
    let failure: Error | null = null;
    recorder.ondataavailable = (event) => {
      if (event.data.size) {
        chunks.push(event.data);
        bytes += event.data.size;
      }
      if (bytes > 500 * 1024 * 1024) failure = Error('Video exceeds the 500 MiB export limit.');
    };
    const stopped = new Promise<void>((resolve) => {
      recorder!.onstop = () => resolve();
      recorder!.onerror = (event) => {
        failure = Error(String(event));
        resolve();
      };
    });
    recorder.start(1000);
    const wallStart = performance.now();
    for (let index = 0; index < frames; index++) {
      options.signal?.throwIfAborted();
      if (failure) throw failure;
      render(Math.min(take.frames[last].time, start + index / fps));
      track.requestFrame();
      options.onProgress?.((index + 1) / frames);
      await delay(
        Math.max(0, wallStart + ((index + 1) * 1000) / fps - performance.now()),
        options.signal,
      );
    }
    if (failure) throw failure;
    if (recorder.state !== 'inactive') recorder.stop();
    await stopped;
    if (failure) throw failure;
    options.signal?.throwIfAborted();
    if (!chunks.length) throw Error('The video encoder returned no frames.');
    return new Blob(chunks, { type: 'video/webm' });
  } finally {
    if (recorder?.state === 'recording') recorder.stop();
    stream?.getTracks().forEach((track) => track.stop());
    pipeline?.dispose();
    disposeEnvironment?.();
    disposeDocument(sceneDocument);
    renderer?.dispose();
    renderer?.forceContextLoss();
  }
}
function delay(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer);
      reject(signal?.reason ?? new DOMException('Aborted', 'AbortError'));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', abort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
  });
}

async function waitForTextureImages(root: THREE.Object3D, signal?: AbortSignal) {
  const images = new Set<HTMLImageElement>();
  const collect = (value: unknown) => {
    const texture = value as THREE.Texture | undefined;
    if (!texture?.isTexture) return;
    for (const image of Array.isArray(texture.image) ? texture.image : [texture.image])
      if (image instanceof HTMLImageElement) images.add(image);
  };
  root.traverse((object) => {
    // Materials own the standard maps; lights can also own projected texture maps.
    Object.values(object).forEach(collect);
    const material = (object as THREE.Mesh).material;
    for (const item of material ? (Array.isArray(material) ? material : [material]) : [])
      Object.values(item).forEach(collect);
  });
  signal?.throwIfAborted();
  if (!images.size) return;
  await new Promise<void>((resolve, reject) => {
    const cleanup = () => signal?.removeEventListener('abort', abort);
    const abort = () => {
      cleanup();
      reject(signal?.reason ?? new DOMException('Aborted', 'AbortError'));
    };
    signal?.addEventListener('abort', abort, { once: true });
    Promise.all([...images].map((image) => image.decode())).then(
      () => {
        cleanup();
        resolve();
      },
      () => {
        cleanup();
        reject(Error('A texture image could not be decoded.'));
      },
    );
    if (signal?.aborted) abort();
  });
}
