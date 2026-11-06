import type { SceneDocument } from '../contracts';
import { loadWorkspace, saveWorkspace, type Keyframe, type WorkspaceTimeline } from '../editor';
import { disposeDocument } from '../scene';
import { createModelBundle, type ModelBundle } from '../resources/modelBundle';
import type { PhysicsBackendFactory, PhysicsSnapshot } from './contracts';
import { PhysicsDocument } from './PhysicsDocument';
import { validateRecordedSimulation } from './sceneState';
import { checkModelLoad, yieldModelLoad, type ModelLoadOptions } from '../resources/loading';

interface StoredPhysics {
  version: 1;
  backend: string;
  bundle: { id: string; name: string; entry: string; files: { path: string; base64: string }[] };
  snapshot: PhysicsSnapshot;
}

function encodeBytes(bytes: Uint8Array): string {
  const parts: string[] = [];
  for (let offset = 0; offset < bytes.length; offset += 32768)
    parts.push(String.fromCharCode(...bytes.subarray(offset, offset + 32768)));
  return btoa(parts.join(''));
}

async function decodeBundle(value: StoredPhysics['bundle'], options?: ModelLoadOptions): Promise<ModelBundle> {
  if (!value || typeof value.id !== 'string' || typeof value.name !== 'string' ||
    typeof value.entry !== 'string' || !Array.isArray(value.files) || !value.files.length || value.files.length > 10000)
    throw new Error('Invalid saved model package.');
  let total = 0;
  const files: ModelBundle['files'] = [];
  let budgetStart = performance.now();
  for (const file of value.files) {
    checkModelLoad(options);
    if (!file || typeof file.path !== 'string' || typeof file.base64 !== 'string' ||
      (total += file.base64.length * 0.75) > 512 * 1024 * 1024)
      throw new Error('Invalid saved model resource.');
    const decoded = atob(file.base64);
    files.push({ path: file.path, bytes: Uint8Array.from(decoded, (character) => character.charCodeAt(0)) });
    if (performance.now() - budgetStart >= 8) {
      options?.onProgress?.({ stage: 'resources', completed: files.length, total: value.files.length, resource: file.path });
      await yieldModelLoad(options); budgetStart = performance.now();
    }
  }
  const bundle = await createModelBundle(value.name, value.entry, files);
  if (bundle.id !== value.id) throw new Error('Saved model resources do not match their identity.');
  return bundle;
}

/** Persist engine resources and integration state alongside ordinary viewer edits. */
export async function savePhysicsWorkspace(
  doc: SceneDocument, keys: Keyframe[], duration: number, timeline: WorkspaceTimeline,
  runtime: PhysicsDocument | null,
): Promise<string> {
  if (!runtime) return saveWorkspace(doc, keys, duration, timeline);
  runtime.session.pause();
  const snapshot = await runtime.session.capture();
  if (runtime.document !== doc) throw new Error('The active simulation document changed while saving.');
  const data = JSON.parse(saveWorkspace(doc, keys, duration, timeline));
  const { bundle } = runtime;
  const physics: StoredPhysics = { version: 1, backend: runtime.session.current.source!.backend,
    bundle: { id: bundle.id, name: bundle.name, entry: bundle.entry,
      files: bundle.files.map(({ path, bytes }) => ({ path, base64: encodeBytes(bytes) })) }, snapshot };
  return JSON.stringify({ ...data, physics }, null, 2);
}

/** Compile and validate a candidate completely before replacing the active workspace. */
export async function loadPhysicsWorkspace(
  content: string, resolveBackend: (id: string) => PhysicsBackendFactory, options?: ModelLoadOptions,
): Promise<ReturnType<typeof loadWorkspace> & { runtime: PhysicsDocument | null }> {
  await yieldModelLoad(options);
  options?.onProgress?.({ stage: 'resources' });
  const data = JSON.parse(content);
  const loaded = loadWorkspace(content);
  let runtime: PhysicsDocument | null = null;
  let generated: SceneDocument | undefined;
  try {
    checkModelLoad(options);
    if (data.physics === undefined) {
      if (loaded.doc.root.userData.physicsSourceId) throw new Error('Missing simulation resources and state.');
      if ([...loaded.keys, ...loaded.timeline.snapshots, ...loaded.timeline.takes.flatMap((take) => take.frames)]
        .some((state) => state.simulation)) throw new Error('Recorded simulation states require their backend.');
      return { ...loaded, runtime: null };
    }
    const saved = data.physics as StoredPhysics;
    if (!saved || saved.version !== 1 || typeof saved.backend !== 'string' || !saved.snapshot)
      throw new Error('Unsupported saved simulation format.');
    const factory = resolveBackend(saved.backend);
    runtime = await PhysicsDocument.load(await decodeBundle(saved.bundle, options), factory, options);
    generated = runtime.document;
    for (const state of [...loaded.keys, ...loaded.timeline.snapshots, ...loaded.timeline.takes.flatMap((take) => take.frames)])
      validateRecordedSimulation(state.simulation, runtime.session.current.source!, runtime.session.current.snapshot!);
    await runtime.restore(loaded.doc, saved.snapshot);
    checkModelLoad(options);
    disposeDocument(generated);
    return { ...loaded, runtime };
  } catch (error) {
    runtime?.dispose();
    if (generated) disposeDocument(generated);
    disposeDocument(loaded.doc);
    throw error;
  }
}
