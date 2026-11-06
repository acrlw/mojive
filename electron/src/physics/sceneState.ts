import type { SceneDocument } from '../contracts';
import { applyPose, assertPoseCompatible, capturePose, type RecordedSceneState } from '../editor';
import type { PhysicsDocument } from './PhysicsDocument';
import { validateEnvironment } from './environment';
import type { PhysicsSnapshot, PhysicsSource } from './contracts';

/** Capture the published frame and its corresponding integration state as one unit. */
export function captureSceneState(document: SceneDocument, runtime: PhysicsDocument | null): RecordedSceneState {
  if (!runtime) return { pose: capturePose(document) };
  if (runtime.document !== document || !runtime.session.current.snapshot)
    throw new Error('The active simulation does not match this scene.');
  return { pose: capturePose(document), simulation: structuredClone(runtime.session.current.snapshot) };
}

export function validateRecordedSimulation(value: PhysicsSnapshot | undefined, source: PhysicsSource, template: PhysicsSnapshot): void {
  if (!value || value.sourceId !== source.id || value.backend !== source.backend || value.format !== template.format ||
    !Array.isArray(value.state) || value.state.length !== template.state.length || !value.state.every(Number.isFinite))
    throw new Error('The recorded simulation state is incompatible with this model.');
  if (!value.environment || typeof value.environment !== 'object' || Array.isArray(value.environment) ||
    Object.keys(value.environment).length !== source.environment.length)
    throw new Error('The recorded simulation state has incomplete environment settings.');
  validateEnvironment(source.environment, value.environment);
}

/** Backend restoration succeeds before any local viewer objects or cursor are changed. */
export async function applySceneState(document: SceneDocument, state: RecordedSceneState, runtime: PhysicsDocument | null): Promise<void> {
  assertPoseCompatible(document, state.pose);
  if (!runtime) {
    if (state.simulation || document.root.userData.physicsSourceId)
      throw new Error('This recorded state requires its simulation backend.');
    applyPose(document, state.pose);
    return;
  }
  if (runtime.document !== document) throw new Error('The active simulation does not match this scene.');
  validateRecordedSimulation(state.simulation, runtime.session.current.source!, runtime.session.current.snapshot!);
  runtime.session.pause();
  await runtime.session.execute({ type: 'restore', snapshot: state.simulation! });
  applyPose(document, state.pose, true);
}
