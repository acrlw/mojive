import type { ModelBundle } from '../resources/modelBundle';
import type { ModelLoadProgress } from '../resources/loading';
import type { PhysicsCommand, PhysicsSource, PhysicsFrame, PhysicsSnapshot, EnvironmentValues } from './contracts';

export type BackendRequest =
  | { type: 'load'; bundle: ModelBundle }
  | { type: 'frame' | 'environment' | 'capture' }
  | { type: 'execute'; command: PhysicsCommand };
export type BackendResponse = PhysicsFrame | PhysicsSnapshot | EnvironmentValues |
  { source: PhysicsSource; defaults: EnvironmentValues };
export type WorkerRequest = BackendRequest & { requestId: number };
export type WorkerResponse = { requestId: number } & (
  | { ok: true; value: BackendResponse }
  | { ok: false; error: string }
  | { progress: ModelLoadProgress }
);
