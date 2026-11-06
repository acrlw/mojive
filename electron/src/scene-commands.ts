import type { SceneDocument } from './contracts';
import type { PerturbationTarget, Transform } from './physics/contracts';
export type { PerturbationTarget } from './physics/contracts';

export interface PerturbationCommands {
  bodyAt(entityId: string): (Transform & { id: string }) | null;
  update(target: PerturbationTarget): void;
  clear(): void;
}

/** Runtime command routing is deliberately excluded from serializable scene metadata. */
export interface SceneCommands {
  setJointPosition(id: string, position: number[]): void;
  perturbation?: PerturbationCommands;
}
const owners = new WeakMap<SceneDocument, SceneCommands>();
export function bindSceneCommands(document: SceneDocument, commands: SceneCommands): () => void {
  owners.set(document, commands);
  return () => { if (owners.get(document) === commands) owners.delete(document); };
}
export function sceneCommands(document: SceneDocument): SceneCommands | undefined {
  return owners.get(document);
}
