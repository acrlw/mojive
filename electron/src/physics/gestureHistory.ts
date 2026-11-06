import type { PhysicsSnapshot, PhysicsSource } from './contracts';
import type { SimulationSession } from './SimulationSession';

export interface PhysicsHistoryEntry {
  kind: 'physics';
  session: SimulationSession;
  source: PhysicsSource;
  captured: Promise<PhysicsSnapshot>;
}

/** Queue immediately, before the gesture's commands, without serializing the viewer's assets. */
export function capturePhysicsHistory(session: SimulationSession): PhysicsHistoryEntry {
  const source = session.current.source;
  if (!source) throw new Error('The simulation is no longer active.');
  return { kind: 'physics', session, source, captured: session.capture() };
}

export async function physicsHistoryChanged(before: PhysicsHistoryEntry, after: PhysicsHistoryEntry): Promise<boolean> {
  if (before.session !== after.session || before.source !== after.source)
    throw new Error('The simulation history belongs to a different model.');
  const snapshots = await Promise.all([before.captured, after.captured]);
  return JSON.stringify(snapshots[0]) !== JSON.stringify(snapshots[1]);
}

export async function restorePhysicsHistory(entry: PhysicsHistoryEntry, session?: SimulationSession): Promise<void> {
  const snapshot = await entry.captured;
  if (session !== entry.session || session.current.source !== entry.source)
    throw new Error('The simulation history belongs to a different model.');
  session.pause();
  // The backend validates and restores its complete state atomically. The existing
  // SceneBinding consumes the published frame, so geometry and materials stay intact.
  await session.execute({ type: 'restore', snapshot });
}
