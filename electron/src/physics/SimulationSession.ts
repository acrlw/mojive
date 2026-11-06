import type {
  PhysicsBackend,
  PhysicsCommand,
  PhysicsFrame,
  PhysicsSnapshot,
  EnvironmentValues,
  PhysicsSource,
} from './contracts';
import { validateFrame } from './frame.ts';

export interface SimulationState {
  source: PhysicsSource | null;
  frame: PhysicsFrame | null;
  snapshot: PhysicsSnapshot | null;
  environment: EnvironmentValues;
  playing: boolean;
  speed: number;
  previousFrames: number;
  error: string | null;
}

type TargetCommand = Extract<PhysicsCommand, { type: 'jointPose' | 'perturbTarget' }>;
interface PendingTarget {
  command: TargetCommand;
  completion: Promise<PhysicsFrame>;
}

function sameTarget(left: TargetCommand, right: TargetCommand): boolean {
  return left.type === 'jointPose'
    ? right.type === 'jointPose' && left.id === right.id
    : right.type === 'perturbTarget' && left.target.bodyId === right.target.bodyId &&
      left.target.mode === right.target.mode;
}

/** Serial command ownership and playback scheduling, independent of UI and engines. */
export class SimulationSession {
  private backend: PhysicsBackend | null = null;
  private state: SimulationState = {
    source: null,
    frame: null,
    snapshot: null,
    environment: {},
    playing: false,
    speed: 1,
    previousFrames: 0,
    error: null,
  };
  private listeners = new Set<(state: SimulationState) => void>();
  private queue: Promise<unknown> = Promise.resolve();
  private pendingTarget: PendingTarget | undefined;
  private generation = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private lastTick = 0;
  private elapsed = 0;
  private loadEpoch = 0;
  private playbackEpoch = 0;
  private readonly history: (PhysicsSnapshot | undefined)[] = new Array(1000);
  private historyStart = 0;
  private historySize = 0;
  private historyCursor = -1;
  private historyDirty = false;

  private clearHistory(): void {
    this.history.fill(undefined);
    this.historyStart = 0;
    this.historySize = 0;
    this.historyCursor = -1;
    this.historyDirty = false;
  }
  private appendHistory(snapshot: PhysicsSnapshot): void {
    // A resumed branch replaces its future. The ring bounds retained states without shifting them each frame.
    while (this.historySize > this.historyCursor + 1) {
      this.history[(this.historyStart + --this.historySize) % this.history.length] = undefined;
    }
    if (this.historySize === this.history.length) {
      this.historyStart = (this.historyStart + 1) % this.history.length;
      this.historySize--;
    }
    this.history[(this.historyStart + this.historySize++) % this.history.length] = snapshot;
    this.historyCursor = this.historySize - 1;
    this.historyDirty = false;
  }
  private get previousFrames(): number {
    return Math.max(0, this.historyCursor + (this.historyDirty ? 1 : 0));
  }

  get current(): SimulationState {
    return this.state;
  }
  subscribe(listener: (state: SimulationState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
  private publish(patch: Partial<SimulationState>): void {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) listener(this.state);
  }

  async attach(candidate: PhysicsBackend): Promise<void> {
    const epoch = ++this.loadEpoch;
    try {
      const [frame, snapshot] = await Promise.all([candidate.frame(), candidate.capture()]);
      validateFrame(candidate.source, frame);
      if (snapshot.sourceId !== candidate.source.id)
        throw new Error('Simulation state does not match its source.');
      if (epoch !== this.loadEpoch) throw new Error('Simulation loading was superseded.');
      this.pause();
      this.generation++;
      this.pendingTarget = undefined;
      this.backend?.dispose();
      this.backend = candidate;
      this.elapsed = 0;
      this.clearHistory();
      this.appendHistory(snapshot);
      this.publish({
        source: candidate.source,
        frame,
        snapshot,
        environment: snapshot.environment,
        previousFrames: 0,
        error: null,
      });
    } catch (error) {
      candidate.dispose();
      throw error;
    }
  }

  private enqueue<T>(action: (backend: PhysicsBackend) => Promise<T>): Promise<T> {
    // Captures and discrete commands delimit gestures; later targets cannot replace work before them.
    this.pendingTarget = undefined;
    const backend = this.backend,
      generation = this.generation;
    const operation = this.queue.then(() => {
      if (!backend || backend !== this.backend || generation !== this.generation)
        throw new Error('The simulation is no longer active.');
      return action(backend);
    });
    this.queue = operation.catch(() => undefined);
    return operation;
  }

  async execute(command: PhysicsCommand): Promise<PhysicsFrame> {
    if (command.type === 'jointPose') this.pause();
    return this.enqueue((backend) => this.apply(backend, command));
  }

  /** Keep the latest unstarted pointer target without building a queue of obsolete poses. */
  updateTarget(command: TargetCommand): Promise<PhysicsFrame> {
    // Joint posing edits qpos directly; physical perturbation intentionally continues simulation.
    if (command.type === 'jointPose') this.pause();
    if (this.pendingTarget && sameTarget(this.pendingTarget.command, command)) {
      this.pendingTarget.command = command;
      return this.pendingTarget.completion;
    }
    const pending: PendingTarget = {
      command,
      completion: this.enqueue((backend) => {
        if (this.pendingTarget === pending) this.pendingTarget = undefined;
        return this.apply(backend, pending.command);
      }),
    };
    this.pendingTarget = pending;
    return pending.completion;
  }

  stepBack(): Promise<PhysicsFrame> {
    this.pause();
    this.elapsed = 0;
    return this.enqueue((backend) => {
      if (!this.previousFrames) throw new Error('No previous simulation frame is available.');
      const cursor = this.historyCursor - (this.historyDirty ? 0 : 1);
      const snapshot = this.history[(this.historyStart + cursor) % this.history.length]!;
      // Frame navigation restores dynamics; model settings belong to document undo.
      return this.apply(backend, { type: 'restore', snapshot: { ...snapshot, environment: this.state.environment } }, cursor);
    });
  }

  private async apply(
    backend: PhysicsBackend,
    command: PhysicsCommand,
    historyCursor?: number,
  ): Promise<PhysicsFrame> {
    try {
      const frame = await backend.execute(command);
      validateFrame(backend.source, frame);
      const snapshot = await backend.capture();
      if (backend === this.backend) {
        if (historyCursor !== undefined) {
          this.historyCursor = historyCursor;
          this.historyDirty = false;
        } else if (command.type === 'reset' || command.type === 'restore') {
          this.clearHistory();
          this.appendHistory(snapshot);
        } else if (command.type === 'step') this.appendHistory(snapshot);
        else if (command.type !== 'environment' && command.type !== 'resetEnvironment') this.historyDirty = true;
        this.publish({
          frame,
          snapshot,
          environment: snapshot.environment,
          previousFrames: this.previousFrames,
          error: null,
        });
      }
      return frame;
    } catch (error) {
      if (backend === this.backend) {
        this.pause();
        this.publish({ error: error instanceof Error ? error.message : String(error) });
      }
      throw error;
    }
  }

  capture(): Promise<PhysicsSnapshot> {
    return this.enqueue((backend) => backend.capture());
  }

  /** Reserve cancellation's place before the next gesture, even while its baseline is pending. */
  restoreCaptured(captured: Promise<PhysicsSnapshot>): Promise<PhysicsFrame> {
    this.pause();
    const generation = this.generation;
    // Observe rejection now: the queue may still be executing an earlier slow command.
    const result = captured.then(
      (snapshot) => ({ ok: true as const, snapshot }),
      (error: unknown) => ({ ok: false as const, error }),
    );
    return this.enqueue(async (backend) => {
      const outcome = await result;
      if (backend !== this.backend || generation !== this.generation)
        throw new Error('The simulation is no longer active.');
      if (!outcome.ok) throw outcome.error;
      return this.apply(backend, { type: 'restore', snapshot: outcome.snapshot });
    });
  }

  setSpeed(speed: number): void {
    if (!Number.isFinite(speed) || speed <= 0 || speed > 16)
      throw new Error('Playback speed must be between 0 and 16.');
    this.publish({ speed });
  }
  play(): void {
    if (!this.backend || this.state.playing) return;
    this.lastTick = performance.now();
    this.elapsed = 0;
    this.playbackEpoch++;
    this.publish({ playing: true });
    this.schedule();
  }
  pause(): void {
    this.playbackEpoch++;
    clearTimeout(this.timer);
    this.timer = undefined;
    if (this.state.playing) this.publish({ playing: false });
  }
  private schedule(): void {
    this.timer = setTimeout(() => {
      void this.tick();
    }, 16);
  }
  private async tick(): Promise<void> {
    if (!this.state.playing || !this.backend) return;
    const generation = this.generation,
      playback = this.playbackEpoch,
      now = performance.now();
    this.elapsed += Math.min((now - this.lastTick) / 1000, 0.1) * this.state.speed;
    this.lastTick = now;
    const timestep = this.state.environment.timestep;
    if (typeof timestep !== 'number' || !Number.isFinite(timestep) || timestep <= 0) {
      this.pause();
      this.publish({ error: 'The backend did not provide a valid simulation timestep.' });
      return;
    }
    const count = Math.min(250, Math.floor(this.elapsed / timestep));
    try {
      if (count > 0) {
        await this.execute({ type: 'step', count });
        if (playback === this.playbackEpoch) this.elapsed -= count * timestep;
      }
      if (this.state.playing && this.generation === generation && playback === this.playbackEpoch)
        this.schedule();
    } catch {
      /* execute publishes the failure and stops playback. */
    }
  }

  detach(): void {
    this.pause();
    this.generation++;
    this.pendingTarget = undefined;
    this.loadEpoch++;
    this.elapsed = 0;
    this.backend?.dispose();
    this.backend = null;
    this.clearHistory();
    this.publish({
      source: null,
      frame: null,
      snapshot: null,
      environment: {},
      previousFrames: 0,
      error: null,
    });
  }
  dispose(): void {
    this.detach();
    this.listeners.clear();
  }
}
