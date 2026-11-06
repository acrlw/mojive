import type { ModelBundle } from '../resources/modelBundle';
import type { PhysicsBackend, PhysicsCommand, PhysicsFrame, PhysicsSnapshot, PhysicsSource, EnvironmentValues } from './contracts';
import type { BackendRequest, BackendResponse, WorkerResponse } from './workerProtocol';

/** Transport composition is shared by any engine running in a dedicated Worker. */
class WorkerTransport {
  private worker: Worker;
  private sequence = 0;
  private disposed = false;
  private pending = new Map<number, { resolve: (value: BackendResponse) => void; reject: (error: Error) => void }>();

  constructor(worker: Worker) {
    this.worker = worker;
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const reply = event.data, waiting = this.pending.get(reply.requestId);
      if (!waiting) return;
      this.pending.delete(reply.requestId);
      if (reply.ok) waiting.resolve(reply.value); else waiting.reject(new Error(reply.error));
    };
    worker.onerror = (event) => this.close(new Error(event.message || 'Simulation worker failed.'));
    worker.onmessageerror = () => this.close(new Error('Simulation worker response could not be decoded.'));
  }

  request(message: BackendRequest): Promise<BackendResponse> {
    if (this.disposed) return Promise.reject(new Error('The simulation worker has been closed.'));
    const requestId = ++this.sequence;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      try { this.worker.postMessage({ ...message, requestId }); }
      catch (error) { this.pending.delete(requestId); reject(error); }
    });
  }

  private close(error: Error): void {
    if (this.disposed) return;
    this.disposed = true;
    this.worker.terminate();
    for (const waiting of this.pending.values()) waiting.reject(error);
    this.pending.clear();
  }
  dispose(): void { this.close(new Error('The simulation worker has been closed.')); }
}

export class WorkerBackend implements PhysicsBackend {
  readonly source: PhysicsSource;
  readonly defaults: EnvironmentValues;
  private transport: WorkerTransport;
  private constructor(transport: WorkerTransport, source: PhysicsSource, defaults: EnvironmentValues) {
    this.transport = transport; this.source = source; this.defaults = defaults;
  }

  static async load(worker: Worker, bundle: ModelBundle): Promise<WorkerBackend> {
    const transport = new WorkerTransport(worker);
    try {
      const result = await transport.request({ type: 'load', bundle }) as { source: PhysicsSource; defaults: EnvironmentValues };
      return new WorkerBackend(transport, result.source, result.defaults);
    } catch (error) { transport.dispose(); throw error; }
  }
  async frame(): Promise<PhysicsFrame> { return await this.transport.request({ type: 'frame' }) as PhysicsFrame; }
  async environment(): Promise<EnvironmentValues> { return await this.transport.request({ type: 'environment' }) as EnvironmentValues; }
  async capture(): Promise<PhysicsSnapshot> { return await this.transport.request({ type: 'capture' }) as PhysicsSnapshot; }
  async execute(command: PhysicsCommand): Promise<PhysicsFrame> { return await this.transport.request({ type: 'execute', command }) as PhysicsFrame; }
  dispose(): void { this.transport.dispose(); }
}

export function loadMujocoBackend(bundle: ModelBundle): Promise<PhysicsBackend> {
  return WorkerBackend.load(new Worker(new URL('./backends/mujoco/worker.ts', import.meta.url), { type: 'module' }), bundle);
}
