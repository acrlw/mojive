import type { ModelBundle } from '../resources/modelBundle';
import { checkModelLoad, yieldModelLoad, type ModelLoadOptions } from '../resources/loading';
import type { PhysicsBackend, PhysicsCommand, PhysicsFrame, PhysicsSnapshot, PhysicsSource, EnvironmentValues } from './contracts';
import type { BackendRequest, BackendResponse, WorkerResponse } from './workerProtocol';

/** Transport composition is shared by any engine running in a dedicated Worker. */
class WorkerTransport {
  private worker: Worker;
  private sequence = 0;
  private disposed = false;
  private pending = new Map<number, { resolve: (value: BackendResponse) => void; reject: (error: Error) => void }>();

  constructor(worker: Worker, private readonly onProgress?: ModelLoadOptions['onProgress']) {
    this.worker = worker;
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const reply = event.data, waiting = this.pending.get(reply.requestId);
      if (!waiting) return;
      if ('progress' in reply) { this.onProgress?.(reply.progress); return; }
      this.pending.delete(reply.requestId);
      if (reply.ok) waiting.resolve(reply.value); else waiting.reject(new Error(reply.error));
    };
    worker.onerror = (event) => this.close(new Error(event.message || 'Simulation worker failed.'));
    worker.onmessageerror = () => this.close(new Error('Simulation worker response could not be decoded.'));
  }

  request(message: BackendRequest, transfer: Transferable[] = []): Promise<BackendResponse> {
    if (this.disposed) return Promise.reject(new Error('The simulation worker has been closed.'));
    const requestId = ++this.sequence;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      try { this.worker.postMessage({ ...message, requestId }, transfer); }
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
  dispose(error = new Error('The simulation worker has been closed.')): void { this.close(error); }
}

export class WorkerBackend implements PhysicsBackend {
  readonly source: PhysicsSource;
  readonly defaults: EnvironmentValues;
  private transport: WorkerTransport;
  private constructor(transport: WorkerTransport, source: PhysicsSource, defaults: EnvironmentValues) {
    this.transport = transport; this.source = source; this.defaults = defaults;
  }

  static async load(worker: Worker, bundle: ModelBundle, options?: ModelLoadOptions): Promise<WorkerBackend> {
    const transport = new WorkerTransport(worker, options?.onProgress);
    const abort = () => transport.dispose(new DOMException('Model loading was cancelled.', 'AbortError'));
    options?.signal?.addEventListener('abort', abort, { once: true });
    try {
      checkModelLoad(options);
      // Keep the caller's package for save/reopen; transfer a batched copy instead of
      // making postMessage synchronously serialize every mesh on the UI thread.
      const files: ModelBundle['files'] = [];
      let budget = performance.now();
      for (const file of bundle.files) {
        files.push({ path: file.path, bytes: file.bytes.slice() });
        if (performance.now() - budget > 8) { await yieldModelLoad(options); budget = performance.now(); }
      }
      checkModelLoad(options);
      const result = await transport.request({ type: 'load', bundle: { ...bundle, files } },
        files.map((file) => file.bytes.buffer as ArrayBuffer)) as { source: PhysicsSource; defaults: EnvironmentValues };
      checkModelLoad(options);
      return new WorkerBackend(transport, result.source, result.defaults);
    } catch (error) { transport.dispose(); throw error; }
    finally { options?.signal?.removeEventListener('abort', abort); }
  }
  async frame(): Promise<PhysicsFrame> { return await this.transport.request({ type: 'frame' }) as PhysicsFrame; }
  async environment(): Promise<EnvironmentValues> { return await this.transport.request({ type: 'environment' }) as EnvironmentValues; }
  async capture(): Promise<PhysicsSnapshot> { return await this.transport.request({ type: 'capture' }) as PhysicsSnapshot; }
  async execute(command: PhysicsCommand): Promise<PhysicsFrame> { return await this.transport.request({ type: 'execute', command }) as PhysicsFrame; }
  dispose(): void { this.transport.dispose(); }
}

export function loadMujocoBackend(bundle: ModelBundle, options?: ModelLoadOptions): Promise<PhysicsBackend> {
  return WorkerBackend.load(new Worker(new URL('./backends/mujoco/worker.ts', import.meta.url), { type: 'module' }), bundle, options);
}
