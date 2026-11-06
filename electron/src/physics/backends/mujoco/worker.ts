import loadMujoco from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
import { MujocoBackend } from './MujocoBackend';
import type { WorkerRequest, WorkerResponse, BackendResponse } from '../../workerProtocol';
import type { PhysicsFrame } from '../../contracts';

let backend: MujocoBackend | undefined;
const runtime = loadMujoco({ locateFile: (file: string) => file.endsWith('.wasm') ? wasmUrl : file });
let queue: Promise<void> = Promise.resolve();

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  queue = queue.then(async () => {
    try {
      let value: BackendResponse;
      if (request.type === 'load') {
        const candidate = await MujocoBackend.load(await runtime, request.bundle);
        backend?.dispose(); backend = candidate;
        value = { source: backend.source, defaults: backend.defaults };
      } else {
        if (!backend) throw new Error('No simulation model is loaded.');
        value = request.type === 'execute' ? await backend.execute(request.command)
          : await backend[request.type]();
      }
      const response: WorkerResponse = { requestId: request.requestId, ok: true, value };
      const transfers: ArrayBuffer[] = [];
      if ('bodyTransforms' in value && value.bodyTransforms instanceof Float64Array) {
        transfers.push(value.bodyTransforms.buffer as ArrayBuffer);
        transfers.push((value.jointPositions as Float64Array).buffer as ArrayBuffer);
        transfers.push((value.controls as Float64Array).buffer as ArrayBuffer);
        transfers.push((value.jointFrames as Float64Array).buffer as ArrayBuffer);
        transfers.push((value.cameraTransforms as Float64Array).buffer as ArrayBuffer);
        transfers.push((value.lightFrames as Float64Array).buffer as ArrayBuffer);
        for (const update of (value as PhysicsFrame).geometryUpdates ?? [])
          transfers.push(update.positions.buffer as ArrayBuffer, update.normals.buffer as ArrayBuffer);
      }
      self.postMessage(response, { transfer: transfers });
    } catch (error) {
      const response: WorkerResponse = { requestId: request.requestId, ok: false,
        error: error instanceof Error ? error.message : String(error) };
      self.postMessage(response);
    }
  });
};
