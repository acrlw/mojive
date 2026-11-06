export type ModelLoadStage = 'resources' | 'verify' | 'runtime' | 'compile' | 'source' | 'scene';

export interface ModelLoadProgress {
  stage: ModelLoadStage;
  completed?: number;
  total?: number;
  resource?: string;
  elapsedMs?: number;
}

export interface ModelLoadOptions {
  signal?: AbortSignal;
  onProgress?: (progress: ModelLoadProgress) => void;
}

export function checkModelLoad(options?: ModelLoadOptions): void {
  options?.signal?.throwIfAborted();
}

/** Give painting and cancellation a turn between renderer-side resource batches. */
export async function yieldModelLoad(options?: ModelLoadOptions): Promise<void> {
  checkModelLoad(options);
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  checkModelLoad(options);
}
