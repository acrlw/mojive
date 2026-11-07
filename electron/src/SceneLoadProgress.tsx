import { useEffect, useRef, useState } from 'react';
import { Button } from './components';
import { InfoDialog } from './InfoDialog';
import { t } from './i18n';
import type { ModelLoadOptions, ModelLoadProgress } from './resources/loading';
import './scene-load-progress.css';

export interface SceneLoadState {
  label: string;
  startedAt: number;
  progress: ModelLoadProgress;
}

/** One candidate owns progress and cancellation; the installed scene stays intact. */
export function useSceneLoading() {
  const active = useRef<AbortController | null>(null);
  const [state, setState] = useState<SceneLoadState | null>(null);
  useEffect(() => () => { active.current?.abort(); active.current = null; }, []);
  function cancel() {
    active.current?.abort();
    active.current = null;
    setState(null);
  }
  async function run<T>(label: string, task: (options: ModelLoadOptions) => Promise<T>, prepare?: (signal: AbortSignal) => Promise<boolean>) {
    active.current?.abort();
    const controller = new AbortController();
    active.current = controller;
    setState(null);
    try {
      // Resolve unsaved changes before starting resources or showing load progress.
      if (prepare && !await prepare(controller.signal)) return;
      controller.signal.throwIfAborted();
      setState({ label, startedAt: performance.now(), progress: { stage: 'resources' } });
      // Paint feedback before resource enumeration or model decoding starts.
      await new Promise<void>((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
      controller.signal.throwIfAborted();
      return await task({ signal: controller.signal, onProgress: (progress) => {
        if (active.current === controller && !controller.signal.aborted)
          setState((previous) => previous && ({ ...previous, progress }));
      } });
    } catch (error) {
      if (!controller.signal.aborted) throw error;
    } finally {
      if (active.current === controller) { active.current = null; setState(null); }
    }
  }
  return { state, run, cancel, isActive: () => active.current !== null };
}

export const modelLoadStageLabels: Record<ModelLoadProgress['stage'], string> = {
  resources: 'Reading resources',
  verify: 'Checking resources',
  runtime: 'Starting simulation engine',
  compile: 'Compiling model',
  source: 'Preparing scene data',
  scene: 'Building scene',
};

export function SceneLoadProgress({ state, onCancel }: { state: SceneLoadState; onCancel: () => void }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const update = () => setElapsed((performance.now() - state.startedAt) / 1000);
    update();
    const timer = setInterval(update, 250);
    return () => clearInterval(timer);
  }, [state.startedAt]);
  const { progress } = state;
  const determinate = progress.total !== undefined && progress.total > 0 && progress.completed !== undefined;
  return <InfoDialog label={t('Loading model')} onClose={onCancel} className="scene-load-dialog">
    <div data-testid="scene-load-progress" data-stage={progress.stage} data-started-at={state.startedAt}>
      <div className="scene-load-heading"><strong>{t('Loading model')}</strong><span>{elapsed.toFixed(1)} s</span></div>
      <div className="scene-load-name" title={state.label}>{state.label}</div>
      <div className="scene-load-stage" role="status"><span>{t(modelLoadStageLabels[progress.stage])}</span>
        {determinate && <span>{progress.completed} / {progress.total}</span>}
      </div>
      <div className={`scene-load-bar${determinate ? '' : ' indeterminate'}`} role="progressbar"
        aria-label={t(modelLoadStageLabels[progress.stage])} aria-valuemin={0}
        aria-valuenow={determinate ? progress.completed : undefined} aria-valuemax={determinate ? progress.total : undefined}>
        <span style={determinate ? { width: `${Math.min(100, 100 * progress.completed! / progress.total!)}%` } : undefined} />
      </div>
      <div className="scene-load-footer"><span title={progress.resource}>{progress.resource ?? ''}</span>
        <Button variant="secondary" data-testid="scene-load-cancel" onClick={onCancel}>Cancel</Button>
      </div>
    </div>
  </InfoDialog>;
}
