import { Button, Numeric } from '../components';
import { t } from '../i18n';
import { actionParts, type InputBindings } from '../input-bindings';
import type { SimulationState } from './SimulationSession';
import './simulation.css';

export function SimulationToolbar({
  state,
  recording,
  bindings,
  onToggle,
  onPrevious,
  onStep,
  onReset,
  onSpeed,
  onEnvironment,
}: {
  state: SimulationState;
  recording: boolean;
  bindings: InputBindings;
  onToggle: () => void;
  onPrevious: () => void;
  onStep: () => void;
  onReset: () => void;
  onSpeed: (speed: number) => void;
  onEnvironment: () => void;
}) {
  return (
    <div className="simulation-toolbar" role="toolbar" aria-label={t('Simulation')}>
      <div className="simulation-actions">
        <Button icon="reset" title="Reset simulation" disabled={recording} onClick={onReset} />
        <Button icon="env" title="Environment" onClick={onEnvironment} />
      </div>
      <div className="simulation-transport">
        <Button
          icon="stepb"
          title="Previous simulation frame"
          hintKeys={actionParts(bindings, 'previousFrame')}
          disabled={state.playing || recording || !state.previousFrames}
          onClick={onPrevious}
        />
        <Button
          icon={state.playing ? 'pause' : 'play'}
          active={state.playing}
          disabled={recording}
          title={state.playing ? 'Pause simulation' : 'Run simulation'}
          hintKeys={actionParts(bindings, 'playback')}
          onClick={onToggle}
        />
        <Button
          icon="stepf"
          title="Step simulation"
          hintKeys={actionParts(bindings, 'stepFrame')}
          disabled={state.playing || recording}
          onClick={onStep}
        />
      </div>
      <div className="simulation-speed" data-tooltip={t('Simulation speed')}>
        <Numeric
          label="Simulation speed"
          value={state.speed}
          min={0.01}
          max={16}
          step={0.01}
          precision={2}
          unit="×"
          onChange={onSpeed}
        />
      </div>
    </div>
  );
}
