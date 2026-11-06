import test from 'node:test';
import assert from 'node:assert/strict';
import { SimulationSession } from '../src/physics/SimulationSession.ts';
import { capturePhysicsHistory, physicsHistoryChanged, restorePhysicsHistory } from '../src/physics/gestureHistory.ts';

async function fixture(t) {
  const source = { id: 'model', bodies: [], joints: [], cameras: [], lights: [], actuators: [], geometries: [],
    toJSON() { throw new Error('History must not serialize scene assets.'); } };
  let state = { sourceId: source.id, backend: 'fixture', format: 'full-state', state: [0, 0, 0],
    environment: { gravity: [0, 0, -9.81] } };
  const backend = {
    source, defaults: state.environment,
    async capture() { return structuredClone(state); },
    async frame() { return { sourceId: source.id, time: state.state[0],
      bodyTransforms: new Float64Array(), jointFrames: new Float64Array(), cameraTransforms: new Float64Array(),
      lightFrames: new Float64Array(), jointPositions: new Float64Array(), controls: new Float64Array() }; },
    async execute(command) {
      if (command.type === 'restore') state = structuredClone(command.snapshot);
      if (command.type === 'controls') state.state[1] = command.values[0].value;
      if (command.type === 'step') { state.state[0] += command.count * 0.002; state.state[2] += command.count * state.state[1]; }
      if (command.type === 'perturbTarget') state.interaction = { target: command.target };
      if (command.type === 'clearPerturbation') delete state.interaction;
      if (command.type === 'environment') state.environment = { ...state.environment, ...command.values };
      return this.frame();
    },
    dispose() {},
  };
  const session = new SimulationSession();
  await session.attach(backend);
  t.after(() => session.dispose());
  return { session, backend };
}

test('gesture capture brackets queued commands and retains complete physics state without serializing assets', async (t) => {
  const { session } = await fixture(t);
  const initial = await session.capture();
  const before = capturePhysicsHistory(session);
  const updates = [
    session.execute({ type: 'controls', values: [{ id: 'motor', value: 0.7 }] }),
    session.execute({ type: 'step', count: 3 }),
    session.execute({ type: 'environment', values: { gravity: [0, 0, 0] } }),
    session.execute({ type: 'perturbTarget', target: { bodyId: 'body', position: [1, 2, 3], rotation: [0, 0, 0, 1] } }),
  ];
  const after = capturePhysicsHistory(session);
  await Promise.all(updates);
  assert.deepEqual(await before.captured, initial);
  const changed = await session.capture();
  assert.deepEqual(await after.captured, changed);
  assert.equal(await physicsHistoryChanged(before, after), true);
  assert.ok(changed.interaction);
  await restorePhysicsHistory(before, session);
  assert.deepEqual(await session.capture(), initial, 'Undo restores dynamics, controls, environment, and transient interaction together.');
  await restorePhysicsHistory(after, session);
  assert.deepEqual(await session.capture(), changed, 'Redo restores the exact state, not just the displayed body pose.');
});

test('cancel restores the gesture baseline and unchanged gestures do not create history', async (t) => {
  const { session } = await fixture(t);
  const before = capturePhysicsHistory(session);
  assert.equal(await physicsHistoryChanged(before, capturePhysicsHistory(session)), false);
  await session.execute({ type: 'controls', values: [{ id: 'motor', value: 1 }] });
  await restorePhysicsHistory(before, session);
  assert.deepEqual(await session.capture(), await before.captured);
});

test('capture failures cannot commit or restore a stale baseline', async (t) => {
  const { session, backend } = await fixture(t);
  const capture = backend.capture.bind(backend);
  backend.capture = async () => { throw new Error('Capture rejected'); };
  const failed = capturePhysicsHistory(session);
  await assert.rejects(failed.captured, /Capture rejected/);
  backend.capture = capture;
  const after = capturePhysicsHistory(session);
  await assert.rejects(physicsHistoryChanged(failed, after), /Capture rejected/);
  await assert.rejects(restorePhysicsHistory(failed, session), /Capture rejected/);
  assert.deepEqual(await session.capture(), await after.captured);
});

test('failed restore leaves the entry reusable and a replaced model cannot receive it', async (t) => {
  const { session, backend } = await fixture(t);
  const before = capturePhysicsHistory(session);
  await session.execute({ type: 'controls', values: [{ id: 'motor', value: 0.9 }] });
  const changed = await session.capture();
  const execute = backend.execute.bind(backend);
  backend.execute = async (command) => {
    if (command.type === 'restore') throw new Error('Restore rejected');
    return execute(command);
  };
  await assert.rejects(restorePhysicsHistory(before, session), /Restore rejected/);
  assert.deepEqual(await session.capture(), changed);
  backend.execute = execute;
  await restorePhysicsHistory(before, session);
  assert.deepEqual(await session.capture(), await before.captured);
  const other = await fixture(t);
  await assert.rejects(restorePhysicsHistory(before, other.session), /different model/);
  await session.attach({ ...backend, source: { ...backend.source } });
  await assert.rejects(restorePhysicsHistory(before, session), /different model/);
});
