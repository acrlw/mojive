import test from 'node:test';
import assert from 'node:assert/strict';
import loadMujoco from '@mujoco/mujoco';
import { SimulationSession } from '../src/physics/SimulationSession.ts';
import { MujocoBackend } from '../src/physics/backends/mujoco/MujocoBackend.ts';
import { createModelBundle } from '../src/resources/modelBundle.ts';

const mj = await loadMujoco();
const bundle = await createModelBundle('History', 'model.xml', [{ path: 'model.xml', bytes: new TextEncoder().encode(`
<mujoco><option timestep="0.002"/><worldbody><body pos="0 0 1"><joint name="hinge" axis="0 1 0"/>
<geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.5"/></body></worldbody>
<actuator><motor joint="hinge" ctrlrange="-2 2"/></actuator></mujoco>`)}]);
async function session(t) {
  const runtime = new SimulationSession();
  t.after(() => runtime.dispose());
  const backend = await MujocoBackend.load(mj, bundle);
  await runtime.attach(backend);
  return { runtime, backend };
}

test('history restores full physical state and resumes deterministic simulation from the restored frame', async (t) => {
  const { runtime, backend } = await session(t);
  await runtime.execute({ type: 'controls', values: [{ id: backend.source.actuators[0].id, value: 0.5 }] });
  await runtime.execute({ type: 'step', count: 20 });
  const before = await runtime.capture();
  await runtime.execute({ type: 'step', count: 10 });
  const after = await runtime.capture();
  await runtime.stepBack();
  assert.deepEqual(await runtime.capture(), before);
  await runtime.execute({ type: 'step', count: 10 });
  assert.deepEqual(await runtime.capture(), after);
  await runtime.execute({ type: 'controls', values: [{ id: backend.source.actuators[0].id, value: 0.8 }] });
  await runtime.stepBack();
  assert.deepEqual(await runtime.capture(), after, 'An edited current frame returns to its saved state first.');
  await Promise.all([runtime.stepBack(), runtime.stepBack()]);
  assert.equal(runtime.current.frame.time, 0, 'Repeated clicks run in the command queue.');
  assert.equal(runtime.current.previousFrames, 0);
  await assert.rejects(runtime.stepBack(), /No previous/);
});

test('frame navigation preserves current model settings rather than undoing parameter edits', async (t) => {
  const { runtime } = await session(t);
  await runtime.execute({ type: 'environment', values: { timestep: 0.005, gravity: [0, 0, 0] } });
  assert.equal(runtime.current.previousFrames, 0);
  await runtime.execute({ type: 'step', count: 1 });
  await runtime.stepBack();
  assert.equal(runtime.current.frame.time, 0);
  assert.equal(runtime.current.environment.timestep, 0.005);
  assert.deepEqual(runtime.current.environment.gravity, [0, 0, 0]);
  await runtime.execute({ type: 'step', count: 1 });
  assert.equal(runtime.current.frame.time, 0.005);
});

test('history retains 1000 frames, replaces a resumed future, and clears for reset or a new model', async (t) => {
  const { runtime } = await session(t);
  for (let i = 0; i < 1004; i++) await runtime.execute({ type: 'step', count: 1 });
  assert.equal(runtime.current.previousFrames, 999);
  for (let i = 0; i < 999; i++) await runtime.stepBack();
  assert.ok(Math.abs(runtime.current.frame.time - 0.010) < 1e-12);
  assert.equal(runtime.current.previousFrames, 0);
  await runtime.execute({ type: 'step', count: 1 });
  assert.equal(runtime.current.previousFrames, 1, 'Discarded future frames cannot be revisited.');
  await runtime.execute({ type: 'reset' });
  assert.equal(runtime.current.previousFrames, 0);
  assert.equal(runtime.current.frame.time, 0);
  await runtime.execute({ type: 'step', count: 1 });
  await runtime.attach(await MujocoBackend.load(mj, bundle));
  assert.equal(runtime.current.previousFrames, 0);
  assert.equal(runtime.current.frame.time, 0);
});

test('failed history restoration keeps the cursor and the next attempt restores the same frame', async (t) => {
  const { runtime, backend } = await session(t);
  await runtime.execute({ type: 'step', count: 5 });
  const before = runtime.current;
  const execute = backend.execute.bind(backend);
  backend.execute = async (command) => {
    if (command.type === 'restore') throw new Error('Restore rejected');
    return execute(command);
  };
  await assert.rejects(runtime.stepBack(), /Restore rejected/);
  assert.equal(runtime.current.previousFrames, before.previousFrames);
  assert.equal(runtime.current.frame, before.frame);
  backend.execute = execute;
  await runtime.stepBack();
  assert.equal(runtime.current.frame.time, 0);
  assert.equal(runtime.current.error, null);
});
