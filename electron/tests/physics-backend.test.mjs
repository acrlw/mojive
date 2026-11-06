import test from 'node:test';
import assert from 'node:assert/strict';
import loadMujoco from '@mujoco/mujoco';
import { createModelBundle, modelPath } from '../src/resources/modelBundle.ts';
import { MujocoBackend } from '../src/physics/backends/mujoco/MujocoBackend.ts';

const mj = await loadMujoco();
const encode = (text) => new TextEncoder().encode(text);
const xml = `<mujoco model="viewer acceptance"><option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="ball" pos="0 0 1"><freejoint name="free"/><geom type="sphere" size="0.1"/></body>
    <body name="hinge" pos="1 0 1"><joint name="hinge" axis="0 1 0" range="-90 90"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.5"/></body>
  </worldbody>
  <actuator><motor name="motor" joint="hinge" ctrlrange="-2 2"/></actuator>
</mujoco>`;
const bundle = await createModelBundle('Acceptance', 'model.xml', [{ path: 'model.xml', bytes: encode(xml) }]);
async function backend(t) {
  const engine = await MujocoBackend.load(mj, bundle);
  t.after(() => engine.dispose());
  return engine;
}

test('model packages canonicalize paths and verify all content identities', async () => {
  assert.equal(modelPath('a/../b\\c.xml'), 'b/c.xml');
  for (const path of ['../escape', '/absolute', 'C:\\file', 'https://file']) assert.throws(() => modelPath(path));
  await assert.rejects(createModelBundle('x', 'a', [{ path: 'a', bytes: encode('a') }, { path: './a', bytes: encode('b') }]), /Duplicate/);
  await assert.rejects(MujocoBackend.load(mj, { ...bundle, id: 'stale' }), /identity/);
});

test('real WASM loads relative includes and rejects missing resources without corrupting another model', async (t) => {
  const live = await backend(t);
  const packageWithInclude = await createModelBundle('Included', 'models/main.xml', [
    { path: 'models/main.xml', bytes: encode('<mujoco><include file="parts/body.xml"/></mujoco>') },
    { path: 'models/parts/body.xml', bytes: encode('<mujoco><worldbody><geom type="box" size="1 1 1"/></worldbody></mujoco>') },
  ]);
  const included = await MujocoBackend.load(mj, packageWithInclude);
  assert.equal(included.source.geometries.length, 1);
  included.dispose();
  const missing = await createModelBundle('Missing', 'model.xml', [{ path: 'model.xml', bytes: encode('<mujoco><include file="absent.xml"/></mujoco>') }]);
  await assert.rejects(MujocoBackend.load(mj, missing));
  assert.equal((await live.frame()).time, 0);
  await live.execute({ type: 'step', count: 10 });
  assert.ok((await live.frame()).time > 0);
});

test('simulation passes ten seconds and publishes actual world transforms', async (t) => {
  const engine = await backend(t);
  const frame = await engine.execute({ type: 'step', count: 6000 });
  assert.ok(frame.time > 11.99);
  const ball = engine.source.bodies.findIndex((body) => body.name === 'ball');
  assert.ok(Math.abs(frame.bodyTransforms[ball * 7 + 2] - 0.1) < 0.01);
  assert.ok(frame.bodyTransforms.every(Number.isFinite));
});

test('environment inherits model defaults, validates atomically, and resets independently', async (t) => {
  const engine = await backend(t);
  assert.deepEqual(engine.defaults.gravity, [0, 0, -9.81]);
  await engine.execute({ type: 'environment', values: { gravity: [0, 0, 0], timestep: 0.005 } });
  await assert.rejects(engine.execute({ type: 'environment', values: { gravity: [1, 2, 3], timestep: -1 } }), /Invalid/);
  assert.deepEqual((await engine.environment()).gravity, [0, 0, 0]);
  await engine.execute({ type: 'step', count: 100 });
  const time = (await engine.frame()).time;
  await engine.execute({ type: 'resetEnvironment' });
  assert.equal((await engine.frame()).time, time);
  assert.deepEqual(await engine.environment(), engine.defaults);
  await engine.execute({ type: 'environment', values: { timestep: 0.003 } });
  await engine.execute({ type: 'reset' });
  assert.equal((await engine.frame()).time, 0);
  assert.equal((await engine.environment()).timestep, 0.003);
});

test('full state includes control, velocity, perturbation and environment with deterministic continuation', async (t) => {
  const engine = await backend(t);
  const motor = engine.source.actuators[0].id;
  const body = engine.source.bodies.find((item) => item.name === 'ball').id;
  await engine.execute({ type: 'controls', values: [{ id: motor, value: 0.5 }] });
  await engine.execute({ type: 'perturb', bodyId: body, force: [0.01, 0, 0], torque: [0, 0.001, 0] });
  await engine.execute({ type: 'step', count: 180 });
  const snapshot = await engine.capture();
  await engine.execute({ type: 'step', count: 100 });
  const expected = await engine.capture();
  await engine.execute({ type: 'environment', values: { gravity: [0, 0, -1] } });
  await engine.execute({ type: 'restore', snapshot });
  assert.deepEqual(await engine.capture(), snapshot);
  await engine.execute({ type: 'step', count: 100 });
  assert.deepEqual(await engine.capture(), expected);
  await assert.rejects(engine.execute({ type: 'restore', snapshot: { ...snapshot, sourceId: 'different' } }), /different/);
  assert.deepEqual(await engine.capture(), expected);
});

test('controls and joint writes use explicit IDs, validate batches, and convert neutral quaternions', async (t) => {
  const engine = await backend(t);
  const motor = engine.source.actuators[0].id;
  await assert.rejects(engine.execute({ type: 'controls', values: [{ id: motor, value: 1 }, { id: 'missing', value: 1 }] }));
  assert.equal((await engine.frame()).controls[0], 0);
  await assert.rejects(engine.execute({ type: 'controls', values: [{ id: motor, value: 3 }] }));
  const joint = engine.source.joints.find((item) => item.name === 'free');
  await engine.execute({ type: 'jointPose', id: joint.id, position: [0, 0, 2, 0, 0, 1, 1] });
  const pose = (await engine.frame()).jointPositions.slice(joint.offset, joint.offset + 7);
  assert.deepEqual(Array.from(pose.slice(0, 3)), [0, 0, 2]);
  assert.ok(Math.abs(pose[5] - Math.SQRT1_2) < 1e-12);
  assert.ok(Math.abs(pose[6] - Math.SQRT1_2) < 1e-12);
});

test('repeated loads and idempotent disposal release independent WASM handles', async () => {
  for (let i = 0; i < 8; i++) {
    const engine = await MujocoBackend.load(mj, bundle);
    await engine.execute({ type: 'step', count: 5 });
    engine.dispose(); engine.dispose();
    await assert.rejects(engine.frame(), /closed/);
  }
});

test('held perturbation targets produce forces, restore deterministically, and release completely', async (t) => {
  const engine = await backend(t);
  await engine.execute({ type: 'environment', values: { gravity: [0, 0, 0] } });
  const bodyIndex = engine.source.bodies.findIndex((item) => item.name === 'ball');
  const bodyId = engine.source.bodies[bodyIndex].id;
  const initial = await engine.capture();
  const target = { bodyId, position: [0.2, 0, 1], rotation: [0, 0, 0, 1],
    localPosition: [0, 0, 0], mode: 'translation', strength: 1 };
  await engine.execute({ type: 'perturbTarget', target });
  await engine.execute({ type: 'step', count: 80 });
  assert.ok((await engine.frame()).bodyTransforms[bodyIndex * 7] > 0.01);
  const held = await engine.capture();
  assert.deepEqual(held.interaction.target, target);
  await engine.execute({ type: 'step', count: 40 });
  const expected = await engine.capture();
  await engine.execute({ type: 'restore', snapshot: held });
  await engine.execute({ type: 'step', count: 40 });
  assert.deepEqual(await engine.capture(), expected);
  await assert.rejects(engine.execute({ type: 'perturbTarget', target: { ...target, bodyId: engine.source.bodies[0].id } }), /dynamic/);
  assert.deepEqual(await engine.capture(), expected);
  await engine.execute({ type: 'clearPerturbation' });
  assert.equal((await engine.capture()).interaction, undefined);
  // A cleared spring must not pull the body back to its old target after a pose edit.
  const free = engine.source.joints.find((joint) => joint.name === 'free');
  await engine.execute({ type: 'jointPose', id: free.id, position: [0, 0, 1, 0, 0, 0, 1] });
  await engine.execute({ type: 'step', count: 40 });
  assert.ok(Math.abs((await engine.frame()).bodyTransforms[bodyIndex * 7]) < 1e-12);
  await engine.execute({ type: 'restore', snapshot: initial });
  await engine.execute({ type: 'perturbTarget', target: { ...target, position: [0, 0, 1],
    rotation: [0, Math.sin(0.3), 0, Math.cos(0.3)], mode: 'rotation' } });
  await engine.execute({ type: 'step', count: 80 });
  assert.ok(Math.abs((await engine.frame()).bodyTransforms[bodyIndex * 7 + 4]) > 0.01);
  await engine.execute({ type: 'reset' });
  assert.equal((await engine.capture()).interaction, undefined);
});
