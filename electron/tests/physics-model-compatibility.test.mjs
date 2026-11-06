import test from 'node:test';
import assert from 'node:assert/strict';
import loadMujoco from '@mujoco/mujoco';
import { createModelBundle } from '../src/resources/modelBundle.ts';
import { MujocoBackend } from '../src/physics/backends/mujoco/MujocoBackend.ts';
import { explainModelCompatibility, MujocoModelCompatibilityError } from '../src/physics/backends/mujoco/modelCompatibility.ts';

const mj = await loadMujoco();
const bundle = (xml) => createModelBundle('Compatibility', 'model.xml', [
  { path: 'model.xml', bytes: new TextEncoder().encode(xml) },
]);
const flex = (integrator, contact = '') => `<mujoco><option integrator="${integrator}" solver="CG"/>
  <worldbody><flexcomp name="soft" type="grid" count="3 3 3" spacing=".2 .2 .2" dim="3" mass="5">
    <contact selfcollide="none" ${contact}/><edge damping="1"/><elasticity young="1000" poisson="0.3"/>
  </flexcomp></worldbody></mujoco>`;

test('real WASM reports model version migrations without changing physics or an existing session', async (t) => {
  const live = await MujocoBackend.load(mj, await bundle('<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom size=".1"/></body></worldbody></mujoco>'));
  t.after(() => live.dispose());
  const original = await live.capture();
  for (const [xml, code, evidence] of [
    [flex('discrete', 'internal="false"'), 'removed-flex-internal', /unrecognized attribute: 'internal'/],
    [flex('implicitfast'), 'flex-integrator-migration', /flex elasticity is no longer integrated implicitly/],
  ]) {
    const source = await bundle(xml);
    const before = structuredClone(source);
    await assert.rejects(MujocoBackend.load(mj, source), (error) => {
      assert.ok(error instanceof MujocoModelCompatibilityError);
      assert.equal(error.code, code);
      assert.equal(error.entry, 'model.xml');
      assert.equal(error.runtimeVersion, mj.mj_versionString());
      assert.match(String(error.cause), evidence);
      assert.ok(error.message.includes(mj.mj_versionString()));
      return true;
    });
    assert.deepEqual(source, before);
    assert.deepEqual(await live.capture(), original);
  }
  const advanced = await live.execute({ type: 'step', count: 2 });
  assert.ok(advanced.time > 0);
});

test('a model explicitly authored with the discrete flex integrator loads and simulates', async (t) => {
  const engine = await MujocoBackend.load(mj, await bundle(flex('discrete')));
  t.after(() => engine.dispose());
  assert.equal((await engine.environment()).integrator, mj.mjtIntegrator.mjINT_DISCRETE.value);
  assert.ok(engine.source.environment.find((field) => field.id === 'integrator').choices
    .some((choice) => choice.value === mj.mjtIntegrator.mjINT_DISCRETE.value));
  const frame = await engine.execute({ type: 'step', count: 4 });
  assert.ok(frame.time > 0);
  assert.ok(frame.bodyTransforms.every(Number.isFinite));
});

test('unrelated compiler failures keep their original identity and diagnostic', () => {
  const missing = new Error('XML Error: could not open file: missing.xml');
  assert.equal(explainModelCompatibility(missing, 'model.xml', mj.mj_versionString()), missing);
  const unrelatedInternal = new Error("XML Error: unrecognized attribute: 'internal'\nElement 'body', line 1");
  assert.equal(explainModelCompatibility(unrelatedInternal, 'model.xml', mj.mj_versionString()), unrelatedInternal);
});
