import test from 'node:test';
import assert from 'node:assert/strict';
import { SimulationSession } from '../src/physics/SimulationSession.ts';
import { capturePhysicsHistory, restorePhysicsHistory } from '../src/physics/gestureHistory.ts';

const joint = (value, id = 'joint-a') => ({ type: 'jointPose', id, position: [value] });
const perturb = (value, mode = 'translation') => ({ type: 'perturbTarget', target: {
  bodyId: 'body', mode, localPosition: [0, 0, 0], position: [value, 0, 0], rotation: [0, 0, 0, 1], strength: 1,
} });

function backend(id = 'model') {
  const source = { id, bodies: [], joints: [], cameras: [], lights: [], actuators: [], geometries: [] };
  let snapshot = { sourceId: id, backend: 'fixture', format: 'full-state', state: [0, 0, 0],
    environment: { timestep: 0.002 } };
  let blocked;
  const commands = [];
  return {
    source, commands,
    async frame() { return { sourceId: id, time: snapshot.state[0], bodyTransforms: new Float64Array(),
      jointFrames: new Float64Array(), cameraTransforms: new Float64Array(), lightFrames: new Float64Array(),
      jointPositions: new Float64Array(), controls: new Float64Array() }; },
    async capture() { return structuredClone(snapshot); },
    async execute(command) {
      commands.push(structuredClone(command));
      const wait = blocked;
      blocked = undefined;
      if (wait) { wait.started(); await wait.released; }
      if (command.type === 'jointPose') snapshot.state[command.id === 'joint-a' ? 1 : 2] = command.position[0];
      if (command.type === 'perturbTarget') snapshot.interaction = structuredClone(command.target);
      if (command.type === 'clearPerturbation') delete snapshot.interaction;
      if (command.type === 'restore') snapshot = structuredClone(command.snapshot);
      if (command.type === 'reset') { snapshot.state = [0, 0, 0]; delete snapshot.interaction; }
      if (command.type === 'step') snapshot.state[0] += command.count * 0.002;
      return this.frame();
    },
    blockNext() {
      let started, release;
      const waiting = new Promise((resolve) => { started = resolve; });
      const released = new Promise((resolve) => { release = resolve; });
      blocked = { started, released };
      return { waiting, release };
    },
    dispose() {},
  };
}

async function fixture(t) {
  const session = new SimulationSession(), engine = backend();
  await session.attach(engine);
  t.after(() => session.dispose());
  return { session, engine };
}

test('a delayed worker consumes only the active and latest drag targets; commit and undo retain the final pose', async (t) => {
  const { session, engine } = await fixture(t);
  const before = capturePhysicsHistory(session);
  const gate = engine.blockNext();
  const first = session.updateTarget(joint(1));
  await gate.waiting;
  const waiting = session.updateTarget(joint(2));
  for (let value = 3; value <= 1000; value++)
    assert.equal(session.updateTarget(joint(value)), waiting, 'Obsolete pointer samples share one pending result.');
  const after = capturePhysicsHistory(session);
  gate.release();
  await Promise.all([first, waiting, after.captured]);
  assert.deepEqual(engine.commands.map((command) => command.position[0]), [1, 1000]);
  assert.equal((await after.captured).state[1], 1000, 'Pointer release captures the final target, never an older acknowledgement.');
  await restorePhysicsHistory(before, session);
  assert.equal(session.current.snapshot.state[1], 0);
  await restorePhysicsHistory(after, session);
  assert.equal(session.current.snapshot.state[1], 1000);
});

test('capture, reset, and step are barriers that later drag targets cannot cross', async (t) => {
  const { session, engine } = await fixture(t);
  const one = session.updateTarget(joint(1));
  const captured = session.capture();
  const two = session.updateTarget(joint(2));
  const reset = session.execute({ type: 'reset' });
  const three = session.updateTarget(joint(3));
  const step = session.execute({ type: 'step', count: 1 });
  const four = session.updateTarget(joint(4));
  await Promise.all([one, two, reset, three, step, four]);
  assert.equal((await captured).state[1], 1);
  assert.deepEqual(engine.commands.map((command) => command.type === 'jointPose' ? command.position[0] : command.type),
    [1, 2, 'reset', 3, 'step', 4]);
  assert.deepEqual(session.current.snapshot.state, [0.002, 4, 0]);
});

test('different joint and perturbation identities remain ordered and clearing perturbation seals its last target', async (t) => {
  const { session, engine } = await fixture(t);
  const operations = [session.updateTarget(joint(1)), session.updateTarget(joint(2, 'joint-b')),
    session.updateTarget(joint(3)), session.updateTarget(perturb(1)), session.updateTarget(perturb(2, 'rotation')),
    session.execute({ type: 'clearPerturbation' }), session.updateTarget(perturb(3))];
  await Promise.all(operations);
  assert.equal(engine.commands.length, 7);
  assert.deepEqual(session.current.snapshot.state, [0, 3, 2]);
  assert.deepEqual(session.current.snapshot.interaction, perturb(3).target);
  await session.execute({ type: 'clearPerturbation' });
  assert.equal(session.current.snapshot.interaction, undefined);
});

test('joint posing pauses simulation at the write boundary while physical perturbation keeps playback running', async (t) => {
  const { session } = await fixture(t);
  session.play();
  const pose = session.updateTarget(joint(1));
  assert.equal(session.current.playing, false);
  await pose;
  session.play();
  await session.updateTarget(perturb(1));
  assert.equal(session.current.playing, true);
  session.pause();
  session.play();
  await session.execute(joint(2));
  assert.equal(session.current.playing, false, 'Discrete pose writes obey the same pause rule.');
});

test('a joint pose queued behind an active playback step prevents the tick from rescheduling', { timeout: 2000 }, async (t) => {
  const { session, engine } = await fixture(t);
  const gate = engine.blockNext();
  t.after(gate.release);
  session.play();
  await gate.waiting;
  assert.equal(engine.commands[0].type, 'step', 'The real playback scheduler must reach the backend.');
  const pose = session.updateTarget(joint(7));
  assert.equal(session.current.playing, false);
  gate.release();
  await pose;
  const committed = await session.capture();
  assert.equal(committed.state[1], 7);
  await new Promise((resolve) => setTimeout(resolve, 80));
  assert.deepEqual(engine.commands.map((command) => command.type), ['step', 'jointPose'],
    'Finishing an older tick must not restart playback after the pose paused it.');
  assert.deepEqual((await session.capture()).state, committed.state);
  assert.equal(session.current.playing, false);
});

test('a replaced simulation cannot inherit a queued target or receive an old frame', async (t) => {
  const { session, engine } = await fixture(t);
  const gate = engine.blockNext(), first = session.updateTarget(joint(1));
  await gate.waiting;
  const obsolete = session.updateTarget(joint(2));
  const rejected = assert.rejects(obsolete, /no longer active/);
  session.detach();
  const replacement = backend('replacement');
  await session.attach(replacement);
  const current = session.updateTarget(joint(3));
  assert.notEqual(current, obsolete);
  gate.release();
  await Promise.all([first, rejected, current]);
  assert.deepEqual(engine.commands.map((command) => command.position[0]), [1]);
  assert.deepEqual(replacement.commands.map((command) => command.position[0]), [3]);
  assert.equal(session.current.frame.sourceId, 'replacement');
  assert.equal(session.current.snapshot.state[1], 3);
});

test('a rejected coalesced target reports failure without poisoning subsequent gesture commands', async (t) => {
  const { session, engine } = await fixture(t);
  const execute = engine.execute.bind(engine);
  engine.execute = async (command) => {
    if (command.type === 'jointPose' && command.position[0] === 2) throw new Error('Pose rejected');
    return execute(command);
  };
  const first = session.updateTarget(joint(1)), latest = session.updateTarget(joint(2));
  assert.equal(first, latest);
  await assert.rejects(latest, /Pose rejected/);
  assert.equal(session.current.snapshot.state[1], 0);
  assert.match(session.current.error, /Pose rejected/);
  await session.updateTarget(joint(3));
  assert.equal(session.current.snapshot.state[1], 3);
  assert.equal(session.current.error, null);
});
