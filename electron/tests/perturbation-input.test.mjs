import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';

test('Ctrl drags perturb the selection through production pointer routing and MuJoCo', { timeout: 60000 }, async (t) => {
  const server = await createServer({ configFile: false, server: { host: '127.0.0.1', port: 0, watch: null }, logLevel: 'error' });
  await server.listen(); t.after(() => server.close());
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage({ viewport: { width: 1100, height: 760 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.stack));
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/tests/harness/perturbation-review.html`);
  await page.waitForFunction(() => window.perturbationReview?.api?.current);
  const canvas = page.getByTestId('viewport-canvas');
  await canvas.focus();
  await page.waitForTimeout(450);
  const capture = () => page.evaluate(async () => {
    const f = window.perturbationReview;
    const state = await f.runtime.session.capture();
    return { target: state.interaction?.target ?? null, begins: f.begins, commits: f.commits, cancels: f.cancels,
      selected: f.selected, bodyId: f.body.id, camera: f.api.current.getCameraView(), errors: f.errors };
  });
  const initial = await capture();
  // This point is empty sky, deliberately not a hit on the selected body.
  await page.mouse.move(360, 120);
  await page.keyboard.down('Control');
  await page.mouse.down();
  await page.mouse.move(430, 160, { steps: 5 });
  let held = await capture();
  assert.equal(held.target?.bodyId, initial.bodyId);
  assert.equal(held.target.mode, 'translation');
  assert.equal(held.begins, 1);
  assert.equal(held.selected, initial.selected);
  assert.deepEqual(held.camera, initial.camera, 'Ctrl must not orbit the view');
  assert.ok(Math.hypot(...held.target.position.map((v, i) => v - [0, 0, .7][i])) > .05);
  const displacement = await page.evaluate(async () => {
    const f = window.perturbationReview;
    const before = f.body.object.position.toArray();
    await f.runtime.session.execute({ type: 'step', count: 80 }); f.refresh();
    return f.body.object.position.distanceTo({ x: before[0], y: before[1], z: before[2] });
  });
  assert.ok(displacement > .001, `MuJoCo must move the selected body under force: ${displacement}`);
  await mkdir('../output/electron/verification', { recursive: true });
  await canvas.screenshot({ path: '../output/electron/verification/perturbation-translation.png' });
  await page.mouse.up(); await page.keyboard.up('Control');
  assert.equal((await capture()).target, null);
  assert.equal((await capture()).commits, 1);

  await page.evaluate(async () => {
    const f = window.perturbationReview; await f.runtime.session.execute({ type: 'reset' }); f.refresh();
  });
  await page.keyboard.down('Control');
  await page.mouse.down({ button: 'right' });
  await page.mouse.move(490, 200, { steps: 5 });
  held = await capture();
  assert.equal(held.target.mode, 'rotation');
  assert.deepEqual(held.target.localPosition, [0, 0, 0], 'Rotation pivots at the selected body origin');
  assert.ok(Math.abs(held.target.rotation[3]) < .999);
  const rotation = await page.evaluate(async () => {
    const f = window.perturbationReview; await f.runtime.session.execute({ type: 'step', count: 80 }); f.refresh();
    return f.body.object.quaternion.toArray();
  });
  assert.ok(Math.abs(rotation[3]) < .99999, 'MuJoCo must rotate the selected body under torque');
  await canvas.screenshot({ path: '../output/electron/verification/perturbation-rotation.png' });
  // Releasing Ctrl ends the interaction even while the mouse remains down.
  await page.keyboard.up('Control');
  assert.equal((await capture()).target, null);
  await page.mouse.up({ button: 'right' });
  assert.equal((await capture()).commits, 2);

  for (const cancel of ['Escape', 'blur', 'capture']) {
    await canvas.focus(); await page.keyboard.down('Control'); await page.mouse.down();
    await page.mouse.move(510, 220);
    assert.ok((await capture()).target);
    if (cancel === 'Escape') await page.keyboard.press('Escape');
    else if (cancel === 'blur') await canvas.evaluate(c => c.blur());
    else await canvas.dispatchEvent('lostpointercapture', { pointerId: 1 });
    assert.equal((await capture()).target, null, `${cancel} must clear held force`);
    await page.mouse.up(); await page.keyboard.up('Control');
  }
  assert.equal((await capture()).cancels, 3);
  for (const select of ['fixed', null]) {
    await page.evaluate(key => { const f = window.perturbationReview; f.select(key ? f[key].id : null); }, select);
    await canvas.focus(); await page.keyboard.down('Control'); await page.mouse.down();
    await page.mouse.move(530, 240); await page.mouse.up(); await page.keyboard.up('Control');
    assert.equal((await capture()).target, null);
    assert.equal((await capture()).begins, 5, 'Fixed and empty selections must not claim a physical transaction');
  }
  await page.evaluate(async () => {
    const f = window.perturbationReview;
    await f.runtime.session.execute({ type: 'reset' });
    f.select(f.body.id); f.refresh();
  });
  for (const hit of ['body', 'other']) {
    const point = await page.evaluate(key => {
      const f = window.perturbationReview; return f.project(f[key]);
    }, hit);
    await page.mouse.move(...point);
    await page.keyboard.down('Control'); await page.mouse.down();
    held = await capture();
    assert.equal(held.target?.bodyId, initial.bodyId, 'Hitting another body must not transfer the selection');
    if (hit === 'body') {
      const local = held.target.localPosition;
      const half = [.2, .15, .2];
      assert.ok(local.every((v, i) => Math.abs(v) <= half[i] + 1e-5));
      assert.ok(local.some((v, i) => Math.abs(Math.abs(v) - half[i]) < 1e-5),
        'Translation grabs the actual surface, not the body center plane');
    }
    await page.mouse.move(point[0] + 25, point[1] - 15, { steps: 3 });
    await page.mouse.up(); await page.keyboard.up('Control');
    assert.equal((await capture()).target, null);
  }
  assert.deepEqual((await capture()).errors, []);
  assert.deepEqual(errors, []);
});
