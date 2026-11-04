import assert from 'node:assert/strict';
import { test, before, after } from 'node:test';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
let server, browser, page;
before(async () => {
  server = await createServer({
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    logLevel: 'error',
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 700, height: 940 }, deviceScaleFactor: 2 });
  page.setDefaultTimeout(6000);
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/tests/harness/joint-panel-review.html`);
  await page.waitForFunction(() => window.jointReview);
  await page.locator('.joint-entry').first().waitFor();
});
after(async () => {
  await browser?.close();
  await server?.close();
});
test('joint panel uses limit headings, localized types, collapsible 3/6 DoF pose and native toolbar', async () => {
  assert.equal(await page.locator('.joint-entry').count(), 9);
  assert.equal(await page.getByRole('slider').count(), 7);
  const geometry = await page
    .getByRole('slider')
    .first()
    .evaluate((element) => {
      const thumb = element.querySelector('.joint-rail-thumb'),
        track = element.querySelector('.joint-rail-track'),
        card = element.closest('.joint-entry');
      return {
        thumb: (thumb.getBoundingClientRect().width / devicePixelRatio) * devicePixelRatio,
        track: track.getBoundingClientRect().height,
        limits: element.querySelectorAll('.joint-rail-limit').length,
        cardRadius: getComputedStyle(card).borderRadius,
        inset: getComputedStyle(card).padding,
        fieldRadius: getComputedStyle(card.querySelector('.number-field')).borderRadius,
      };
    });
  assert.equal(geometry.thumb, 12);
  assert.equal(geometry.track, 4);
  assert.equal(geometry.limits, 0);
  assert.equal(geometry.cardRadius, '12px');
  assert.equal(geometry.inset, '8px');
  assert.equal(geometry.fieldRadius, '4px');
  assert.match(await page.locator('.joint-entry').first().innerText(), /−120…120°/);
  assert.equal(await page.locator('.joint-pose-summary').count(), 2);
  await page.getByTitle('Search joints', { exact: true }).click();
  await page.getByRole('textbox', { name: 'Search joints', exact: true }).fill('Ball joint');
  assert.equal(await page.locator('.joint-entry').count(), 1);
  assert.match(await page.locator('.joint-entry').innerText(), /3 DoF/);
  await page.getByTitle('Search joints', { exact: true }).click();
  await page.getByTitle('Copy state', { exact: true }).click();
  const copied = await page.evaluate(() => JSON.parse(window.jointReview.copied));
  assert.equal(copied.length, 18);
  assert.ok(copied.every(Number.isFinite));
});
test('scalar pointer release outside, Escape at a limit, and subsequent keyboard edit each own one transaction', async () => {
  const slider = page.getByRole('slider', { name: '06_precision_hinge_y slider', exact: true });
  await slider.scrollIntoViewIfNeeded();
  const box = await slider.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + 14);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width + 100, box.y + 14);
  const held = await page.evaluate(() => ({
    value: window.jointReview.doc.joints.find((j) => j.name === '06_precision_hinge_y').value,
    begins: window.jointReview.begins,
    commits: window.jointReview.commits,
  }));
  assert.ok(Math.abs(held.value - Math.PI / 90) < 1e-10);
  assert.equal(held.begins, 1);
  assert.equal(held.commits, 0);
  await page.keyboard.press('Escape');
  await page.mouse.up();
  assert.equal(
    await page.evaluate(
      () => window.jointReview.doc.joints.find((j) => j.name === '06_precision_hinge_y').value,
    ),
    0,
  );
  assert.equal(await page.evaluate(() => window.jointReview.commits), 1);
  await slider.focus();
  await slider.press('End');
  assert.equal(await page.evaluate(() => window.jointReview.commits), 2);
  await page.mouse.move(box.x + box.width / 2, box.y + 14);
  await page.mouse.down();
  await page.mouse.move(box.x - 40, box.y + 14);
  await page.mouse.up();
  assert.equal(await page.evaluate(() => window.jointReview.commits), 3);
  assert.ok(Number(await slider.getAttribute('aria-valuenow')) < 0);
});
test('expanded ball/free fields edit real configuration and remain readable in light and fractional narrow panes', async () => {
  const ball = page.locator('.joint-entry').filter({ hasText: '03_ball_xyz' }),
    free = page.locator('.joint-entry').filter({ hasText: '04_free_6dof' });
  await ball.getByTitle('Expand joint pose', { exact: true }).click();
  await ball.getByRole('textbox', { name: 'Rotation X', exact: true }).fill('20');
  await ball.getByRole('textbox', { name: 'Rotation X', exact: true }).press('Enter');
  const x = await page.evaluate(
    () => window.jointReview.doc.joints.find((j) => j.name === '03_ball_xyz').rotation.x,
  );
  assert.ok(Math.abs(x - Math.sin(Math.PI / 18)) < 1e-6);
  for (const pitch of [89, 91]) {
    await ball.getByRole('textbox', { name: 'Rotation Y', exact: true }).fill(String(pitch));
    await ball.getByRole('textbox', { name: 'Rotation Y', exact: true }).press('Enter');
  }
  assert.equal(
    Number(await ball.getByRole('textbox', { name: 'Rotation X', exact: true }).inputValue()),
    20,
  );
  assert.equal(
    Number(await ball.getByRole('textbox', { name: 'Rotation Y', exact: true }).inputValue()),
    91,
  );
  assert.equal(
    Number(await ball.getByRole('textbox', { name: 'Rotation Z', exact: true }).inputValue()),
    0,
  );
  await ball.getByRole('textbox', { name: 'Rotation Y', exact: true }).fill('0');
  await ball.getByRole('textbox', { name: 'Rotation Y', exact: true }).press('Enter');
  await free.getByTitle('Expand joint pose', { exact: true }).click();
  assert.equal(
    Number(await free.getByRole('textbox', { name: 'Position X', exact: true }).inputValue()),
    1.65,
  );
  assert.equal(
    Number(await free.getByRole('textbox', { name: 'Position Z', exact: true }).inputValue()),
    0.55,
  );
  await free.getByRole('textbox', { name: 'Position Z', exact: true }).fill('0.25');
  await free.getByRole('textbox', { name: 'Position Z', exact: true }).press('Enter');
  assert.equal(
    await page.evaluate(
      () =>
        window.jointReview.doc.joints
          .find((j) => j.name === '04_free_6dof')
          .object.getWorldPosition(new window.jointReview.doc.root.position.constructor()).z,
    ),
    0.25,
  );
  await mkdir('../output/electron/verification', { recursive: true });
  await page.locator('.joint-panel-list').evaluate((el) => (el.scrollTop = 0));
  await page
    .locator('#joint-pane')
    .screenshot({ path: '../output/electron/verification/joint-panel-native-dark.png' });
  await page.evaluate(() => {
    window.reviewTheme('light');
    document.getElementById('joint-pane').style.width = '320px';
    document.getElementById('joint-pane').style.zoom = '1.25';
  });
  const bounds = await page
    .locator('.joint-scalar-fields')
    .first()
    .evaluate((el) => ({ width: el.clientWidth, scroll: el.scrollWidth }));
  assert.ok(bounds.scroll <= bounds.width);
  await page
    .locator('#joint-pane')
    .screenshot({ path: '../output/electron/verification/joint-panel-native-light-125.png' });
});

test('free world pose edits invert the authored parent frame and survive save/restore', async () => {
  const result = await page.evaluate(async () => {
    const { jointDisplayPose, setJointDisplayPose } = await import('/src/joint-pose.ts');
    const { saveWorkspace, loadWorkspace } = await import('/src/editor.ts');
    const THREE = await import('/node_modules/three/build/three.module.js');
    const doc = window.jointReview.doc,
      joint = doc.joints.find((j) => j.name === '04_free_6dof');
    doc.root.position.set(1, 2, 3);
    doc.root.rotation.set(0, 0, 0.4);
    doc.root.scale.setScalar(1.6);
    const target = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.4, -0.3, 0.8, 'ZYX'));
    setJointDisplayPose(doc, joint, { position: [2, -1, 4], rotation: target.toArray() });
    const pose = jointDisplayPose(joint),
      restored = loadWorkspace(saveWorkspace(doc, [], 10)).doc;
    const after = jointDisplayPose(restored.joints.find((j) => j.name === '04_free_6dof'));
    return {
      position: pose.position.toArray(),
      angle: pose.rotation.angleTo(target),
      restored: after.position.toArray(),
      restoredAngle: after.rotation.angleTo(target),
      local: joint.position.toArray(),
    };
  });
  result.position.forEach((value, index) => assert.ok(Math.abs(value - [2, -1, 4][index]) < 1e-10));
  result.restored.forEach((value, index) => assert.ok(Math.abs(value - [2, -1, 4][index]) < 1e-10));
  assert.ok(result.angle < 1e-7);
  assert.ok(result.restoredAngle < 1e-7);
  assert.notDeepEqual(result.local, result.position);
});
