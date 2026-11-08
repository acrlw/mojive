import { chooseSampleScene } from '../tests/helpers/editable-scene.mjs';
import assert from 'node:assert/strict';
import { _electron } from 'playwright';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const out = path.join(root, '../output/electron/verification');
await mkdir(out, { recursive: true });
const profile = await mkdtemp(path.join(out, 'actuator-review-profile-'));
const modelName = 'card-controls-review.xml';
const modelPath = path.join(out, modelName);
const source = await readFile(path.join(root, '../assets/joint_gizmo.xml'), 'utf8');
await writeFile(
  modelPath,
  source.replace(
    '</mujoco>',
    `<actuator>
  <motor name="hinge_control" joint="01_revolute_y" ctrllimited="true" ctrlrange="-0.05 0.05"/>
  <motor name="slide_control_with_a_long_name" joint="02_prismatic_x" ctrllimited="true" ctrlrange="-1000000.123456789 1000000.123456789"/>
</actuator></mujoco>`,
  ),
);
const env = { ...process.env };
delete env.MOJIVE_DEV_URL;
const report = {
  application: 'Production Electron App',
  indexSha256: createHash('sha256')
    .update(await readFile(path.join(root, 'dist/index.html')))
    .digest('hex'),
  states: [],
  errors: [],
};
let app, page;
try {
  app = await _electron.launch({
    args: [root, '--user-data-dir=' + profile],
    cwd: root,
    env,
    timeout: 45000,
  });
  page = await app.firstWindow();
  page.setDefaultTimeout(15000);
  page.on('pageerror', (error) => report.errors.push(error.stack));
  await app.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0].setContentSize(1540, 980),
  );
  await page.locator('.tree-row').first().waitFor();
  assert.equal(new URL(page.url()).protocol, 'mojive-app:');
  const action = async (id) =>
    app.evaluate(({ Menu, BrowserWindow }, id) => {
      const item = Menu.getApplicationMenu().getMenuItemById(id);
      assertAvailable(item, id);
      item.click(item, BrowserWindow.getAllWindows()[0], {});
      function assertAvailable(item, id) {
        if (!item || !item.enabled) throw Error(`Unavailable menu action: ${id}`);
      }
    }, id);
  const rail = (icon) => page.locator(`.activity-rail > button:has([data-icon=${icon}])`);
  const card = () => page.locator('.actuator-entry').first();
  const heading = () => card().locator('.entity-card-heading');
  const color = (locator) =>
    locator.evaluate((node) =>
      node.dataset.smoothPaint
        ? JSON.parse(node.dataset.smoothPaint).backgroundColor
        : getComputedStyle(node).backgroundColor,
    );
  const geometry = (locator) =>
    locator.evaluate((node) => {
      const style = getComputedStyle(node),
        paint = node.dataset.smoothPaint ? JSON.parse(node.dataset.smoothPaint) : style,
        heading = node.querySelector('.entity-card-heading'),
        hs = getComputedStyle(heading),
        headingPaint = heading.dataset.smoothPaint ? JSON.parse(heading.dataset.smoothPaint) : hs;
      const icon = heading.querySelector('.icon'),
        list = node.closest('.joint-panel-list');
      return {
        radius: paint.borderRadius,
        inset: style.padding,
        shadow: paint.boxShadow,
        border: style.borderBottomWidth,
        background: paint.backgroundColor,
        headingHeight: hs.height,
        headingRadius: headingPaint.borderRadius,
        headingInset: hs.padding,
        headingWeight: hs.fontWeight,
        icon: [getComputedStyle(icon).width, getComputedStyle(icon).height],
        gap: getComputedStyle(node.querySelector('.entity-card-content')).marginTop,
        overflow: list.scrollWidth > list.clientWidth,
      };
    });
  for (const [theme, language, width, scale] of [
    ['light', 'zh', 320, 1],
    ['dark', 'en', 440, 1],
    ['light', 'en', 280, 1.25],
    ['dark', 'zh', 440, 1.25],
  ]) {
    const name = `actuator-cards-${theme}-${language}-${width}-${scale}x`;
    await page.evaluate(
      ({ theme, language, width, scale }) => {
        localStorage.setItem(
          'mojive.electron.settings.v1',
          JSON.stringify({ appearance: theme, language, uiScale: scale }),
        );
        localStorage.setItem(
          'mojive.layout',
          JSON.stringify({ left: width, right: 340, bottom: 260 }),
        );
      },
      { theme, language, width, scale },
    );
    await page.reload();
    await page.locator('.tree-row').first().waitFor();
    await page.evaluate(() => document.fonts.ready);
    // Exercise the shared cards through the production MuJoCo session and command path.
    await action('file.openSimulation');
    const pathInput = page.locator('.file-dialog form input');
    await pathInput.fill(out);
    await pathInput.press('Enter');
    await page.getByRole('option', { name: modelName, exact: true }).dblclick();
    await page.locator('.file-dialog').waitFor({ state: 'detached' });
    await page.locator('.scene-loading-overlay').waitFor({ state: 'detached' });
    await rail('control').click();
    const liveCard = page.locator('[data-actuator-name="hinge_control"]');
    await liveCard.waitFor();
    const liveValue = liveCard.locator('.number-readout input');
    await liveValue.dblclick();
    await liveValue.fill('0.025');
    await liveValue.press('Enter');
    await page.waitForFunction(
      () => document.querySelector('[data-actuator-name="hinge_control"] input')?.value === '0.025',
    );
    await liveValue.click({ button: 'right' });
    assert.equal(await app.evaluate(({ clipboard }) => clipboard.readText()), '0.025');
    assert.deepEqual(
      await liveCard
        .locator('.range-endpoint')
        .evaluateAll((nodes) => nodes.map((node) => node.dataset.value)),
      ['-0.05', '0.05'],
    );
    const liveGeometry = await geometry(liveCard);
    assert.equal(liveGeometry.overflow, false);
    await page.mouse.move(20, 20);
    await page.screenshot({ path: path.join(out, name + '-live-controls.png') });
    await liveCard.locator('.range-endpoint').first().hover();
    assert.match(await page.locator('.statusbar').innerText(), /Copy value|复制数值/);
    await page.screenshot({ path: path.join(out, name + '-numeric-status.png') });
    await liveCard.locator('button:has([data-icon=reset])').click();
    await page.waitForFunction(
      () => document.querySelector('[data-actuator-name="hinge_control"] input')?.value === '0.000',
    );
    await rail('pose').click();
    await page.locator('.joint-entry').first().waitFor();
    await page.screenshot({ path: path.join(out, name + '-live-joints.png') });
    await chooseSampleScene(page, 'Atelier arm', language === 'zh');
    await page
      .getByRole('alertdialog')
      .getByRole('button', { name: language === 'zh' ? '不保存' : "Don't Save", exact: true })
      .click();
    await page.waitForFunction(() =>
      document.querySelector('.document-title')?.textContent.includes('Atelier'),
    );
    await rail('control').click();
    await card().waitFor();
    assert.equal(await page.locator('.actuator-entry').count(), 3);
    assert.equal(await page.locator('.actuators-panel [role=slider]').count(), 0);
    const ratios = await page.locator('.actuator-reduction > span:last-child').allTextContents();
    assert.deepEqual(ratios, ['120', '100', '60']);
    const normal = await color(heading());
    await heading().hover();
    // Rounded controls paint through a surface layer in production.
    await page.waitForFunction((before) => {
      const node = document.querySelector('.actuator-entry .entity-card-heading');
      if (!node) return false;
      const paint = node.dataset.smoothPaint
        ? JSON.parse(node.dataset.smoothPaint)
        : getComputedStyle(node);
      return paint.backgroundColor !== before;
    }, normal);
    const hover = await color(heading());
    await page.mouse.down();
    await page.waitForFunction((before) => {
      const node = document.querySelector('.actuator-entry .entity-card-heading');
      const paint = node.dataset.smoothPaint
        ? JSON.parse(node.dataset.smoothPaint)
        : getComputedStyle(node);
      return paint.backgroundColor !== before;
    }, hover);
    const held = await color(heading());
    assert.notEqual(normal, hover);
    assert.notEqual(hover, held);
    await page.mouse.up();
    assert.equal(await heading().getAttribute('aria-expanded'), 'true');
    assert.ok((await page.locator('.inspector-heading').innerText()).includes('shoulder'));
    const details = await card().locator('dd').allTextContents();
    assert.equal(details.length, 2);
    assert.ok(details.some((value) => value.includes('SimpleTransmission')));
    assert.ok(details.some((value) => value.includes('shoulder')));
    await heading().click();
    assert.equal(await heading().getAttribute('aria-expanded'), 'false');
    await heading().focus();
    await page.keyboard.press('Space');
    assert.equal(await heading().getAttribute('aria-expanded'), 'true');
    await page.keyboard.press('Enter');
    assert.equal(await heading().getAttribute('aria-expanded'), 'false');
    await heading().dblclick();
    assert.equal(await heading().getAttribute('aria-expanded'), 'true');
    await page.waitForTimeout(450);
    assert.equal(await card().locator('.actuator-joint').innerText(), 'shoulder');
    assert.equal(await card().locator('.actuator-summary button').count(), 0);
    const actuatorGeometry = await geometry(card());
    assert.equal(actuatorGeometry.shadow, 'none');
    assert.equal(actuatorGeometry.border, '0px');
    assert.equal(actuatorGeometry.overflow, false);
    assert.equal(actuatorGeometry.radius, '12px');
    assert.equal(actuatorGeometry.inset, '8px');
    assert.equal(actuatorGeometry.headingRadius, '4px');
    await page.mouse.move(20, 20);
    await page.waitForTimeout(200);
    await page.screenshot({ path: path.join(out, name + '.png') });
    await rail('pose').click();
    const selectedJoint = page.locator('.joint-entry.selected');
    await selectedJoint.waitFor();
    assert.ok((await selectedJoint.locator('.entity-card-title').innerText()).includes('shoulder'));
    const jointGeometry = await geometry(selectedJoint);
    assert.deepEqual(
      jointGeometry,
      actuatorGeometry,
      'Joint and actuator shell geometry/paint must match',
    );
    await selectedJoint.scrollIntoViewIfNeeded();
    await page.mouse.move(20, 20);
    await page.screenshot({ path: path.join(out, name + '-joints.png') });
    report.states.push({
      name,
      theme,
      language,
      width,
      scale,
      normal,
      hover,
      held,
      ratios,
      details,
      actuatorGeometry,
      jointGeometry,
      liveGeometry,
    });
    console.log(`PASS ${name}`);
  }
  assert.deepEqual(report.errors, []);
  await writeFile(path.join(out, 'actuator-cards-review.json'), JSON.stringify(report, null, 2));
} finally {
  if (app) {
    await app
      .evaluate(({ BrowserWindow }) =>
        BrowserWindow.getAllWindows().forEach((window) => window.destroy()),
      )
      .catch(() => {});
    await app.close();
  }
  await rm(profile, { recursive: true, force: true });
}
