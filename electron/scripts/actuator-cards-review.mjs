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
    args: [root, '--hidden', '--user-data-dir=' + profile],
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
  const color = (locator) => locator.evaluate((node) => getComputedStyle(node).backgroundColor);
  const geometry = (locator) =>
    locator.evaluate((node) => {
      const style = getComputedStyle(node),
        heading = node.querySelector('.entity-card-heading'),
        hs = getComputedStyle(heading);
      const icon = heading.querySelector('.icon'),
        list = node.closest('.joint-panel-list');
      return {
        radius: style.borderRadius,
        inset: style.padding,
        shadow: style.boxShadow,
        border: style.borderBottomWidth,
        background: style.backgroundColor,
        headingHeight: hs.height,
        headingRadius: hs.borderRadius,
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
    await action('scene.atelier_arm');
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
    const hover = await color(heading());
    await page.mouse.down();
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
    await card().locator('.actuator-summary > button').click();
    assert.equal(await heading().getAttribute('aria-expanded'), 'true');
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
