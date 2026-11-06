import { openEditableJointScene } from './helpers/editable-scene.mjs';
import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser;
before(async () => {
  server = await createServer({ server: { host: '127.0.0.1', port: 0 }, logLevel: 'error' });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
async function workspace(platform = 'Linux x86_64', scale = 1) {
  const page = await browser.newPage({ viewport: { width: 1540, height: 980 } });
  page.setDefaultTimeout(6000);
  await page.addInitScript(
    ({ platform, scale }) => {
      Object.defineProperty(navigator, 'platform', { value: platform });
      localStorage.setItem(
        'mojive.electron.settings.v1',
        JSON.stringify({ language: 'en', uiScale: scale }),
      );
    },
    { platform, scale },
  );
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  await openEditableJointScene(page);
  await page.evaluate(() => document.fonts.ready);
  return page;
}
const trigger = (page, name) => page.locator(`[data-menu="${name}"]`);
async function assertMenuColumns(menu) {
  const rows = await menu.locator(':scope > .workspace-menu-row').evaluateAll((elements) =>
    elements.map((element) => {
      const row = element.getBoundingClientRect();
      const icon = element.querySelector('.workspace-menu-icon .icon').getBoundingClientRect();
      const label = element.querySelector('.workspace-menu-label').getBoundingClientRect();
      const indicator = element.querySelector('.workspace-menu-indicator').getBoundingClientRect();
      return {
        name: element.textContent.trim(),
        iconX: icon.x,
        labelX: label.x,
        iconInset: icon.x - row.x,
        labelGap: label.x - icon.right,
        indicatorRight: indicator.right,
      };
    }),
  );
  assert.ok(rows.length > 1, 'compare real menu entries with different label and shortcut widths');
  for (const row of rows) {
    assert.ok(Math.abs(row.iconX - rows[0].iconX) < 0.5, `${row.name}: icon column must align`);
    assert.ok(Math.abs(row.labelX - rows[0].labelX) < 0.5, `${row.name}: label column must align`);
    assert.ok(Math.abs(row.indicatorRight - rows[0].indicatorRight) < 0.5, `${row.name}: indicator column must align`);
    assert.ok(Math.abs(row.iconInset - 10) < 0.5, `${row.name}: icon needs the authored leading inset`);
    assert.ok(Math.abs(row.labelGap - 8) < 0.5, `${row.name}: icon and label keep the authored gap`);
  }
}
test('menu activation transfers by hover and arrows, then ends on Escape, action, outside click and toggle', async () => {
  const page = await workspace();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await trigger(page, 'Edit').hover();
  assert.equal(await page.locator('.menu').count(), 0);
  await trigger(page, 'File').click();
  for (const name of ['Edit', 'Add', 'View', 'Simulate', 'Window', 'Help', 'File']) {
    await trigger(page, name).hover();
    await page.locator(`.menu-${name.toLowerCase()}:popover-open`).waitFor();
    const [anchor, popup] = await Promise.all([
      trigger(page, name).boundingBox(),
      page.locator('.menu').boundingBox(),
    ]);
    assert.ok(Math.abs(Math.max(8, anchor.x) - popup.x) < 1, `${name} must follow its own trigger`);
    assert.equal(await page.locator('.menu:popover-open').count(), 1);
  }
  await page.keyboard.press('ArrowRight');
  await page.locator('.menu-edit:popover-open').waitFor();
  await page.keyboard.press('ArrowLeft');
  await page.locator('.menu-file:popover-open').waitFor();
  await page.keyboard.press('Escape');
  assert.equal(await trigger(page, 'File').evaluate((el) => el === document.activeElement), true);
  await trigger(page, 'View').hover();
  assert.equal(await page.locator('.menu').count(), 0);
  await trigger(page, 'View').click();
  await trigger(page, 'View').click();
  assert.equal(await page.locator('.menu').count(), 0);
  await trigger(page, 'Edit').click();
  await page.mouse.click(900, 970);
  await trigger(page, 'Add').hover();
  assert.equal(await page.locator('.menu').count(), 0);
  await trigger(page, 'View').click();
  await page.getByRole('menuitem', { name: /^Frame all/ }).click();
  await trigger(page, 'Help').hover();
  assert.equal(await page.locator('.menu').count(), 0);
  assert.deepEqual(errors, []);
  await page.close();
});
test('shared submenu ownership supports pointer transfer, keyboard return and checked dispatch', async () => {
  const page = await workspace();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await trigger(page, 'View').click();
  const projection = page.getByRole('menuitem', { name: 'Projection', exact: true });
  await projection.hover();
  const popup = page.locator('.menu-submenu-projection:popover-open');
  await popup.waitFor();
  assert.equal(await page.getByRole('menuitemradio', { name: 'Perspective', exact: true }).getAttribute('aria-checked'), 'true');
  await page.getByRole('menuitemradio', { name: 'Orthographic', exact: true }).hover();
  assert.equal(await page.locator('.menu-view:popover-open').count(), 1, 'parent stays open when entering its submenu');
  await assertMenuColumns(page.locator('.menu-view:popover-open'));
  await assertMenuColumns(popup);
  await page.screenshot({ path: '../output/electron/verification/workspace-menu-projection.png' });
  await page.getByRole('menuitemradio', { name: 'Orthographic', exact: true }).click();
  assert.equal(await page.locator('.menu:popover-open').count(), 0);
  await trigger(page, 'View').click();
  await projection.focus();
  await page.keyboard.press('ArrowRight');
  await popup.waitFor();
  assert.equal(await page.getByRole('menuitemradio', { name: 'Orthographic', exact: true }).getAttribute('aria-checked'), 'true');
  await page.keyboard.press('ArrowLeft');
  assert.equal(await popup.count(), 0);
  assert.equal(await projection.evaluate((el) => el === document.activeElement), true);
  await page.keyboard.press('ArrowDown');
  assert.equal(await page.getByRole('menuitem', { name: 'Transform tool', exact: true }).evaluate((el) => el === document.activeElement), true);
  await page.keyboard.press('ArrowRight');
  await page.locator('.menu-submenu-transformtool:popover-open').waitFor();
  await page.keyboard.press('End');
  assert.equal(await page.getByRole('menuitemradio', { name: 'Joint', exact: true }).evaluate((el) => el === document.activeElement), true);
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('.menu-view:popover-open').count(), 1);
  await page.keyboard.press('Escape');
  await trigger(page, 'Window').click();
  await page.getByRole('menuitem', { name: 'Panels', exact: true }).hover();
  await page.getByRole('menuitemradio', { name: 'Cameras', exact: true }).click();
  await page.locator('.cameras-panel').waitFor();
  assert.deepEqual(errors, []);
  await page.close();
});
test('fallback file shortcuts use the same scene open and save-as commands as the menu', async () => {
  const page = await workspace();
  await page.evaluate(() => {
    window.mojive = {
      hasNativeMenu: false,
      platform: 'linux',
      browseFiles: async () => ({ path: '/workspace', parent: null, entries: [], shortcuts: [] }),
      onMenuAction: () => () => {},
      updateMenuState: async () => {},
    };
  });
  await trigger(page, 'File').click();
  assert.equal(await page.getByRole('menuitem', { name: /^Open Scene/ }).locator('kbd').innerText(), 'Ctrl+O');
  assert.equal(await page.getByRole('menuitem', { name: /^Save Scene As/ }).locator('kbd').innerText(), 'Ctrl+Shift+S');
  await page.keyboard.press('Escape');
  await page.keyboard.press('Control+o');
  const open = page.getByRole('dialog', { name: 'Open saved scene', exact: true });
  await open.waitFor();
  await open.getByRole('button', { name: 'Cancel', exact: true }).click();
  await page.keyboard.press('Control+Shift+s');
  const save = page.getByRole('dialog', { name: 'Save scene as', exact: true });
  await save.waitFor();
  await save.getByRole('button', { name: 'Cancel', exact: true }).click();
  assert.equal(await page.locator('.file-dialog').count(), 0);
  await page.close();
});
for (const [platform, command, redo] of [
  ['Linux x86_64', 'Ctrl+Z', 'Ctrl+Shift+Z'],
  ['Win32', 'Ctrl+Z', 'Ctrl+Shift+Z'],
  ['MacIntel', '⌘ Z', '⇧ ⌘ Z'],
])
  test(`platform key hints and dispatch: ${platform}`, async () => {
    const page = await workspace(platform);
    await trigger(page, 'Edit').click();
    assert.equal(await page.locator('.menu-edit kbd').nth(0).innerText(), command);
    assert.equal(await page.locator('.menu-edit kbd').nth(1).innerText(), redo);
    assert.equal(
      await page.locator('.menu-edit kbd').nth(2).innerText(),
      platform === 'MacIntel' ? '⌫' : 'Backspace',
    );
    await page.keyboard.press('Escape');
    const api = await page.evaluate(async () => {
      const { shortcut, commandKey } = await import('/src/platform.ts');
      return {
        alt: shortcut('Drag', ['alt']),
        ctrl: commandKey({ ctrlKey: true, metaKey: false }),
        meta: commandKey({ ctrlKey: false, metaKey: true }),
      };
    });
    assert.equal(api.alt, platform === 'MacIntel' ? '⌥ Drag' : 'Alt+Drag');
    assert.equal(api.ctrl, platform !== 'MacIntel');
    assert.equal(api.meta, platform === 'MacIntel');
    await trigger(page, 'Help').click();
    await page.getByRole('menuitem', { name: 'Interaction guide' }).click();
    assert.equal(
      await page.locator('.guide-grid').getByRole('img', { name: command, exact: true }).count(),
      1,
    );
    if (platform !== 'MacIntel')
      assert.doesNotMatch(await page.locator('.guide-grid').innerText(), /[⌘⌥⇧⌫]/);
    await page.close();
  });
test('tree context targets the right-clicked node, preserves selection and supports keyboard, clipboard and undo', async () => {
  const page = await workspace('Linux x86_64', 1.25);
  await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.locator('.tree-row').first().click();
  const row = page
    .locator('.tree-row')
    .filter({ has: page.locator('.tree-label', { hasText: /^01_revolute_y$/ }) });
  await row.click({ button: 'right' });
  await page.locator('.menu-object:popover-open').waitFor();
  assert.equal(await row.getAttribute('aria-selected'), 'true');
  await page.getByRole('menuitem', { name: 'Copy name', exact: true }).click();
  assert.equal(await page.evaluate(() => navigator.clipboard.readText()), '01_revolute_y');
  await row.focus();
  await page.keyboard.press('Shift+F10');
  await page.getByRole('menuitem', { name: 'Hide object', exact: true }).click();
  assert.equal(await row.locator('.tree-visibility').getAttribute('aria-pressed'), 'true');
  await page.keyboard.press('Control+z');
  assert.equal(await row.locator('.tree-visibility').getAttribute('aria-pressed'), 'false');
  await row.click({ button: 'right' });
  await page
    .locator('.menu-object')
    .screenshot({ path: '../output/electron/verification/scene-context-menu.png' });
  await page.keyboard.press('Escape');
  assert.equal(await row.evaluate((el) => el === document.activeElement), true);
  assert.equal(await row.getAttribute('aria-selected'), 'true');
  await page.close();
});
