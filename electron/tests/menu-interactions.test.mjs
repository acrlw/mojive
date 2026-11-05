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
  await page.locator('.tree-row').first().waitFor();
  await page.evaluate(() => document.fonts.ready);
  return page;
}
const trigger = (page, name) => page.locator(`[data-menu="${name}"]`);
test('menu activation transfers by hover and arrows, then ends on Escape, action, outside click and toggle', async () => {
  const page = await workspace();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await trigger(page, 'Edit').hover();
  assert.equal(await page.locator('.menu').count(), 0);
  await trigger(page, 'File').click();
  for (const name of ['Edit', 'Add', 'View', 'Help', 'File']) {
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
  await page.getByRole('menuitem', { name: 'Frame all', exact: true }).click();
  await trigger(page, 'Help').hover();
  assert.equal(await page.locator('.menu').count(), 0);
  assert.deepEqual(errors, []);
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
      platform === 'MacIntel' ? '⌫' : 'Delete',
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
