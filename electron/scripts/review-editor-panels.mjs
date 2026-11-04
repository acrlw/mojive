import { createServer as createTcpServer } from 'node:net';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
import { mkdir } from 'node:fs/promises';
const root = fileURLToPath(new URL('../', import.meta.url)),
  output = new URL('../../output/electron/verification/', import.meta.url);
await mkdir(output, { recursive: true });
const socket = createTcpServer();
await new Promise((resolve) => socket.listen(0, '127.0.0.1', resolve));
const port = socket.address().port;
await new Promise((resolve) => socket.close(resolve));
const server = await createServer({
  root,
  server: { host: '127.0.0.1', port, strictPort: true },
  logLevel: 'error',
});
let browser;
try {
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage({
    viewport: { width: 1100, height: 820 },
    deviceScaleFactor: 2,
  });
  await page.goto(`http://127.0.0.1:${port}/tests/harness/ui-settings-review.html`);
  await page.waitForFunction(() => window.uiState);
  await page.evaluate(() => document.fonts.ready);
  await page.getByRole('button', { name: '设置', exact: true }).click();
  await page
    .getByRole('dialog')
    .screenshot({ path: fileURLToPath(new URL('settings-dark-zh.png', output)) });
  await page.getByRole('textbox', { name: '搜索设置' }).click();
  await page
    .getByRole('dialog')
    .screenshot({ path: fileURLToPath(new URL('settings-search-mouse-focus.png', output)) });
  await page.getByRole('combobox', { name: '语言', exact: true }).click();
  await page.getByRole('option', { name: 'English', exact: true }).click();
  await page.getByRole('button', { name: 'Light', exact: true }).click();
  await page
    .getByRole('dialog')
    .screenshot({ path: fileURLToPath(new URL('settings-light-en.png', output)) });
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Search output', exact: true }).click();
  await page.getByRole('textbox', { name: 'Search output', exact: true }).fill('missing mesh');
  await page
    .locator('.mojive-output')
    .screenshot({ path: fileURLToPath(new URL('output-filtered.png', output)) });
  await page.getByRole('button', { name: 'Settings', exact: true }).click();
  await page.getByRole('button', { name: 'Rendering', exact: true }).click();
  await page.getByRole('switch').first().click();
  for (const theme of ['dark', 'light']) {
    await page.evaluate((theme) => window.uiActions.updateSettings({ appearance: theme }), theme);
    await page
      .getByRole('dialog')
      .screenshot({ path: fileURLToPath(new URL(`settings-switches-${theme}.png`, output)) });
  }
  await page.keyboard.press('Escape');
  await page.goto(`http://127.0.0.1:${port}/tests/harness/ui-controls-review.html`);
  await page.waitForFunction(() => window.controlState);
  await page.evaluate(() => document.fonts.ready);
  await page.getByRole('button', { name: 'Material color', exact: true }).click();
  await page.screenshot({
    path: fileURLToPath(new URL('shared-color-picker.png', output)),
    clip: { x: 16, y: 16, width: 460, height: 720 },
  });
  await page.keyboard.press('Escape');
  await page.getByRole('combobox', { name: 'Projection' }).click();
  await page.screenshot({
    path: fileURLToPath(new URL('shared-select.png', output)),
    clip: { x: 16, y: 16, width: 460, height: 400 },
  });
  await page.goto(`http://127.0.0.1:${port}/tests/harness/ui-confirm-review.html`);
  await page.waitForFunction(() => window.confirmHarness);
  await page.evaluate(() => {
    window.confirmHarness.update({ appearance: 'dark', language: 'zh' });
    window.confirmHarness.show('close');
  });
  await page
    .getByRole('alertdialog')
    .screenshot({ path: fileURLToPath(new URL('confirm-scene-dark-zh.png', output)) });
  await page.evaluate(() => window.confirmHarness.update({ appearance: 'light', language: 'en' }));
  await page
    .getByRole('alertdialog')
    .screenshot({ path: fileURLToPath(new URL('confirm-scene-light-en.png', output)) });
  console.log('Captured native-token settings, output, shared controls and scene confirmation.');
} finally {
  await browser?.close();
  await server.close();
}
