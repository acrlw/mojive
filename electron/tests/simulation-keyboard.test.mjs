import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { createServer } from 'vite';
import { chromium } from 'playwright';

let server, browser, base;
before(async () => {
  server = await createServer({ server: { host: '127.0.0.1', port: 0, hmr: false }, logLevel: 'error' });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
});
after(async () => { await browser?.close(); await server?.close(); });

test('simulation shortcuts share transport commands and respect local input owners', async () => {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  page.setDefaultTimeout(10000);
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.addInitScript(() => localStorage.setItem('mojive.electron.settings.v1', JSON.stringify({ language: 'en' })));
  await page.goto(base);
  const transport = page.getByRole('toolbar', { name: 'Simulation', exact: true });
  await transport.waitFor();
  await page.locator('.busy-overlay').waitFor({ state: 'detached' });
  const time = async () => parseFloat(await page.locator('.status-time').innerText());
  const waitTime = (value) => page.waitForFunction((value) => parseFloat(document.querySelector('.status-time').textContent) === value, value);
  const globalFocus = () => page.evaluate(() => document.activeElement?.blur());
  const previous = transport.getByRole('button', { name: 'Previous simulation frame', exact: true });
  const step = transport.getByRole('button', { name: 'Step simulation', exact: true });
  assert.equal(await previous.isDisabled(), true);
  assert.equal(await previous.getAttribute('data-tooltip-keys'), '["←"]');
  assert.equal(await step.getAttribute('data-tooltip-keys'), '["→"]');

  // The default sample is a real MuJoCo document, so stepping changes Worker state.
  await globalFocus();
  await page.keyboard.press('ArrowLeft');
  assert.equal(await time(), 0);
  await page.keyboard.press('ArrowRight');
  await page.waitForFunction(() => parseFloat(document.querySelector('.status-time').textContent) > 0);
  const timestep = await time();
  assert.equal(await previous.isEnabled(), true);
  await page.keyboard.press('ArrowLeft');
  await waitTime(0);
  assert.equal(await previous.isDisabled(), true);
  await step.click();
  await waitTime(timestep);
  await globalFocus();
  await page.keyboard.press('ArrowLeft');
  await waitTime(0);

  await page.keyboard.down('Space');
  await transport.getByRole('button', { name: 'Pause simulation', exact: true }).waitFor();
  await page.keyboard.down('Space'); // Browser auto-repeat must not toggle back to paused.
  assert.equal(await transport.getByRole('button', { name: 'Pause simulation', exact: true }).count(), 1);
  await page.keyboard.up('Space');
  await page.waitForFunction(() => parseFloat(document.querySelector('.status-time').textContent) > 0.05);
  await page.keyboard.press('Space');
  await transport.getByRole('button', { name: 'Run simulation', exact: true }).waitFor();
  await transport.getByRole('button', { name: 'Reset simulation', exact: true }).click();
  await waitTime(0);

  const speed = page.getByRole('textbox', { name: 'Simulation speed', exact: true });
  await speed.dblclick();
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('Space');
  await page.keyboard.press('Escape');
  assert.equal(await time(), 0);
  assert.equal(await transport.getByRole('button', { name: 'Run simulation', exact: true }).count(), 1);

  const settings = page.locator('.title-actions button:has([data-icon=settings])');
  await settings.focus();
  await page.keyboard.press('Space'); // Native button activation retains Space ownership.
  await page.locator('.settings-dialog[open]').waitFor();
  await page.locator('.settings-dialog').focus();
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('Space');
  assert.equal(await time(), 0);
  await page.locator('.settings-dialog').getByRole('button', { name: 'Close', exact: true }).click();

  await page.locator('.bottom-tabs').getByRole('button', { name: /^Timeline/ }).click();
  await page.getByTestId('timeline').focus();
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('Space');
  assert.equal(await transport.getByRole('button', { name: 'Run simulation', exact: true }).count(), 1);
  assert.equal(await time(), 0);

  assert.deepEqual(errors, []);
  await page.close();
});
