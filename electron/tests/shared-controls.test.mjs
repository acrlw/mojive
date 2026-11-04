import { createServer as createTcpServer } from 'node:net';
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { fileURLToPath } from 'node:url';
let server, browser, page;
before(async () => {
  const socket = createTcpServer();
  await new Promise((resolve) => socket.listen(0, '127.0.0.1', resolve));
  const port = socket.address().port;
  await new Promise((resolve) => socket.close(resolve));
  server = await createServer({
    root: fileURLToPath(new URL('../', import.meta.url)),
    server: { host: '127.0.0.1', port, strictPort: true },
    logLevel: 'error',
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 1100, height: 820 } });
  page.setDefaultTimeout(5000);
  page.on('pageerror', (error) => {
    throw error;
  });
  await page.goto(`http://127.0.0.1:${port}/tests/harness/ui-controls-review.html`);
  await page.waitForFunction(() => window.controlState);
});
after(async () => {
  await browser?.close();
  await server?.close();
});
test('numeric inactive captions keep authored precision; pointer scrubbing owns one transaction', async () => {
  const field = page.getByRole('textbox', { name: 'Numeric', exact: true });
  assert.equal(await field.inputValue(), '1.235');
  const box = await field.boundingBox();
  await page.mouse.move(box.x + 30, box.y + 14);
  await page.mouse.down();
  await page.mouse.move(box.x + 70, box.y + 14, { steps: 4 });
  await page.mouse.up();
  assert.ok(Math.abs((await page.evaluate(() => window.controlState.value)) - 1.63456789) < 1e-10);
  assert.deepEqual(await page.evaluate(() => window.controlEvents), ['begin', 'commit']);
  assert.equal(await field.inputValue(), '1.635');
});
test('double-click enters full precision; Escape restores atomically and invalid entry cannot commit', async () => {
  const field = page.getByRole('textbox', { name: 'Numeric', exact: true });
  await page.evaluate(() => (window.controlEvents = []));
  await field.dblclick();
  assert.equal(await field.inputValue(), '1.6345678900000002');
  await field.fill('17.123456');
  await field.press('Escape');
  assert.ok(Math.abs((await page.evaluate(() => window.controlState.value)) - 1.63456789) < 1e-10);
  assert.deepEqual(await page.evaluate(() => window.controlEvents), ['begin', 'cancel']);
  await page.evaluate(() => (window.controlEvents = []));
  await field.fill('12');
  await field.fill('invalid');
  await page.getByRole('button', { name: 'Finish', exact: true }).click();
  assert.ok(Math.abs((await page.evaluate(() => window.controlState.value)) - 1.63456789) < 1e-10);
  assert.deepEqual(await page.evaluate(() => window.controlEvents), ['begin', 'cancel']);
});
test('Escape during captured scrub restores value and clears pointer transaction', async () => {
  const field = page.getByRole('textbox', { name: 'Numeric', exact: true });
  await page.evaluate(() => (window.controlEvents = []));
  const box = await field.boundingBox();
  await page.mouse.move(box.x + 20, box.y + 14);
  await page.mouse.down();
  await page.mouse.move(box.x + 80, box.y + 14);
  await page.keyboard.press('Escape');
  await page.mouse.up();
  assert.ok(Math.abs((await page.evaluate(() => window.controlState.value)) - 1.63456789) < 1e-10);
  assert.deepEqual(await page.evaluate(() => window.controlEvents), ['begin', 'cancel']);
});
test('axis badge resets its component; select and switch are themed keyboard controls', async () => {
  assert.equal(
    await page.getByRole('textbox', { name: 'Position Y', exact: true }).inputValue(),
    '0.000',
  );
  await page.getByRole('button', { name: 'Reset Position Z', exact: true }).click();
  assert.equal(await page.evaluate(() => window.controlState.vector[2]), 0);
  const select = page.getByRole('combobox', { name: 'Projection' });
  await select.focus();
  await select.press('ArrowDown');
  await select.press('ArrowDown');
  await select.press('Enter');
  assert.equal(await page.evaluate(() => window.controlState.option), 'back');
  await page.getByRole('switch', { name: 'Show grid' }).press('Space');
  assert.equal(await page.evaluate(() => window.controlState.enabled), false);
  assert.equal(await page.locator('select,input[type=color],input[type=checkbox]').count(), 0);
});
test('color picker previews continuously and Escape rolls back one popup transaction', async () => {
  await page.evaluate(() => (window.controlEvents = []));
  await page.getByRole('button', { name: 'Material color', exact: true }).click();
  const sv = page.getByRole('slider', { name: 'Saturation and value' });
  await sv.click({ position: { x: 80, y: 60 } });
  assert.notEqual(await page.evaluate(() => window.controlState.color), '#9CBF8D');
  await page.keyboard.press('Escape');
  assert.equal(await page.evaluate(() => window.controlState.color), '#9CBF8D');
  assert.deepEqual(await page.evaluate(() => window.controlEvents), [
    'color begin',
    'color cancel',
  ]);
  await page.getByRole('button', { name: 'Material color', exact: true }).click();
  await page.getByRole('textbox', { name: 'Hex color' }).fill('#F00');
  await page.getByRole('textbox', { name: 'Hex color' }).press('Enter');
  await page.getByRole('button', { name: 'Done', exact: true }).click();
  assert.equal(await page.evaluate(() => window.controlState.color), '#FF0000');
  assert.deepEqual(await page.evaluate(() => window.controlEvents), [
    'color begin',
    'color cancel',
    'color begin',
    'color commit',
  ]);
});
test('native color text grammar accepts explicit normalized, byte, short hex and CSS alpha values', async () => {
  const result = await page.evaluate(() =>
    ['#abc8', 'rgb255(128,64,0)', 'rgba(128,64,0,0.5)', 'rgb1(.5,.25,0)'].map((value) =>
      window.controlsModules.parseColor(value),
    ),
  );
  assert.deepEqual(result, [
    [170 / 255, 187 / 255, 204 / 255, 136 / 255],
    [128 / 255, 64 / 255, 0, 1],
    [128 / 255, 64 / 255, 0, 0.5],
    [0.5, 0.25, 0, 1],
  ]);
});

test('late clipboard reads cannot write through a closed or reopened color transaction', async () => {
  await page.evaluate(() => {
    window.mojive = {
      readClipboardText: () => new Promise((resolve) => (window.resolveColorPaste = resolve)),
      writeClipboardText: async () => {},
    };
    window.controlEvents = [];
  });
  await page.getByRole('button', { name: 'Material color', exact: true }).click();
  await page.getByRole('button', { name: 'Paste', exact: true }).click();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  await page.getByRole('button', { name: 'Material color', exact: true }).click();
  await page.evaluate(() => window.resolveColorPaste('#00FF00'));
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(resolve)));
  assert.equal(await page.evaluate(() => window.controlState.color), '#FF0000');
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  assert.deepEqual(await page.evaluate(() => window.controlEvents), [
    'color begin',
    'color cancel',
    'color begin',
    'color cancel',
  ]);
  await page.evaluate(() => delete window.mojive);
  assert.deepEqual(
    await page.evaluate(() => window.controlsModules.parseColor('AABBCC', undefined, 0.25)),
    [170 / 255, 187 / 255, 204 / 255, 0.25],
  );
});

test('consecutive fill/Enter edits retain exactly0→2→1 and invalid drafts never enter stored state', async () => {
  const field = page.getByRole('textbox', { name: 'Numeric', exact: true });
  await page.evaluate(() => {
    window.controlActions.setValue(0);
    window.controlEvents = [];
  });
  for (const value of [0, 2, 1, 0, 2, 1]) {
    await field.fill(String(value));
    await field.press('Enter');
    assert.equal(await page.evaluate(() => window.controlState.value), value);
    assert.equal(await field.inputValue(), value.toFixed(3));
  }
  await field.fill('invalid');
  assert.equal(await page.evaluate(() => window.controlState.value), 1);
  await field.press('Enter');
  assert.equal(await page.evaluate(() => window.controlState.value), 1);
  await field.press('Escape');
  assert.equal(await field.inputValue(), '1.000');
  assert.deepEqual(await page.evaluate(() => window.controlEvents), [
    ...Array.from({ length: 6 }, () => ['begin', 'commit']).flat(),
    'begin',
    'cancel',
  ]);
});

test('scrubbing transfers focus from search text to the compound numeric owner', async () => {
  await page.evaluate(() => {
    const input = document.createElement('input');
    input.id = 'search-focus-test';
    input.setAttribute('aria-label', 'Search focus test');
    document.body.append(input);
    input.focus();
    window.controlActions.setValue(1.25);
    window.controlEvents = [];
  });
  // React batches the fixture setter; geometry queries do not await its commit.
  // Start from the caption the user can actually see before exercising ownership.
  await page.waitForFunction(
    () =>
      window.controlState.value === 1.25 &&
      document.querySelector('[aria-label="Numeric"]').value === '1.250',
  );
  assert.equal(await page.evaluate(() => document.activeElement?.id), 'search-focus-test');
  const field = page.getByRole('textbox', { name: 'Numeric', exact: true }),
    box = await field.boundingBox();
  await page.mouse.move(box.x + 20, box.y + 14);
  await page.mouse.down();
  await page.mouse.move(box.x + 50, box.y + 14);
  await page.mouse.up();
  const state = await page.evaluate(() => ({
    value: window.controlState.value,
    editing: document.activeElement instanceof HTMLInputElement,
    owner: document.activeElement?.classList.contains('number-field'),
    events: window.controlEvents,
  }));
  assert.equal(state.value, 1.55);
  assert.equal(state.editing, false);
  assert.equal(state.owner, true);
  assert.deepEqual(state.events, ['begin', 'commit']);
  await page.evaluate(() => document.getElementById('search-focus-test').remove());
});

test('ordinary UI corners keep semantic circles and derive connected control corners from their frame', async () => {
  const frames = await page.locator('button,.number-field,.axis-field').evaluateAll((nodes) =>
    nodes.map((node) => ({
      clip: getComputedStyle(node).clipPath,
      shape: getComputedStyle(node).cornerShape,
    })),
  );
  assert.ok(
    frames.every(({ clip, shape }) => clip === 'none' && /^(round|superellipse\(1\))$/.test(shape)),
  );
  const field = page.locator('.axis-field').first();
  const corners = await field.evaluate((node) => {
    const radii = (child) =>
      [
        'borderTopLeftRadius',
        'borderTopRightRadius',
        'borderBottomRightRadius',
        'borderBottomLeftRadius',
      ].map((key) => parseFloat(getComputedStyle(child)[key]));
    const sample = () => [...node.children].map(radii);
    const normal = sample();
    const previous = node.style.borderRadius;
    node.style.borderRadius = '0px';
    const square = sample();
    node.style.borderRadius = previous;
    return { normal, square };
  });
  assert.deepEqual(corners.normal, [
    [4, 0, 0, 4],
    [0, 4, 4, 0],
  ]);
  assert.deepEqual(corners.square, [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ]);
  const toggle = page.getByRole('switch', { name: 'Show grid' });
  assert.equal(
    await toggle.locator('span').evaluate((node) => getComputedStyle(node).borderRadius),
    '50%',
  );
  await page.getByRole('combobox', { name: 'Projection' }).click();
  const nested = await page.locator('.select-options').evaluate((node) => ({
    outer: parseFloat(getComputedStyle(node).borderRadius),
    inset: parseFloat(getComputedStyle(node).padding),
    inner: parseFloat(getComputedStyle(node.firstElementChild).borderRadius),
    clip: getComputedStyle(node).clipPath,
  }));
  assert.equal(nested.inner, Math.max(0, nested.outer - nested.inset));
  assert.equal(nested.clip, 'none');
  await page.keyboard.press('Escape');
});
