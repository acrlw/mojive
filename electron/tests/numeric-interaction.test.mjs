import assert from 'node:assert/strict';
import test, { before, after, beforeEach } from 'node:test';
import { mkdir, writeFile } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';

let server, browser, page, base;
const errors = [];
const authored = 1.23456789;
const field = () => page.getByRole('textbox', { name: 'Numeric', exact: true });
const owner = () => field().locator('..');
const state = () =>
  page.evaluate(() => ({
    value: window.controlState.value,
    events: [...window.controlEvents],
    textOwner: document.activeElement instanceof HTMLInputElement,
    compoundOwner: document.activeElement?.classList.contains('number-field') ?? false,
  }));
const paint = () =>
  owner().evaluate((el) => ({
    background: getComputedStyle(el).backgroundColor,
    shadow: getComputedStyle(el).boxShadow,
    outline: getComputedStyle(el).outlineStyle,
    cursor: getComputedStyle(el.querySelector('input')).cursor,
    editing: el.classList.contains('editing'),
  }));
async function point() {
  const box = await field().boundingBox();
  return { x: box.x + 28, y: box.y + box.height / 2 };
}
async function startDrag(distance = 30) {
  const p = await point();
  await page.mouse.move(p.x, p.y);
  await page.mouse.down();
  await page.mouse.move(p.x + distance, p.y, { steps: 3 });
  return p;
}
async function numericDraft(value) {
  await field().dblclick();
  await field().fill(String(value));
}
before(async () => {
  server = await createServer({
    server: { host: '127.0.0.1', port: 0, hmr: false },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 1100, height: 820 } });
  page.setDefaultTimeout(5000);
  page.on('pageerror', (error) => errors.push(error.message));
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
beforeEach(async () => {
  errors.length = 0;
  await page.goto(`${base}/tests/harness/ui-controls-review.html`);
  await page.waitForFunction(() => window.controlState?.value === 1.23456789);
  await field().waitFor();
  await page.evaluate(() => document.fonts.ready);
});

test('single pointer click and subthreshold movement retain scrub mode without a transaction', async () => {
  await field().click();
  assert.deepEqual(await state(), {
    value: authored,
    events: [],
    textOwner: false,
    compoundOwner: true,
  });
  assert.equal(await field().inputValue(), '1.235');
  assert.equal((await paint()).editing, false);
  await page.keyboard.type('7');
  assert.equal(
    (await state()).value,
    authored,
    'A single click must not turn typing into a value edit',
  );
  assert.equal(await field().inputValue(), '1.235');
  const p = await point();
  await page.mouse.move(p.x, p.y);
  await page.mouse.down();
  await page.mouse.move(p.x + 2, p.y);
  await page.mouse.up();
  await page.getByRole('button', { name: 'Finish', exact: true }).click();
  assert.deepEqual((await state()).events, []);
  assert.equal((await state()).value, authored);
  assert.deepEqual(errors, []);
});

test('double click exposes complete authored precision and text edit commits exactly once', async () => {
  await field().dblclick();
  assert.equal(await field().inputValue(), String(authored));
  assert.equal((await paint()).editing, true);
  assert.equal((await state()).textOwner, true);
  const selected = await field().evaluate((el) => [el.selectionStart, el.selectionEnd]);
  assert.deepEqual(selected, [0, String(authored).length]);
  await field().fill('17.123456789');
  await field().press('Enter');
  assert.equal((await state()).value, 17.123456789);
  assert.equal(await field().inputValue(), '17.123');
  assert.deepEqual((await state()).events, ['begin', 'commit']);
  assert.equal((await paint()).editing, false);
});

test('Tab enters an accessible editor and unchanged exit preserves stored precision', async () => {
  await page.evaluate(() => {
    const before = document.createElement('button');
    before.id = 'before-numeric';
    before.textContent = 'Before numeric';
    document.querySelector('main').before(before);
    before.focus();
  });
  await page.keyboard.press('Tab');
  assert.equal(await field().evaluate((el) => document.activeElement === el), true);
  assert.equal((await paint()).editing, true);
  await page.keyboard.press('Tab');
  assert.equal(
    (await state()).value,
    authored,
    'Rounded inactive text must never replace stored precision',
  );
  await field().focus();
  await field().fill('2.123456789');
  await field().press('Enter');
  assert.equal((await state()).value, 2.123456789);
  assert.equal(await field().inputValue(), '2.123');
});

test('rest, hover and held paint differ immediately before any movement in both themes', async () => {
  const evidence = [];
  for (const appearance of ['dark', 'light']) {
    await page.evaluate(async (appearance) => {
      const { applySettings, DEFAULT_SETTINGS } = await import('/src/settings.tsx');
      applySettings({ ...DEFAULT_SETTINGS, language: 'en', appearance });
      window.controlEvents = [];
    }, appearance);
    await page.getByRole('button', { name: 'Finish', exact: true }).click();
    await page.mouse.move(800, 650);
    const rest = await paint();
    await owner().screenshot({ path: `../output/electron/verification/numeric-${appearance}-rest.png` });
    const p = await point();
    await page.mouse.move(p.x, p.y);
    const hover = await paint();
    assert.notEqual(
      hover.background,
      rest.background,
      `${appearance}: hover visibly differs from rest`,
    );
    await owner().screenshot({ path: `../output/electron/verification/numeric-${appearance}-hover.png` });
    await page.mouse.down();
    const held = await paint();
    assert.notEqual(
      held.background,
      hover.background,
      `${appearance}: press is visible before drag threshold`,
    );
    assert.equal(held.editing, false);
    assert.equal(
      held.outline,
      'none',
      'Pointer focus must not paint the browser default focus ring',
    );
    assert.equal((await state()).value, authored);
    assert.deepEqual((await state()).events, [], 'Press alone starts no undo transaction');
    await owner().screenshot({ path: `../output/electron/verification/numeric-${appearance}-held.png` });
    await page.mouse.up();
    const released = await paint();
    assert.equal(released.background, hover.background);
    assert.equal(released.editing, false);
    assert.deepEqual((await state()).events, []);
    evidence.push({ appearance, rest, hover, held, released });
  }
  await writeFile('../output/electron/verification/numeric-interaction-paint.json', JSON.stringify(evidence, null, 2));
});

test('Escape during a held scrub restores one transaction and permits the next gesture', async () => {
  await startDrag(40);
  assert.ok(Math.abs((await state()).value - (authored + 0.4)) < 1e-10);
  assert.deepEqual((await state()).events, ['begin']);
  await page.keyboard.press('Escape');
  await page.mouse.up();
  assert.equal((await state()).value, authored);
  assert.deepEqual((await state()).events, ['begin', 'cancel']);
  assert.equal((await paint()).editing, false);
  await startDrag(20);
  await page.mouse.up();
  assert.ok(Math.abs((await state()).value - (authored + 0.2)) < 1e-10);
  assert.deepEqual((await state()).events, ['begin', 'cancel', 'begin', 'commit']);
});

for (const cancellation of ['pointercancel', 'lostpointercapture', 'blur'])
  test(`${cancellation} releases scrub ownership and the following gesture remains usable`, async () => {
    await field().evaluate((el) => {
      el.addEventListener(
        'pointerdown',
        (e) => {
          window.numericPointer = e.pointerId;
        },
        { once: true },
      );
    });
    const start = await startDrag(30);
    if (cancellation === 'blur') await page.evaluate(() => window.dispatchEvent(new Event('blur')));
    else if (cancellation === 'lostpointercapture')
      await field().evaluate((el) => {
        let captured = el;
        while (captured && !captured.hasPointerCapture(window.numericPointer))
          captured = captured.parentElement;
        if (!captured) throw new Error('The active numeric scrub has no pointer capture owner');
        captured.releasePointerCapture(window.numericPointer);
      });
    else
      await field().dispatchEvent('pointercancel', {
        pointerId: await page.evaluate(() => window.numericPointer),
        bubbles: true,
      });
    // Chromium delivers pending capture changes on the next pointer event.
    // Exercise capture loss while still held, before the ordinary release path.
    if (cancellation === 'lostpointercapture') await page.mouse.move(start.x + 31, start.y);
    await page.mouse.up();
    assert.equal((await state()).value, authored);
    assert.deepEqual((await state()).events, ['begin', 'cancel']);
    assert.equal((await paint()).editing, false);
    await startDrag(10);
    await page.mouse.up();
    assert.ok(Math.abs((await state()).value - (authored + 0.1)) < 1e-10);
    assert.deepEqual((await state()).events, ['begin', 'cancel', 'begin', 'commit']);
  });

test('text Escape and invalid blur cancel atomically before a fresh pointer drag', async () => {
  await numericDraft(20);
  await field().press('Escape');
  assert.equal((await state()).value, authored);
  assert.deepEqual((await state()).events, ['begin', 'cancel']);
  await numericDraft(12);
  await field().fill('invalid');
  assert.equal(await field().getAttribute('aria-invalid'), 'true');
  await page.getByRole('button', { name: 'Finish', exact: true }).click();
  assert.equal((await state()).value, authored);
  assert.deepEqual((await state()).events, ['begin', 'cancel', 'begin', 'cancel']);
  await startDrag(10);
  await page.mouse.up();
  assert.ok(Math.abs((await state()).value - (authored + 0.1)) < 1e-10);
  assert.deepEqual((await state()).events, [
    'begin',
    'cancel',
    'begin',
    'cancel',
    'begin',
    'commit',
  ]);
});

test('unit suffix shares drag ownership and disabled fields reject pointer and keyboard editing', async () => {
  await page.evaluate(async () => {
    const { default: React } = await import('/node_modules/.vite/deps/react.js');
    const { default: ReactDOM } = await import('/node_modules/.vite/deps/react-dom_client.js');
    const { Numeric } = await import('/src/components.tsx');
    const host = document.createElement('div');
    host.style.cssText = 'margin:24px;width:300px;display:grid;gap:12px';
    document.querySelector('main').append(host);
    window.unitNumeric = { value: 2.3456789, disabledChanges: 0, events: [] };
    function Review() {
      const [value, setValue] = React.useState(window.unitNumeric.value);
      return React.createElement(
        React.Fragment,
        null,
        React.createElement(Numeric, {
          label: 'With unit',
          value,
          unit: 'm',
          step: 0.01,
          onChange: (value) => {
            window.unitNumeric.value = value;
            setValue(value);
          },
          onBegin: () => window.unitNumeric.events.push('begin'),
          onCommit: () => window.unitNumeric.events.push('commit'),
        }),
        React.createElement(Numeric, {
          label: 'Disabled number',
          value: 7,
          disabled: true,
          unit: 'm',
          onChange: () => window.unitNumeric.disabledChanges++,
        }),
      );
    }
    ReactDOM.createRoot(host).render(React.createElement(Review));
  });
  const input = page.getByRole('textbox', { name: 'With unit', exact: true });
  await input.waitFor();
  const suffix = input.locator('..').locator('.unit');
  await suffix.click();
  assert.equal(await input.inputValue(), '2.346');
  assert.deepEqual(await page.evaluate(() => unitNumeric.events), []);
  const b = await suffix.boundingBox();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width / 2 + 25, b.y + b.height / 2, { steps: 3 });
  await page.mouse.up();
  assert.ok(Math.abs((await page.evaluate(() => unitNumeric.value)) - 2.5956789) < 1e-10);
  assert.deepEqual(await page.evaluate(() => unitNumeric.events), ['begin', 'commit']);
  const disabled = page.getByRole('textbox', { name: 'Disabled number', exact: true });
  assert.equal(await disabled.isDisabled(), true);
  await disabled.click({ force: true });
  await disabled.dblclick({ force: true });
  await disabled.locator('..').locator('.unit').dblclick();
  await disabled.evaluate((el) => el.focus());
  assert.equal(await disabled.evaluate((el) => document.activeElement === el), false);
  assert.equal(
    await disabled
      .locator('..')
      .evaluate((el) => el.classList.contains('editing') || el.classList.contains('pressed')),
    false,
  );
  assert.equal(await page.evaluate(() => unitNumeric.disabledChanges), 0);
  assert.equal(await disabled.inputValue(), '7.000');
});

test('fine integer dragging retains fractional motion and combines native modifiers', async () => {
  await page.evaluate(async () => {
    const { React, createRoot, Numeric } = window.controlsModules;
    const root = document.createElement('div');
    root.style.cssText = 'position:fixed;left:500px;top:40px;width:300px';
    document.body.append(root);
    function IntegerField() {
      const [value, setValue] = React.useState(0);
      window.integerValue = value;
      return React.createElement(Numeric, {
        label: 'Integer',
        value,
        integer: true,
        step: 1,
        min: -10,
        max: 10,
        onChange: setValue,
      });
    }
    createRoot(root).render(React.createElement(IntegerField));
  });
  const integer = page.getByRole('textbox', { name: 'Integer', exact: true });
  const box = await integer.boundingBox();
  const x = box.x + 30,
    y = box.y + box.height / 2;
  await page.keyboard.down('Alt');
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + 100, y, { steps: 50 });
  await page.mouse.up();
  await page.keyboard.up('Alt');
  assert.equal(await page.evaluate(() => window.integerValue), 1);
  await page.keyboard.down('Shift');
  await page.keyboard.down('Alt');
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + 20, y, { steps: 10 });
  await page.mouse.up();
  await page.keyboard.up('Alt');
  await page.keyboard.up('Shift');
  assert.equal(await page.evaluate(() => window.integerValue), 3);
});
