import assert from 'node:assert/strict';
import { test, before, after } from 'node:test';
import { mkdir } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { installPaintStyle, settlePaint } from './helpers/surface-style.mjs';

let server, browser, page, base;
before(async () => {
  server = await createServer({
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}/tests/harness/`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 700, height: 940 }, deviceScaleFactor: 2 });
  await page.addInitScript(installPaintStyle);
  page.setDefaultTimeout(6000);
});
after(async () => {
  await browser?.close();
  await server?.close();
});

async function actuators() {
  await page.goto(`${base}actuator-readouts-review.html`);
  await page.locator('[data-actuator-name]').first().waitFor();
  await settlePaint(page);
  return page.locator('[data-actuator-name="red"]');
}
test('actuator readouts preserve exact numbers, fixed widths, and input transactions', async () => {
  const card = await actuators();
  const precise = page.locator('[data-actuator-name="precision_actuator_with_a_long_name"]');
  const numeric = precise.locator('.number-readout');
  const beforeWidth = (await numeric.boundingBox()).width;
  assert.match(await numeric.getAttribute('data-tooltip'), /^123456789\.12345\n/);
  await numeric.click({ button: 'right' });
  assert.equal(await page.evaluate(() => window.cardReview.copied), '123456789.12345');
  await precise.locator('.range-min').click({ button: 'right' });
  assert.equal(await page.evaluate(() => window.cardReview.copied), '-123456789.12345');
  assert.equal(
    await precise.locator('.range-min').evaluate((el) => getComputedStyle(el).textOverflow),
    'ellipsis',
  );
  await numeric.dblclick();
  await numeric.locator('input').fill('0.123456789');
  await page.keyboard.press('Enter');
  assert.equal(await page.evaluate(() => window.cardReview.values[1]), 0.123456789);
  assert.equal((await numeric.boundingBox()).width, beforeWidth);
  assert.equal(
    await precise.locator('.entity-card-heading').getAttribute('aria-expanded'),
    'false',
  );
  const readout = card.locator('.number-readout');
  await readout.dblclick();
  await readout.locator('input').fill('42');
  await page.keyboard.press('Enter');
  assert.equal(await page.evaluate(() => window.cardReview.values[0]), 0.05);
  const counts = await page.evaluate(() => [window.cardReview.begins, window.cardReview.commits]);
  const reset = card.getByRole('button', { name: 'Clear control input', exact: true });
  assert.equal((await reset.boundingBox()).width, 28);
  await reset.click();
  assert.deepEqual(
    await page.evaluate(() => [window.cardReview.begins, window.cardReview.commits]),
    counts.map((v) => v + 1),
  );
  assert.equal(await page.evaluate(() => window.cardReview.values[0]), 0);
  assert.equal(await card.locator('.entity-card-heading').getAttribute('aria-expanded'), 'false');
  assert.equal(
    await page.locator('[data-actuator-name="unbounded_control"] [role="slider"]').count(),
    0,
  );
});

test('readouts have distinct idle, hover and pressed states in both themes; drag remains primary', async () => {
  const card = await actuators();
  const readout = card.locator('.number-readout');
  const background = () => readout.evaluate((el) => window.paintStyle(el).backgroundColor);
  for (const theme of ['dark', 'light']) {
    await page.evaluate((value) => window.reviewTheme(value), theme);
    await page.mouse.move(600, 900);
    await settlePaint(page);
    const idle = await background();
    await readout.hover();
    await settlePaint(page);
    const hover = await background();
    assert.notEqual(hover, idle);
    await page.mouse.down();
    await settlePaint(page);
    assert.notEqual(await background(), hover);
    const box = await readout.boundingBox();
    await page.mouse.move(box.x + box.width / 2 + 20, box.y + box.height / 2, { steps: 4 });
    await page.mouse.up();
    assert.equal(await readout.evaluate((el) => el.classList.contains('editing')), false);
  }
  assert.ok((await page.evaluate(() => window.cardReview.values[0])) > 0.025);
});

test('collapsed pose channels retain axis colors, exact copy and editing without expanding', async () => {
  await page.goto(`${base}joint-panel-review.html`);
  const card = page
    .locator('.joint-entry')
    .filter({ has: page.locator('.entity-card-title', { hasText: '03_ball_xyz' }) });
  await card.waitFor();
  await settlePaint(page);
  const colors = await card
    .locator('.axis-field > .axis')
    .evaluateAll((els) => els.map((el) => window.paintStyle(el).backgroundColor));
  assert.equal(colors.length, 3);
  assert.equal(new Set(colors).size, 3);
  const input = card.locator('.number-readout input').first();
  await input.dblclick();
  await input.fill('12.3456789');
  await input.press('Enter');
  assert.equal(await card.locator('.entity-card-heading').getAttribute('aria-expanded'), 'false');
  await input.click({ button: 'right' });
  assert.ok(
    Math.abs(Number(await page.evaluate(() => window.jointReview.copied)) - 12.3456789) < 1e-9,
  );
  assert.equal(await card.locator('.field-unit').count(), 0);
});

test('joint and actuator cards fit narrow and wide panes without changing numeric column widths', async () => {
  await mkdir('../output/electron/verification', { recursive: true });
  for (const fixture of ['actuator-readouts-review', 'joint-panel-review']) {
    await page.goto(`${base}${fixture}.html`);
    const pane = page.locator(fixture.startsWith('actuator') ? '#actuator-pane' : '#joint-pane');
    await pane.waitFor();
    for (const width of [280, 440])
      for (const theme of ['dark', 'light']) {
        await pane.evaluate((el, w) => {
          el.style.width = `${w}px`;
        }, width);
        await page.evaluate((value) => window.reviewTheme(value), theme);
        await page.mouse.move(650, 930);
        await settlePaint(page);
        assert.equal(
          await pane.evaluate((el) =>
            [...el.querySelectorAll('.entity-card')].every(
              (card) => card.getBoundingClientRect().right <= el.getBoundingClientRect().right,
            ),
          ),
          true,
          `${fixture} cards fit ${width}`,
        );
        const failures = await pane
          .locator('.entity-card-header, .range-rail, .joint-pose-summary')
          .evaluateAll((elements) =>
            elements.flatMap((el) => {
              const box = el.getBoundingClientRect();
              return [...el.children]
                .filter((child) => {
                  const b = child.getBoundingClientRect();
                  return b.width && (b.right > box.right + 1 || b.left < box.left - 1);
                })
                .map(() => el.className);
            }),
          );
        assert.deepEqual(failures, [], `${fixture} ${theme} ${width}`);
        await pane.screenshot({
          path: `../output/electron/verification/${fixture}-${theme}-${width}.png`,
        });
      }
  }
});


test('joint units follow the value and reference ticks stay fixed during edits', async () => {
  await page.goto(`${base}joint-panel-review.html?reference`);
  const card = page.locator('.joint-entry').filter({
    has: page.locator('.entity-card-title', { hasText: '01_revolute_y' }),
  });
  await card.waitFor();
  assert.equal(await card.locator('.entity-card-title').textContent(), '01_revolute_y');
  assert.equal(await card.locator('.number-readout .unit').textContent(), '°');
  assert.equal(await card.locator('.range-endpoint .unit').count(), 0);
  const tick = card.locator('.joint-rail-reference');
  assert.equal(await tick.getAttribute('data-value'), String(Math.PI / 6));
  const left = await tick.evaluate((el) => el.style.left);
  assert.ok(Math.abs(parseFloat(left) - 62.5) < 1e-8);
  const input = card.locator('.number-readout input');
  await input.dblclick();
  await input.fill('70');
  await input.press('Enter');
  assert.equal(await tick.evaluate((el) => el.style.left), left);
  const slide = page.locator('.joint-entry').filter({
    has: page.locator('.entity-card-title', { hasText: '02_prismatic_x' }),
  });
  assert.equal(await slide.locator('.joint-rail-reference').count(), 0);
  assert.equal(await slide.locator('.number-readout .unit').textContent(), 'm');
  const actuator = await actuators();
  assert.equal(await actuator.locator('.joint-rail-reference').getAttribute('data-value'), '0');
  assert.equal(await actuator.locator('.joint-rail-reference').evaluate((el) => el.style.left), '50%');
});
