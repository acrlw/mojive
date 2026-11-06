import { openEditableJointScene } from './helpers/editable-scene.mjs';
import { installPaintStyle } from './helpers/surface-style.mjs';
import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { mkdir } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser;
const out = '../output/electron/verification';
before(async () => {
  await mkdir(out, { recursive: true });
  server = await createServer({
    server: { host: '127.0.0.1', port: 0, hmr: false },
    logLevel: 'error',
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
test('shared mouse masters keep solid mirrored halves without seams or divider overflow', async () => {
  const page = await browser.newPage({
    viewport: { width: 1800, height: 1200 },
    deviceScaleFactor: 2,
  });
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
  const base = `http://127.0.0.1:${server.httpServer.address().port}`;
  for (const theme of ['dark', 'light']) {
    await page.goto(`${base}/tests/harness/input-hints-review.html?theme=${theme}&scale=1.25`);
    await page.locator('.mouse-icon-gallery').waitFor();
    await page.evaluate(() => document.fonts.ready);
    await page
      .locator('.mouse-icon-gallery')
      .screenshot({ path: `${out}/mouse-icons-${theme}.png` });
  }
  const seams = await page.evaluate(async () => {
    const failures = [];
    for (const button of ['left', 'right']) {
      const svg = document
        .querySelector(`[data-icon=status-mouse-${button}]`)
        .innerHTML.replaceAll('currentColor', '#000');
      const image = new Image();
      image.src = 'data:image/svg+xml,' + encodeURIComponent(svg);
      await image.decode();
      for (const size of [14, 18, 22, 24, 28, 56, 112])
        for (const scale of [1, 1.25, 2]) {
          const pixels = size * scale;
          const canvas = document.createElement('canvas');
          canvas.width = canvas.height = Math.ceil(pixels + 2);
          const ctx = canvas.getContext('2d');
          for (const offset of [0, 0.25, 0.5]) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(image, offset, offset, pixels, pixels);
            const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
            for (let y = 0; y < canvas.height; y++)
              for (let x = 0; x < canvas.width; x++) {
                let u = ((x + 0.5 - offset) * 24) / pixels;
                const v = ((y + 0.5 - offset) * 24) / pixels;
                if (button === 'right') u = 24 - u;
                // Probe the continuous solid state fill away from the exterior antialias edge.
                if (u >= 5.6 && u <= 11.4 && v >= 9 && v <= 10.4) {
                  const alpha = data[(y * canvas.width + x) * 4 + 3];
                  if (alpha < 248) failures.push({ button, size, scale, offset, u, v, alpha });
                }
                if (pixels >= 112 && u >= 12.2 && u <= 13 && v >= 4 && v <= 9) {
                  const alpha = data[(y * canvas.width + x) * 4 + 3];
                  if (alpha > 10)
                    failures.push({ overflow: true, button, size, scale, offset, u, v, alpha });
                }
              }
          }
        }
    }
    return failures;
  });
  assert.deepEqual(seams, []);
  await page.close();
});
for (const [appearance, language, uiScale, uiFontSize = 14] of [
  ['dark', 'en', 1],
  ['light', 'zh', 1.25],
  ['dark', 'zh', 2],
  ['light', 'en', 1.25, 20],
]) {
  test(`production mouse/key guide and delayed tooltips: ${appearance}/${language}/${uiScale}/${uiFontSize}`, async () => {
    const page = await browser.newPage({
      viewport: { width: 1540, height: 980 },
      deviceScaleFactor: 2,
    });
    await page.addInitScript(installPaintStyle);
    await page.evaluate(installPaintStyle);
    page.setDefaultTimeout(6000);
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.addInitScript(
      (settings) => localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings)),
      { appearance, language, uiScale, uiFontSize },
    );
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
    await page.locator('.tree-row').first().waitFor();
    await page.evaluate(() => document.fonts.ready);
    const rotation = page.locator('.tool-palette button:has([data-icon=rotate])');
    await rotation.hover();
    assert.equal(await page.getByRole('tooltip').count(), 0, 'Hints must not flash immediately');
    const tip = page.getByRole('tooltip');
    await tip.waitFor();
    assert.equal(await tip.locator('kbd').innerText(), 'R');
    assert.equal(
      await tip.locator(':scope > span').first().innerText(),
      language === 'zh' ? '旋转' : 'Rotation',
    );
    assert.equal(await rotation.getAttribute('aria-describedby'), await tip.getAttribute('id'));
    const geometry = await tip.evaluate((el) => {
      const a = el.getBoundingClientRect(),
        key = el.querySelector('kbd'),
        b = key.getBoundingClientRect(),
        css = window.paintStyle(el),
        kcss = window.paintStyle(key);
      const zoom = a.width / el.offsetWidth;
      return {
        outer: parseFloat(css.borderRadius),
        inner: parseFloat(kcss.borderRadius),
        inset: [b.top - a.top, a.bottom - b.bottom, a.right - b.right].map((v) => v / zoom),
        font: css.fontSize,
        keyFont: kcss.fontSize,
        inViewport: a.left >= 0 && a.right <= innerWidth && a.top >= 0 && a.bottom <= innerHeight,
      };
    });
    assert.equal(geometry.outer, 10);
    assert.equal(geometry.inner, 4);
    assert.ok(
      geometry.inset.every((v) => Math.abs(v - 6) < 0.15),
      JSON.stringify(geometry),
    );
    assert.equal(geometry.font, `${uiFontSize}px`);
    assert.equal(geometry.keyFont, '14px');
    assert.ok(geometry.inViewport);
    const palette = await page.locator('.tool-palette').boundingBox(),
      box = await tip.boundingBox();
    const x = Math.max(0, Math.min(palette.x, box.x) - 12),
      y = Math.max(0, Math.min(palette.y, box.y) - 12);
    await page.screenshot({
      path: `${out}/tooltip-${appearance}-${uiScale}${uiFontSize === 14 ? '' : '-large-text'}.png`,
      clip: {
        x,
        y,
        width: Math.min(1540 - x, Math.max(palette.x + palette.width, box.x + box.width) - x + 12),
        height: Math.min(
          980 - y,
          Math.max(palette.y + palette.height, box.y + box.height) - y + 12,
        ),
      },
    });
    await rotation.click();
    await tip.waitFor({ state: 'detached' });
    assert.equal(await rotation.getAttribute('aria-describedby'), null);
    await page.mouse.move(1500, 950);
    await rotation.hover();
    await tip.waitFor();
    await page.keyboard.press('Escape');
    await tip.waitFor({ state: 'detached' });
    await page.locator('[data-menu=Help]').click();
    await page
      .locator('.menu-help')
      .getByRole('menuitem', {
        name: language === 'zh' ? '交互指南' : 'Interaction guide',
        exact: true,
      })
      .click();
    const guide = page.locator('.interaction-guide');
    assert.equal(await guide.locator('.guide-row').count(), 17);
    assert.equal(await guide.locator('[data-icon^=status-mouse-]').count(), 7);
    assert.equal(await guide.locator('.guide-tool').count(), 0);
    for (const [english, chinese, key] of [
      ['Playback', '播放控制', 'Space'],
      ['Previous simulation frame', '上一帧仿真', '←'],
      ['Step simulation', '单步仿真', '→'],
    ]) {
      const row = guide.locator('.guide-row').filter({
        has: page.getByText(language === 'zh' ? chinese : english, { exact: true }),
      });
      assert.equal(await row.count(), 1, `${english} must have its own guide row`);
      assert.deepEqual(await row.locator('kbd').allTextContents(), [key]);
    }
    const pan = guide
      .locator('.guide-row')
      .filter({ hasText: language === 'zh' ? '平移视角' : 'Pan' });
    assert.equal(await pan.locator('.input-mouse').count(), 1);
    const mac = await page.evaluate(() => /Mac/i.test(navigator.platform));
    if (mac) assert.match(await guide.innerText(), /⌘/);
    else assert.doesNotMatch(await guide.innerText(), /[⌘⌥]/);
    assert.equal(
      await guide.getByRole('img', { name: mac ? '⌘ Z' : 'Ctrl+Z', exact: true }).count(),
      1,
    );
    const overflow = await guide.evaluate((el) =>
      [...el.querySelectorAll('.guide-inputs,.guide-gesture,.input-chord')]
        .filter((node) => node.scrollWidth > node.clientWidth + 1)
        .map((node) => node.textContent),
    );
    assert.deepEqual(overflow, []);
    await page.locator('.info-dialog').screenshot({
      path: `${out}/input-guide-${appearance}-${uiScale}${uiFontSize === 14 ? '' : '-large-text'}.png`,
    });
    const close = page.locator('.info-dialog .dialog-heading button');
    await close.hover();
    await page.waitForTimeout(550);
    assert.equal(await tip.count(), 0, 'Obvious Close action does not repeat its name');
    await close.click();
    await tip.waitFor({ state: 'detached' });
    assert.equal(
      await page.locator('[title]').count(),
      0,
      'Native browser title bubbles must not remain',
    );
    assert.deepEqual(errors, []);
    await page.close();
  });
}

test('entity name hints appear only while the actual label is clipped', async () => {
  const page = await browser.newPage({ viewport: { width: 1700, height: 1100 } });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  await openEditableJointScene(page);
  const row = page.locator('.tree-row[data-tooltip="01_revolute_y"]');
  await row.waitFor();
  const label = row.locator('.tree-label');
  await row.hover();
  await page.waitForTimeout(550);
  assert.equal(await page.getByRole('tooltip').count(), 0);
  await page.mouse.move(900, 800);
  await label.evaluate((node) => {
    node.style.maxWidth = '40px';
    node.style.flex = '0 0 40px';
  });
  await row.hover();
  await page.getByRole('tooltip').waitFor();
  assert.equal(await page.getByRole('tooltip').innerText(), '01_revolute_y');
  await label.evaluate((node) => {
    node.style.maxWidth = '';
    node.style.flex = '';
  });
  await page.getByRole('tooltip').waitFor({ state: 'detached' });
  await page.close();
});
