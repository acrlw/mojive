import { installPaintStyle } from './helpers/surface-style.mjs';
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';

let server, browser;
before(async () => {
  server = await createServer({
    cacheDir: 'node_modules/.vite-scrollbars',
    server: { host: '127.0.0.1', port: 0, strictPort: false, watch: null },
    logLevel: 'error',
  });
  await server.listen();
  // Playwright otherwise disables the very native paint/capture path this test exercises.
  browser = await chromium.launch({
    channel: 'chrome',
    headless: true,
    ignoreDefaultArgs: ['--hide-scrollbars'],
  });
  await mkdir('../output/electron/verification/corner-coverage', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
async function app(theme = 'dark', scale = 1) {
  const page = await browser.newPage({
    viewport: { width: Math.round(1540 * scale), height: Math.round(980 * scale) },
  });
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
  page.setDefaultTimeout(6000);
  await page.addInitScript(
    (settings) => localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings)),
    { language: 'en', appearance: theme, uiScale: scale },
  );
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  await page.locator('.tree-row').first().waitFor();
  return page;
}
async function geometry(owner) {
  return owner.evaluate((element) => {
    const r = element.getBoundingClientRect(),
      sx = r.width / element.offsetWidth,
      sy = r.height / element.offsetHeight;
    return {
      x: r.x + element.clientLeft * sx,
      y: r.y + element.clientTop * sy,
      width: element.clientWidth * sx,
      height: element.clientHeight * sy,
      right: r.right,
      bottom: r.bottom,
      sx,
      sy,
      scrollTop: element.scrollTop,
      scrollLeft: element.scrollLeft,
      scrollHeight: element.scrollHeight,
      scrollWidth: element.scrollWidth,
      clientHeight: element.clientHeight,
      clientWidth: element.clientWidth,
    };
  });
}
async function state(owner) {
  return owner.evaluate((element) => ({
    visible: element.hasAttribute('data-scrollbar-visible'),
    hover: element.dataset.scrollbarHover ?? null,
    pressed: element.dataset.scrollbarPressed ?? null,
    top: element.scrollTop,
    left: element.scrollLeft,
  }));
}
async function paint(page, owner, axis, role, path) {
  const scale = await owner.evaluate(
    (element) => element.getBoundingClientRect().width / element.offsetWidth,
  );
  const screenshot = await owner.screenshot(path ? { path } : {});
  return page.evaluate(
    async ({ image, axis, role, scale }) => {
      const img = new Image();
      img.src = `data:image/png;base64,${image}`;
      await img.decode();
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const expected = window
        .paintStyle(document.documentElement)
        .getPropertyValue('--ui-' + role)
        .trim();
      ctx.fillStyle = expected;
      ctx.fillRect(0, 0, 1, 1);
      const rgb = [...ctx.getImageData(0, 0, 1, 1).data].slice(0, 3);
      const x = axis === 'y' ? Math.round(img.width - 5 * scale) : 0,
        y = axis === 'x' ? Math.round(img.height - 5 * scale) : 0,
        w = axis === 'y' ? Math.max(1, Math.floor(3 * scale)) : img.width,
        h = axis === 'x' ? Math.max(1, Math.floor(3 * scale)) : img.height;
      const pixels = ctx.getImageData(x, y, w, h).data;
      let matches = 0;
      for (let i = 0; i < pixels.length; i += 4)
        if (rgb.every((v, j) => Math.abs(v - pixels[i + j]) < 2)) matches++;
      return { matches, expected };
    },
    { image: screenshot.toString('base64'), axis, role, scale },
  );
}
async function fixture(page, scale, modal = false) {
  await page.evaluate(
    ({ scale, modal }) => {
      const shell = document.createElement(modal ? 'dialog' : 'div');
      shell.id = 'scrollbar-fixture';
      Object.assign(shell.style, {
        position: 'fixed',
        left: '70px',
        top: '60px',
        padding: '0',
        margin: '0',
        border: '0',
        zIndex: '999999',
        zoom: String(scale),
        background: 'var(--panel-color)',
      });
      shell.innerHTML =
        '<div id="native-scroll-owner" tabindex="0" style="width:240px;height:180px;overflow:auto"><div style="width:900px;height:1200px;padding:12px;box-sizing:border-box">Selectable content<br><input aria-label="Fixture text" value="Keep native selection"></div></div>';
      document.body.append(shell);
      if (modal) shell.showModal();
    },
    { scale, modal },
  );
  return page.locator('#native-scroll-owner');
}

for (const [theme, scale] of [
  ['dark', 1],
  ['light', 1.25],
  ['dark', 2],
])
  test(`native scrollbar track hover, held capture, idle and both axes ${theme}/${scale}x`, async () => {
    const page = await app(theme, scale),
      owner = await fixture(page, scale);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const initial = await geometry(owner);
    assert.equal((await state(owner)).visible, false);
    const gutterX = initial.right - 3.5 * scale,
      thumbY = initial.y + 12 * scale;
    await page.mouse.move(gutterX, thumbY);
    assert.deepEqual(
      { ...(await state(owner)), top: 0, left: 0 },
      { visible: true, hover: 'y', pressed: null, top: 0, left: 0 },
    );
    assert.ok(
      (
        await paint(
          page,
          owner,
          'y',
          'primary-dim',
          `../output/electron/verification/scrollbar-hover-${theme}-${scale}x.png`,
        )
      ).matches > 0,
    );
    await page.waitForTimeout(1100);
    assert.equal((await state(owner)).visible, false, 'stationary hover does not renew activity');
    assert.equal(
      (
        await paint(
          page,
          owner,
          'y',
          'primary-dim',
          `../output/electron/verification/scrollbar-idle-${theme}-${scale}x.png`,
        )
      ).matches,
      0,
    );
    await page.mouse.move(gutterX, thumbY + scale);
    assert.equal((await state(owner)).visible, true);
    await page.mouse.down();
    assert.equal((await state(owner)).pressed, 'y');
    assert.ok(
      (
        await paint(
          page,
          owner,
          'y',
          'primary',
          `../output/electron/verification/scrollbar-held-${theme}-${scale}x.png`,
        )
      ).matches > 0,
    );
    await page.waitForTimeout(1100);
    assert.equal((await state(owner)).visible, true, 'held thumb stays visible without motion');
    await page.mouse.move(gutterX, thumbY + 70 * scale, { steps: 5 });
    await page.mouse.up();
    assert.ok((await state(owner)).top > 100, 'browser native thumb drag changes scrollTop');
    assert.equal((await state(owner)).pressed, null);
    await page.waitForTimeout(1100);
    assert.equal((await state(owner)).visible, false);
    await page.mouse.move(initial.x + 30 * scale, initial.y + 30 * scale);
    await page.mouse.wheel(0, 100);
    await page.waitForTimeout(40);
    assert.equal((await state(owner)).visible, true);
    assert.equal((await state(owner)).hover, null);
    assert.ok((await paint(page, owner, 'y', 'field-active')).matches > 0);
    // The complete empty track, not only the grab, owns hover and native page-seeking.
    await page.mouse.move(gutterX, initial.y + 160 * scale);
    assert.equal((await state(owner)).hover, 'y');
    await page.mouse.down();
    await page.mouse.up();
    const horizontalY = initial.bottom - 3.5 * scale;
    await page.mouse.move(initial.x + 10 * scale, horizontalY);
    assert.equal((await state(owner)).hover, 'x');
    assert.ok((await paint(page, owner, 'x', 'primary-dim')).matches > 0);
    await page.mouse.down();
    await page.mouse.move(initial.x + 100 * scale, horizontalY, { steps: 5 });
    await page.mouse.up();
    assert.ok((await state(owner)).left > 50);
    const final = await geometry(owner);
    assert.equal(final.clientWidth, initial.clientWidth);
    assert.equal(final.clientHeight, initial.clientHeight);
    assert.deepEqual(errors, []);
    await page.close();
  });

test('native scrollbar end caps keep their geometry when Chromium changes thumb length', async () => {
  for (const scale of [1, 1.25]) {
    const page = await app('dark', scale),
      owner = await fixture(page, scale);
    await owner.evaluate((element) => {
      element.style.background = '#000';
    });
    for (const smoothing of [0.382, 1]) {
      const cap = await page.evaluate(async (smoothing) => {
        const { applyScrollbarPaint } = await import('/src/scrollbar-paint.ts');
        const root = document.documentElement;
        root.style.setProperty('--ui-field-active', '#fff');
        applyScrollbarPaint(root, smoothing);
        return parseFloat(root.style.getPropertyValue('--scrollbar-cap'));
      }, smoothing);
      let previous;
      for (const size of [400, 1200, 100000]) {
        await owner.evaluate((element, size) => {
          Object.assign(element.firstElementChild.style, {
            width: size + 'px',
            height: size + 'px',
          });
          element.dataset.scrollbarVisible = '';
        }, size);
        const screenshot = await owner.screenshot({
          path: `../output/electron/verification/corner-coverage/thumb-${scale}-${smoothing}-${size}.png`,
        });
        const pixels = await page.evaluate(
          async ({ data, scale, cap }) => {
            const img = new Image();
            img.src = 'data:image/png;base64,' + data;
            await img.decode();
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            const band = Math.ceil(7 * scale),
              length = Math.ceil((2 + cap + 1) * scale);
            const y = ctx.getImageData(img.width - band, 0, band, length).data;
            const x = ctx.getImageData(0, img.height - band, length, band).data;
            const center = ctx.getImageData(
              Math.floor(img.width - 3.5 * scale),
              0,
              1,
              img.height - band,
            ).data;
            let visible = 0;
            for (let i = 0; i < center.length; i += 4) if (center[i] > 0) visible++;
            return { x: [...x], y: [...y], visible };
          },
          { data: screenshot.toString('base64'), scale, cap },
        );
        assert.ok(
          pixels.visible / scale >= 2 * cap,
          'native minimum thumb accommodates both unchanged end caps',
        );
        if (previous) {
          assert.deepEqual(pixels.x, previous.x, 'horizontal end cap is not stretched');
          assert.deepEqual(pixels.y, previous.y, 'vertical end cap is not stretched');
        }
        previous = pixels;
      }
    }
    await page.close();
  }
});

test('late modal owners retain text/keyboard input, and cancel/blur/removal release scrollbar state', async () => {
  const page = await app(),
    owner = await fixture(page, 1, true);
  await owner.getByRole('textbox', { name: 'Fixture text' }).fill('Native text input');
  assert.equal(await owner.getByRole('textbox').inputValue(), 'Native text input');
  await owner.focus();
  await page.keyboard.press('PageDown');
  await page.waitForTimeout(200);
  assert.ok((await state(owner)).top > 0);
  assert.equal((await state(owner)).visible, true);
  for (const reason of ['pointercancel', 'blur', 'hide', 'remove']) {
    await owner.evaluate((element) => (element.scrollTop = 0));
    const g = await geometry(owner);
    await page.mouse.move(g.right - 3.5, g.y + 12);
    await page.mouse.down();
    assert.equal((await state(owner)).pressed, 'y');
    await page.evaluate((reason) => {
      const owner = document.querySelector('#native-scroll-owner');
      window.removedOwner = owner;
      if (reason === 'remove') document.querySelector('#scrollbar-fixture').remove();
      else if (reason === 'hide') owner.style.visibility = 'hidden';
      else if (reason === 'blur') window.dispatchEvent(new Event('blur'));
      else owner.dispatchEvent(new PointerEvent('pointercancel', { bubbles: true, pointerId: 1 }));
    }, reason);
    await page.mouse.up();
    assert.equal(
      await page.evaluate(() => window.removedOwner.hasAttribute('data-scrollbar-pressed')),
      false,
    );
    if (reason === 'hide') {
      assert.equal((await state(owner)).visible, false);
      await owner.evaluate((element) => (element.style.visibility = 'visible'));
    }
  }
  await page.close();
});

test('a second pointer cannot steal capture and a same-coordinate no-button move releases a missing pointerup', async () => {
  const page = await app(),
    owner = await fixture(page, 1),
    g = await geometry(owner);
  const result = await owner.evaluate(
    (element, { x, y }) => {
      const send = (type, pointerId, buttons) =>
        element.dispatchEvent(
          new PointerEvent(type, {
            bubbles: true,
            pointerId,
            clientX: x,
            clientY: y,
            button: 0,
            buttons,
          }),
        );
      send('pointermove', 11, 0);
      send('pointerdown', 11, 1);
      send('pointerdown', 12, 1);
      send('pointerup', 12, 0);
      send('pointercancel', 12, 0);
      const retained = element.dataset.scrollbarPressed === 'y';
      send('pointermove', 11, 0);
      return {
        retained,
        released: !element.hasAttribute('data-scrollbar-pressed'),
        visible: element.hasAttribute('data-scrollbar-visible'),
      };
    },
    { x: g.right - 3.5, y: g.y + 12 },
  );
  assert.deepEqual(result, { retained: true, released: true, visible: true });
  await page.waitForTimeout(1100);
  assert.equal((await state(owner)).visible, false);
  await page.close();
});

test('production tree, Inspector, joints, Settings and timeline use the shared owner without waking neighboring panels', async () => {
  const page = await app('light', 1.25);
  await page.setViewportSize({ width: 1925, height: 900 });
  while (await page.locator('.tree-row[aria-expanded=false] > button.tree-chevron').count())
    await page.locator('.tree-row[aria-expanded=false] > button.tree-chevron').first().click();
  await page.locator('.tree-row.type-camera').first().click();
  const inspector = page.locator('.inspector-content');
  await inspector.evaluate((element) => (element.scrollTop = 1));
  await page.waitForTimeout(30);
  assert.equal((await state(inspector)).visible, true);
  const tree = page.locator('.left-content');
  assert.equal(await tree.count(), 1);
  await page.waitForTimeout(1100);
  const g = await geometry(tree);
  await page.mouse.move(g.right - 3.5 * 1.25, g.y + g.height / 2);
  assert.equal(
    (await state(tree)).visible,
    true,
    JSON.stringify({
      g,
      state: await state(tree),
      hit: await page.evaluate(
        ({ x, y }) => document.elementFromPoint(x, y)?.outerHTML.slice(0, 300),
        { x: g.right - 3.5 * 1.25, y: g.y + g.height / 2 },
      ),
    }),
  );
  assert.equal((await state(inspector)).visible, false);
  await tree.evaluate((element) => (element.scrollTop = 0));
  await page.mouse.move(g.right - 3.5 * 1.25, g.y + 12 * 1.25);
  await page.mouse.down();
  await page.mouse.move(g.right - 3.5 * 1.25, g.y + 70 * 1.25, { steps: 5 });
  await page.mouse.up();
  assert.ok(
    (await state(tree)).top > 100,
    'tree thumb drag reaches the native scrollbar, not the pane splitter',
  );
  assert.equal((await geometry(tree)).right, g.right);
  await page.getByRole('button', { name: 'Joints', exact: true }).click();
  const joints = page.locator('.joint-panel-list');
  await joints.evaluate((element) => (element.scrollTop = 20));
  await page.waitForTimeout(30);
  assert.equal((await state(joints)).visible, true);
  await page
    .locator('.activity-rail')
    .getByRole('button', { name: 'Settings', exact: true })
    .click();
  await page.getByRole('textbox', { name: 'Search all settings', exact: true }).fill('a');
  const settings = page.locator('.settings-page');
  await settings.evaluate((element) => (element.scrollTop = 20));
  await page.waitForTimeout(30);
  assert.equal((await state(settings)).visible, true);
  await page
    .locator('.settings-header')
    .getByRole('button', { name: 'Close', exact: true })
    .click();
  const timeline = page.locator('.timeline-body');
  const divider = page.getByRole('separator', { name: 'Resize bottom panel', exact: true });
  const dividerRect = await divider.boundingBox();
  await page.mouse.move(dividerRect.x + 100, dividerRect.y + dividerRect.height / 2);
  await page.mouse.down();
  await page.mouse.move(dividerRect.x + 100, dividerRect.y + 125, { steps: 8 });
  await page.mouse.up();
  await timeline.evaluate((element) => (element.scrollLeft = 120));
  await page.waitForTimeout(30);
  assert.equal((await state(timeline)).visible, true);
  const t = await geometry(timeline);
  assert.ok(t.scrollWidth > t.clientWidth && t.scrollHeight > t.clientHeight);
  await timeline.evaluate((element) => (element.scrollTop = 30));
  await page.waitForTimeout(30);
  assert.ok((await state(timeline)).top > 0);
  await timeline.screenshot({
    path: '../output/electron/verification/scrollbar-timeline-two-axes.png',
  });
  for (let i = 0; i < 7; i++) {
    await page.locator('.titlebar nav').getByRole('button', { name: 'Add', exact: true }).click();
    await page.getByRole('menuitem', { name: 'Box', exact: true }).click();
  }
  await page.locator('.activity-rail').getByRole('button', { name: 'Output', exact: true }).click();
  const output = page.locator('.output-messages');
  await output.hover();
  await page.mouse.wheel(0, -150);
  await page.waitForTimeout(50);
  assert.equal((await state(output)).visible, true);
  assert.ok((await geometry(output)).scrollHeight > (await geometry(output)).clientHeight);
  await page.close();
});

for (const scale of [1, 1.25, 2])
  test(`splitter hit strips preserve boundaries without covering native scroll gutters at ${scale}x`, async () => {
    const page = await app('dark', scale);
    const bounds = await page.evaluate(() => {
      const rect = (selector) => {
        const r = document.querySelector(selector).getBoundingClientRect();
        return {
          left: r.left,
          right: r.right,
          top: r.top,
          bottom: r.bottom,
          width: r.width,
          height: r.height,
        };
      };
      return {
        left: rect('.left-panel'),
        right: rect('.inspector'),
        bottom: rect('.bottom-panel'),
        handles: [...document.querySelectorAll('.panel-splitter')].map((e) => {
          const r = e.getBoundingClientRect(),
            s = window.paintStyle(e, '::after');
          return {
            left: r.left,
            right: r.right,
            top: r.top,
            bottom: r.bottom,
            width: r.width,
            height: r.height,
            afterLeft: parseFloat(s.left),
            afterTop: parseFloat(s.top),
          };
        }),
      };
    });
    const [left, right, bottom] = bounds.handles;
    assert.ok(Math.abs(left.left - bounds.left.right) < 1);
    assert.ok(Math.abs(right.right - bounds.right.left) < 1);
    assert.ok(Math.abs(bottom.bottom - bounds.bottom.top) < 1);
    assert.ok(
      Math.abs(left.width - 7 * scale) < 1 &&
        Math.abs(right.width - 7 * scale) < 1 &&
        Math.abs(bottom.height - 7 * scale) < 1,
    );
    assert.ok(Math.abs(left.left + left.afterLeft * scale - (bounds.left.right - scale)) < 1);
    assert.ok(Math.abs(right.left + right.afterLeft * scale - bounds.right.left) < 1);
    assert.ok(Math.abs(bottom.top + bottom.afterTop * scale - bounds.bottom.top) < 1);
    for (const owner of ['Resize left panel', 'Resize Inspector', 'Resize bottom panel']) {
      const handle = page.getByRole('separator', { name: owner, exact: true }),
        r = await handle.boundingBox(),
        before = await handle.getAttribute('aria-valuenow');
      const horizontal = owner === 'Resize bottom panel';
      await page.mouse.move(
        r.x + (horizontal ? 100 : r.width / 2),
        r.y + (horizontal ? r.height / 2 : 100),
      );
      await page.mouse.down();
      await page.mouse.move(
        r.x + (horizontal ? 100 : r.width / 2 + 20 * scale),
        r.y + (horizontal ? r.height / 2 + 20 * scale : 100),
        { steps: 4 },
      );
      assert.notEqual(await handle.getAttribute('aria-valuenow'), before);
      await page.keyboard.press('Escape');
      await page.mouse.up();
      assert.equal(await handle.getAttribute('aria-valuenow'), before);
    }
    await page.close();
  });
