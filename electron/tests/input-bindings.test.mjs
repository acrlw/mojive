import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
let server, browser, base;
before(async () => {
  await mkdir('../output/electron/verification', { recursive: true });
  server = await createServer({
    server: { host: '127.0.0.1', port: 0, hmr: false },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
const choose = async (page, name, value) => {
  await page.getByRole('combobox', { name, exact: true }).click();
  await page.getByRole('option', { name: value, exact: true }).click();
};
const record = async (page, name, chord) => {
  await page.locator('.settings-dialog').getByRole('button', { name, exact: true }).click();
  await page.keyboard.press(chord);
};
const pointer = async (page, name, button, modifier) => {
  const field = page.getByRole('button', { name, exact: true });
  await field.click();
  if (modifier) await page.keyboard.down(modifier);
  await field.click({ button });
  if (modifier) await page.keyboard.up(modifier);
};
test('settings remap atomically, persist and update tools, menus, guide and timeline hints', async () => {
  const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
  page.setDefaultTimeout(6000);
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.addInitScript(() => {
    if (!localStorage.getItem('mojive.electron.settings.v1'))
      localStorage.setItem('mojive.electron.settings.v1', JSON.stringify({ language: 'en' }));
  });
  await page.goto(base);
  await page.locator('.tree-row').first().waitFor();
  const open = async () => {
    await page.locator('.title-actions button:has([data-icon=settings])').click();
    await page
      .locator('.settings-navigation')
      .getByRole('button', { name: 'Interaction', exact: true })
      .click();
  };
  await open();
  await record(page, 'Scale tool', 'b');
  await record(page, 'Rotation tool', 'b');
  assert.equal(
    await page.getByRole('button', { name: 'Scale tool', exact: true }).innerText(),
    'R',
  );
  await record(page, 'Focus', 'H');
  await record(page, 'Snapping', 'Alt');
  await record(page, 'Capture keyframe', 'V');
  await record(page, 'Playback', 'P');
  assert.equal(await page.locator('.binding-settings [role=combobox]').count(), 1);
  const reset = await page
    .getByRole('button', { name: 'Reset shortcuts', exact: true })
    .boundingBox();
  const group = await page.locator('.binding-heading').boundingBox();
  assert.ok(Math.abs(reset.x + reset.width - group.x - group.width) < 1);
  assert.equal(
    await page.getByRole('button', { name: 'Reset shortcuts', exact: true }).innerText(),
    '',
  );
  await choose(page, 'Navigation preset', 'Blender');
  await pointer(page, 'Orbit 1', 'middle', 'Shift');
  await page.getByRole('alert').waitFor();
  assert.deepEqual(
    await page.evaluate(
      () =>
        JSON.parse(localStorage.getItem('mojive.electron.settings.v1')).inputBindings.pointer.orbit,
    ),
    ['middle'],
  );
  await page.keyboard.down('Alt');
  await page.getByRole('button', { name: 'Orbit 1', exact: true }).click();
  await page.keyboard.up('Alt');
  assert.equal(await page.getByRole('alert').count(), 0);
  const zoom = page.getByRole('button', { name: 'Zoom 1', exact: true });
  await zoom.click();
  await page.keyboard.down('Shift');
  await zoom.hover();
  await page.mouse.wheel(0, 80);
  await page.keyboard.up('Shift');
  await page.waitForFunction(
    () =>
      JSON.parse(localStorage.getItem('mojive.electron.settings.v1')).inputBindings.pointer
        .zoom[0] === 'shift+wheel',
  );
  await page.getByRole('button', { name: 'Add gesture: Orbit', exact: true }).click();
  await page.keyboard.down('Control');
  await page
    .getByRole('button', { name: 'Add gesture: Orbit', exact: true })
    .click({ button: 'right' });
  await page.keyboard.up('Control');
  assert.deepEqual(
    await page.evaluate(
      () =>
        JSON.parse(localStorage.getItem('mojive.electron.settings.v1')).inputBindings.pointer.orbit,
    ),
    ['alt+left', 'ctrl+right'],
  );
  await page.getByRole('button', { name: 'Orbit 2', exact: true }).click();
  await page.keyboard.press('Escape');
  assert.equal(await page.locator('.settings-dialog[open]').count(), 1);
  assert.equal(
    await page.getByRole('button', { name: 'Orbit 2', exact: true }).getAttribute('aria-pressed'),
    'false',
  );
  await page
    .locator('.settings-dialog')
    .screenshot({ path: '../output/electron/verification/input-mouse-settings.png' });
  await page.locator('.binding-settings').first().scrollIntoViewIfNeeded();
  await page
    .locator('.settings-dialog')
    .screenshot({ path: '../output/electron/verification/input-binding-settings.png' });
  await page
    .locator('.settings-header')
    .getByRole('button', { name: 'Close', exact: true })
    .click();
  const rotation = page.locator('.tool-palette button:has([data-icon=rotate])');
  assert.equal(await rotation.getAttribute('aria-label'), 'Rotation (B)');
  const canvas = page.getByTestId('viewport-canvas');
  await canvas.focus();
  await page.keyboard.press('b');
  assert.match(await rotation.getAttribute('class'), /active/);
  await page.keyboard.press('r');
  assert.match(
    await page.locator('.tool-palette button:has([data-icon=scale])').getAttribute('class'),
    /active/,
  );
  const count = await page.locator('.keyframe').count();
  await page.keyboard.press('v');
  assert.equal(await page.locator('.keyframe').count(), count + 1);
  assert.equal(
    await page
      .getByRole('button', { name: 'Capture scene pose at playhead (V)', exact: true })
      .count(),
    1,
  );
  assert.equal(await page.getByRole('button', { name: 'Play (P)', exact: true }).count(), 1);
  await page.locator('[data-menu=View]').click();
  assert.equal(
    await page
      .getByRole('menuitem', { name: /^Frame selection/ })
      .locator('kbd')
      .innerText(),
    'H',
  );
  await page.keyboard.press('Escape');
  await page.locator('[data-menu=Help]').click();
  await page.getByRole('menuitem', { name: 'Interaction guide' }).click();
  const guide = page.locator('.interaction-guide');
  for (const key of ['H', 'B', 'Alt', 'V', 'P'])
    assert.ok((await guide.getByRole('img', { name: key, exact: true }).count()) >= 1);
  assert.equal(
    await guide
      .locator('.guide-row')
      .filter({ hasText: 'Orbit' })
      .locator('[data-icon=status-mouse-left]')
      .count(),
    1,
  );
  await page.locator('.info-dialog .dialog-heading button').click();
  page.once('dialog', (dialog) => dialog.accept());
  await page.reload();
  await rotation.waitFor();
  assert.equal(await rotation.getAttribute('aria-label'), 'Rotation (B)');
  await open();
  assert.equal(
    await page.getByRole('button', { name: 'Rotation tool', exact: true }).innerText(),
    'B',
  );
  await page.getByRole('button', { name: 'Clear binding: Rotation tool', exact: true }).click();
  await page
    .locator('.settings-header')
    .getByRole('button', { name: 'Close', exact: true })
    .click();
  await canvas.focus();
  await page.keyboard.press('b');
  assert.doesNotMatch(await rotation.getAttribute('class'), /active/);
  await open();
  await page.getByRole('button', { name: 'Reset shortcuts', exact: true }).click();
  assert.equal(
    await page.getByRole('button', { name: 'Rotation tool', exact: true }).innerText(),
    'R',
  );
  assert.equal(
    await page.getByRole('combobox', { name: 'Navigation preset', exact: true }).innerText(),
    'Mojive',
  );
  await record(page, 'Rotation tool', 'Control+Alt+h');
  assert.equal(
    await page.getByRole('button', { name: 'Rotation tool', exact: true }).textContent(),
    'Ctrl+Alt+H',
  );
  await page
    .locator('.settings-header')
    .getByRole('button', { name: 'Close', exact: true })
    .click();
  await canvas.focus();
  await page.keyboard.press('Control+Alt+h');
  assert.match(await rotation.getAttribute('class'), /active/);
  await open();
  await page.getByRole('textbox', { name: 'Search all settings', exact: true }).fill('shortcut');
  assert.ok((await page.locator('.binding-settings .settings-row').count()) >= 10);
  assert.deepEqual(errors, []);
  await page.close();
});
test('navigation presets and custom mouse chords own actual viewport orbit, pan and dolly', async () => {
  const page = await browser.newPage({ viewport: { width: 800, height: 700 } });
  page.setDefaultTimeout(6000);
  await page.goto(base + '/tests/harness/selection-review.html');
  await page.waitForFunction(() => window.selectionReview?.api.current);
  const set = async (preset, patch = {}) => {
    await page.evaluate(
      async ({ preset, patch }) => {
        const m = await import('/src/input-bindings.ts');
        window.selectionReview.setInputBindings({
          ...m.DEFAULT_INPUT_BINDINGS,
          pointer: { ...m.NAVIGATION_PRESETS[preset], ...patch },
        });
      },
      { preset, patch },
    );
    await page.evaluate(
      () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))),
    );
  };
  const view = () => page.evaluate(() => window.selectionReview.api.current.getCameraView());
  const drag = async (button, modifier) => {
    if (modifier) await page.keyboard.down(modifier);
    await page.mouse.move(120, 140);
    await page.mouse.down({ button });
    await page.mouse.move(190, 185, { steps: 5 });
    await page.mouse.up({ button });
    if (modifier) await page.keyboard.up(modifier);
    await page.evaluate(
      () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))),
    );
  };
  const distance = (v) => Math.hypot(...v.position.map((n, i) => n - v.target[i]));
  await set('Blender');
  let before = await view();
  await drag('left');
  assert.deepEqual(await view(), before);
  await drag('middle');
  let after = await view();
  assert.notDeepEqual(after.position, before.position);
  assert.deepEqual(after.target, before.target);
  before = after;
  await drag('middle', 'Shift');
  after = await view();
  assert.notDeepEqual(after.target, before.target);
  assert.ok(Math.abs(distance(after) - distance(before)) < 1e-6);
  before = after;
  await drag('middle', 'Control');
  after = await view();
  assert.notEqual(distance(after), distance(before));
  await set('Unity', { zoom: ['shift+wheel'] });
  before = await view();
  await drag('left');
  assert.deepEqual(await view(), before);
  await drag('left', 'Alt');
  assert.notDeepEqual((await view()).position, before.position);
  before = await view();
  await page.mouse.wheel(0, 100);
  assert.deepEqual(await view(), before);
  await page.keyboard.down('Shift');
  await page.mouse.wheel(0, 100);
  await page.keyboard.up('Shift');
  await page.waitForFunction(
    (old) => JSON.stringify(window.selectionReview.api.current.getCameraView()) !== old,
    JSON.stringify(before),
  );
  await page.close();
});

test('WASD/QE translates the camera with native axes and rate, stops on release and respects input ownership', async () => {
  const page = await browser.newPage({ viewport: { width: 900, height: 700 } });
  await page.goto(base + '/tests/harness/selection-review.html');
  await page.waitForFunction(() => window.selectionReview?.api.current);
  const canvas = page.getByTestId('viewport-canvas');
  await canvas.focus();
  await page.evaluate(() => {
    window.flyNow = performance.now();
    Object.defineProperty(performance, 'now', { configurable: true, value: () => window.flyNow });
  });
  const step = async (n = 10) => {
    for (let i = 0; i < n; i++) {
      await page.evaluate(() => {
        window.flyNow += 20;
      });
      await page.evaluate(
        () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))),
      );
    }
  };
  const view = () => page.evaluate(() => window.selectionReview.api.current.getCameraView());
  const sub = (a, b) => a.map((x, i) => x - b[i]);
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const norm = (a) => Math.hypot(...a);
  for (const key of ['w', 's', 'a', 'd', 'q', 'e']) {
    const before = await view();
    await page.keyboard.down(key);
    await step();
    await page.keyboard.up(key);
    await step(1);
    const after = await view(),
      delta = sub(after.target, before.target),
      moved = sub(after.position, before.position);
    assert.ok(norm(sub(delta, moved)) < 1e-6, 'eye and target move together');
    const forward = sub(before.target, before.position),
      distance = norm(forward);
    assert.ok(Math.abs(norm(delta) - distance * 1.6 * 0.2) < 1e-5, `${key}: native speed`);
    if (key === 'w' || key === 's') assert.ok(dot(delta, forward) * (key === 'w' ? 1 : -1) > 0);
    if (key === 'q' || key === 'e') {
      assert.ok(delta[2] * (key === 'q' ? 1 : -1) > 0);
      assert.ok(Math.hypot(delta[0], delta[1]) < 1e-6);
    }
    if (key === 'a' || key === 'd') {
      const right = [forward[1], -forward[0], 0];
      assert.ok(dot(delta, right) * (key === 'd' ? 1 : -1) > 0);
    }
    await step(3);
    assert.deepEqual(await view(), after, 'release stops movement');
  }
  await page.keyboard.down('w');
  await page.keyboard.down('s');
  const cancelled = await view();
  await step();
  assert.deepEqual(await view(), cancelled);
  await page.keyboard.up('s');
  await step();
  assert.notDeepEqual((await view()).target, cancelled.target);
  await page.keyboard.up('w');
  await step(1);
  await page.keyboard.down('w');
  await page.mouse.move(110, 120);
  await page.mouse.down({ button: 'right' });
  const combined = await view();
  await step(3);
  assert.notDeepEqual(
    (await view()).target,
    combined.target,
    'camera pointer navigation can accompany fly keys',
  );
  await page.mouse.up({ button: 'right' });
  const releasedPointer = await view();
  await step(3);
  assert.notDeepEqual(
    (await view()).target,
    releasedPointer.target,
    'pointer release retains held fly ownership',
  );
  await page.keyboard.up('w');
  await page.keyboard.down('Control');
  const commandView = await view();
  await page.keyboard.down('s');
  await step();
  assert.deepEqual(await view(), commandView);
  await page.keyboard.up('s');
  await page.keyboard.up('Control');
  await page.keyboard.down('w');
  await step(2);
  await page.keyboard.press('Escape');
  const escaped = await view();
  await step();
  assert.deepEqual(await view(), escaped);
  await page.keyboard.up('w');
  await page.keyboard.down('w');
  await step(2);
  await page.evaluate(() => {
    const input = document.createElement('input');
    input.id = 'typing-owner';
    document.body.append(input);
    input.focus();
  });
  const blurred = await view();
  await page.keyboard.up('w');
  await page.keyboard.type('wasdqe');
  await step();
  assert.deepEqual(await view(), blurred);
  await canvas.focus();
  await step();
  assert.deepEqual(await view(), blurred, 'focus return never resumes a released gesture');
  await page.close();
});

test('legacy factory bindings migrate while custom keys and new fly bindings remain exclusive', async () => {
  const page = await browser.newPage();
  await page.goto(base + '/tests/harness/selection-review.html');
  const result = await page.evaluate(async () => {
    const m = await import('/src/input-bindings.ts');
    const old = {
      select: 'Q',
      position: 'W',
      rotation: 'E',
      scale: 'R',
      joint: 'J',
      focus: 'F',
      frame: 'T',
      snap: 'Shift',
      capture: 'K',
      playback: 'Space',
    };
    const migrated = m.normalizeInputBindings({ keys: old });
    const custom = m.normalizeInputBindings({ keys: { ...old, rotation: 'B' } });
    const swapped = m.remapKey(m.DEFAULT_INPUT_BINDINGS, 'flyForward', 'G');
    const unique = (keys) => {
      const bound = Object.values(keys).filter(Boolean);
      return new Set(bound).size === bound.length;
    };
    let reserved = false;
    try {
      m.remapKey(swapped, 'flyForward', 'Control+Alt+S');
    } catch {
      reserved = true;
    }
    return {
      migrated: migrated.keys,
      defaults: m.DEFAULT_INPUT_BINDINGS.keys,
      custom: custom.keys,
      swapped: swapped.keys,
      unique: unique(custom.keys) && unique(swapped.keys),
      reserved,
      focusAll: m.focusAllAccelerator(
        m.remapKey(m.DEFAULT_INPUT_BINDINGS, 'flyForward', 'Shift+F'),
      ),
    };
  });
  assert.deepEqual(result.migrated, result.defaults);
  assert.equal(result.custom.rotation, 'B');
  assert.equal(result.custom.select, 'Q');
  assert.equal(result.custom.flyUp, null);
  assert.equal(result.swapped.flyForward, 'G');
  assert.equal(result.swapped.position, 'W');
  assert.equal(result.unique, true);
  assert.equal(result.reserved, true);
  assert.equal(result.focusAll, null);
  await page.close();
});

test('system window shortcuts cannot be recorded or restored as viewport bindings', async () => {
  const page = await browser.newPage();
  await page.goto(base + '/tests/harness/selection-review.html');
  const blocked = await page.evaluate(async () => {
    const m = await import('/src/input-bindings.ts');
    const results = [];
    for (const platform of ['linux', 'win32', 'darwin']) {
      window.mojive = { platform };
      const chords =
        platform === 'darwin'
          ? [
              'Alt+F4',
              'Meta+Q',
              'Meta+W',
              'Meta+H',
              'Alt+Meta+H',
              'Meta+M',
              'Control+Meta+F',
              'Alt+Meta+I',
            ]
          : ['Alt+F4'];
      for (const chord of chords) {
        let rejected = false;
        try {
          m.remapKey(m.DEFAULT_INPUT_BINDINGS, 'rotation', chord);
        } catch {
          rejected = true;
        }
        const restored = m.normalizeInputBindings({
          keys: { ...m.DEFAULT_INPUT_BINDINGS.keys, rotation: chord },
        });
        results.push({ platform, chord, rejected, restored: restored.keys.rotation });
      }
    }
    return results;
  });
  assert.ok(
    blocked.every((value) => value.rejected && value.restored === 'R'),
    JSON.stringify(blocked),
  );
  await page.close();
});

test('fly keys preserve scene-camera View and create one cancelable Pilot transaction', async () => {
  const page = await browser.newPage({ viewport: { width: 1540, height: 980 } });
  await page.goto(base + '/tests/harness/camera-review.html');
  await page.waitForFunction(() => window.cameraReview?.api.current);
  const canvas = page.getByTestId('viewport-canvas');
  const state = () =>
    page.evaluate(() => {
      const f = window.cameraReview;
      return {
        view: f.api.current.getCameraView(),
        begins: f.begins,
        commits: f.commits,
        state: f.state,
      };
    });
  const hold = async () => {
    await page.keyboard.down('w');
    await page.waitForTimeout(100);
  };
  await page.evaluate(() => {
    const f = window.cameraReview;
    f.api.current.useSceneCamera(f.cameraId, false);
  });
  await page.waitForTimeout(450);
  await canvas.focus();
  const locked = await state();
  await hold();
  await page.keyboard.up('w');
  assert.deepEqual((await state()).view, locked.view);
  await page.evaluate(() => {
    const f = window.cameraReview;
    f.api.current.useSceneCamera(f.cameraId, true, false);
  });
  await canvas.focus();
  const before = await state();
  await hold();
  const bounds = await canvas.boundingBox();
  await page.mouse.move(bounds.x + 120, bounds.y + 180);
  await page.mouse.down({ button: 'right' });
  await page.mouse.move(bounds.x + 132, bounds.y + 185, { steps: 3 });
  await page.mouse.up({ button: 'right' });
  assert.equal(
    (await state()).commits,
    before.commits,
    'combined navigation commits after its last input releases',
  );
  await page.keyboard.down('d');
  await page.waitForTimeout(60);
  await page.keyboard.up('w');
  await page.keyboard.up('d');
  await page.waitForFunction((n) => window.cameraReview.commits === n, before.commits + 1);
  const moved = await state();
  assert.notDeepEqual(moved.view.position, before.view.position);
  assert.equal(moved.begins, before.begins + 1);
  assert.equal(moved.commits, before.commits + 1);
  await hold();
  await page.keyboard.press('Escape');
  await page.keyboard.up('w');
  assert.deepEqual(
    (await state()).view.position,
    moved.view.position,
    'Escape restores the Pilot gesture start',
  );
  await page.close();
});
