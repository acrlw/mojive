import { chooseSampleScene } from '../tests/helpers/editable-scene.mjs';
import { _electron as electron } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, writeFile, readFile } from 'node:fs/promises';
import path from 'node:path';
import { build } from 'esbuild';
const root = process.cwd(),
  out = path.join(root, '../output/electron/verification');
await mkdir(out, { recursive: true });
// The desktop keeps its compiled renderer and real MuJoCo startup. Authoring checks
// import the maintained kinematic fixture through the public workspace drop path.
const fixtureBundle = await build({
  stdin: {
    contents: "export { createCatalogScene } from './src/SceneCatalog'; export { saveWorkspace } from './src/editor';",
    resolveDir: root,
  },
  bundle: true,
  write: false,
  format: 'iife',
  globalName: 'editableFixture',
  platform: 'browser',
});
const app = await electron.launch({
  args: [
    root,
    '--hidden',
    '--user-data-dir=' + (await mkdtemp(path.join(out, 'acceptance-profile-'))),
  ],
  cwd: root,
  timeout: 45000,
});
let page;
const errors = [],
  passed = [],
  primitiveNames = [];
const wait = () => page.waitForTimeout(150);
async function action(id) {
  await app.evaluate(({ Menu, BrowserWindow }, id) => {
    const item = Menu.getApplicationMenu()?.getMenuItemById(id);
    if (!item && process.platform !== 'darwin') {
      BrowserWindow.getAllWindows()[0].webContents.send('mojive:menu-action', id);
      return;
    }
    if (!item) throw Error('Missing menu ' + id);
    if (!item.enabled) throw Error('Disabled menu ' + id);
    item.click(item, BrowserWindow.getAllWindows()[0], {});
  }, id);
  await wait();
}
async function check(name, fn) {
  let timer;
  try {
    await Promise.race([
      fn(),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`Desktop check timed out: ${name}`)), 60000);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
  passed.push(name);
  console.log('✓ ' + name);
}
async function capture(name) {
  await page.screenshot({ path: path.join(out, name + '.png'), timeout: 15000 });
}
async function openEditableJointScene() {
  await page.locator('.busy-overlay').waitFor({ state: 'detached' });
  await page.evaluate(`(async () => {
    ${fixtureBundle.outputFiles[0].text}
    const doc = await editableFixture.createCatalogScene('joint_gizmo');
    const transfer = new DataTransfer();
    transfer.items.add(new File([editableFixture.saveWorkspace(doc, [], 10)],
      'joint-fixture.mojive.json', { type: 'application/json' }));
    document.querySelector('.app-shell').dispatchEvent(new DragEvent('drop', {
      bubbles: true, cancelable: true, dataTransfer: transfer,
    }));
  })()`);
  await page.locator('.tree-row.type-robot[data-tooltip="joint_gizmo_acceptance"]').waitFor();
  await page.locator('.busy-overlay').waitFor({ state: 'detached' });
  assert.equal(await page.locator('.simulation-toolbar').count(), 0);
}
async function typeValue(label, value) {
  const input = page.getByLabel(label, { exact: true });
  await input.fill(String(value));
  await input.press('Enter');
}
async function selectNode(name) {
  const button = page.locator('.panel-header').getByRole('button', { name: 'Search', exact: true });
  if ((await page.locator('.panel-search input').count()) === 0) await button.click();
  await page.locator('.panel-search input').fill(name);
  const row = page
    .getByRole('treeitem')
    .filter({ has: page.locator('span').filter({ hasText: new RegExp('^' + name + '$') }) })
    .first();
  await row.click();
  return row;
}
try {
  page = await app.firstWindow();
  page.on('pageerror', (e) => {
    errors.push(e.stack || String(e));
    console.error(e.stack || String(e));
  });
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });
  await page.waitForSelector('[role=treeitem]');
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => {
    localStorage.removeItem('mojive.electron.settings.v1');
    localStorage.removeItem('mojive.layout');
  });
  await page.reload();
  await page.waitForSelector('[role=treeitem]');
  await page.evaluate(() => document.fonts.ready);
  await wait();
  await check('Mojive G2 smoothing retains authored UI radii', async () => {
    await page.waitForFunction(
      () => document.querySelector('.tool-palette')?.dataset.smoothSurface,
    );
    const profile = await page.evaluate(() => {
      const palette = document.querySelector('.tool-palette');
      return {
        radii: JSON.parse(palette.dataset.smoothRadii),
        amount: Number(palette.dataset.smoothAmount),
        paint: palette.style.getPropertyValue('--smooth-paint'),
      };
    });
    assert.deepEqual(profile.radii, [8, 8, 8, 8]);
    assert.equal(profile.amount, 0.6);
    assert.ok(profile.paint.includes('svg'));
  });
  await check(
    'Chinese workspace, platform menu, embedded fonts and no application logo',
    async () => {
      assert.equal(await page.locator('html').getAttribute('lang'), 'zh-CN');
      assert.equal((await page.locator('.document-title').innerText()).trim(), '关节操作示例');
      assert.equal(await page.locator('.timeline .lane-header').innerText(), '关节操作示例');
      await page.locator('.simulation-toolbar').waitFor();
      assert.equal(
        await page.locator('.titlebar nav').count(),
        process.platform === 'darwin' ? 0 : 1,
      );
      assert.equal(await page.locator('.brand-mark').count(), 0);
      const s = await page.evaluate(() => ({
        mi: document.fonts.check('14px MiSans'),
        mono: document.fonts.check('16px "JetBrains Mono"'),
        body: getComputedStyle(document.body).fontFamily,
        native: window.mojive.hasNativeMenu,
      }));
      assert.ok(s.mi && s.mono);
      assert.equal(s.native, process.platform === 'darwin');
      if (s.native) {
        const nativeAdd = await app.evaluate(({ Menu }) => {
          const add = Menu.getApplicationMenu().items.find((item) =>
            item.submenu?.items.some((child) => child.id === 'add.camera'),
          );
          return add.submenu.items.map(({ id, type, label }) => ({ id, type, label }));
        });
        const planeIndex = nativeAdd.findIndex((item) => item.id === 'add.ground');
        assert.ok(planeIndex >= 0);
        assert.equal(nativeAdd[planeIndex + 1].type, 'separator');
        assert.equal(nativeAdd[planeIndex + 2].id, 'add.light');
        assert.equal(nativeAdd.find((item) => item.id === 'add.camera').label, '相机');
      } else {
        await page.locator('[data-menu=File]').click();
        for (const name of ['Edit', 'Add', 'View', 'Help']) {
          await page.locator(`[data-menu=${name}]`).hover();
          await page.locator(`.menu-${name.toLowerCase()}:popover-open`).waitFor();
        }
        await page.locator('[data-menu=Edit]').hover();
        assert.equal(await page.locator('.menu-edit kbd').first().innerText(), 'Ctrl+Z');
        await capture('platform-menu');
        await page.keyboard.press('Escape');
        await page.locator('[data-menu=View]').hover();
        assert.equal(await page.locator('.menu').count(), 0);
      }
      assert.equal(await page.locator('.left-panel .panel-header > .icon').count(), 0);
      assert.equal(await page.getByRole('button', { name: '聚焦全部', exact: true }).count(), 1);
      assert.equal(await page.locator('.viewport-topbar .geometry-selector').count(), 1);
      assert.match(s.body, /MiSans/);
      assert.equal(await page.locator('.geometry-selector button').count(), 4);
    },
  );
  await check(
    'Editable joint workspace retains local authoring and scene context actions',
    async () => {
      await openEditableJointScene();
      const row = page.locator('.tree-row').filter({ hasText: '01_revolute_y' }).first();
      await row.click({ button: 'right' });
      await page.locator('.menu-object:popover-open').waitFor();
      await page.getByRole('menuitem', { name: '聚焦对象', exact: true }).waitFor();
      await page.getByRole('menuitem', { name: '删除对象', exact: true }).waitFor();
      await capture('scene-context');
      const previousClipboard = await app.evaluate(({ clipboard }) => clipboard.readText());
      await page.getByRole('menuitem', { name: '复制名称', exact: true }).click();
      assert.equal(await app.evaluate(({ clipboard }) => clipboard.readText()), '01_revolute_y');
      await app.evaluate(({ clipboard }, text) => clipboard.writeText(text), previousClipboard);
      assert.equal((await page.locator('.type-all').innerText()).trim(), '45');
    },
  );
  await check(
    'Themed shortcut tooltip and graphical interaction guide use production inputs',
    async () => {
      const rotation = page.locator('.tool-palette button:has([data-icon=rotate])');
      await rotation.hover();
      await page.getByRole('tooltip').waitFor();
      assert.equal(await page.getByRole('tooltip').locator('kbd').innerText(), 'R');
      assert.equal(await rotation.getAttribute('title'), null);
      await capture('desktop-tooltip-dark-zh');
      await page.keyboard.press('Escape');
      await action('help.interaction');
      const guide = page.locator('.interaction-guide');
      await guide.waitFor();
      assert.equal(await guide.locator('.guide-row').count(), 17);
      assert.equal(await guide.locator('[data-icon^=status-mouse-]').count(), 7);
      if (process.platform !== 'darwin') assert.doesNotMatch(await guide.innerText(), /[⌘⌥]/);
      await capture('desktop-input-guide-dark-zh');
      await page.locator('.info-dialog .dialog-heading button').click();
    },
  );
  await capture('workspace-revised-dark-zh');
  await check('Settings opens and locale/theme update without reloading the scene', async () => {
    await action('app.settings');
    await page.locator('dialog[open]').waitFor();
    await capture('settings-revised-zh');
    await page.getByRole('combobox', { name: '语言', exact: true }).click();
    await page.getByRole('option', { name: 'English', exact: true }).click();
    await page.locator('dialog[open]').getByRole('button', { name: 'Light', exact: true }).click();
    await page.locator('dialog[open]').getByRole('button', { name: 'Close', exact: true }).click();
    assert.equal(await page.locator('html').getAttribute('data-theme'), 'light');
    assert.equal((await page.getByRole('treeitem').count()) > 0, true);
    await capture('workspace-revised-light-en');
    await action('app.settings');
    await page.locator('dialog[open]').getByRole('button', { name: 'Dark', exact: true }).click();
    await page.locator('dialog[open]').getByRole('button', { name: 'Close', exact: true }).click();
  });
  await check(
    'Desktop binding capture records keys and mouse gestures with shared hints',
    async () => {
      await action('app.settings');
      const dialog = page.locator('.settings-dialog');
      await dialog.getByRole('button', { name: 'Interaction', exact: true }).click();
      await dialog.getByRole('button', { name: 'Scale tool', exact: true }).click();
      await page.keyboard.press('b');
      await dialog.getByRole('button', { name: 'Rotation tool', exact: true }).click();
      await page.keyboard.press('b');
      assert.equal(
        await dialog.getByRole('button', { name: 'Scale tool', exact: true }).innerText(),
        'R',
      );
      await dialog.screenshot({ path: path.join(out, 'desktop-input-bindings-dark.png') });
      const orbit = dialog.getByRole('button', { name: 'Orbit 1', exact: true });
      await orbit.click();
      await page.keyboard.down('Alt');
      await orbit.click({ button: 'left' });
      await page.keyboard.up('Alt');
      const mac = await app.evaluate(() => process.platform === 'darwin');
      assert.equal(await orbit.locator('kbd').innerText(), mac ? '⌥' : 'Alt');
      await dialog.screenshot({ path: path.join(out, 'desktop-mouse-bindings-dark.png') });
      await dialog
        .locator('.settings-header')
        .getByRole('button', { name: 'Close', exact: true })
        .click();
      await page.getByTestId('viewport-canvas').focus();
      await page.keyboard.press('b');
      assert.match(
        await page.locator('.tool-palette button:has([data-icon=rotate])').getAttribute('class'),
        /active/,
      );
      await action('app.settings');
      await dialog.getByRole('button', { name: 'Interaction', exact: true }).click();
      await dialog.getByRole('button', { name: 'Reset shortcuts', exact: true }).click();
      await dialog
        .locator('.settings-header')
        .getByRole('button', { name: 'Close', exact: true })
        .click();
    },
  );
  await check(
    'Desktop fly keys move the camera without switching tools and stop on release',
    async () => {
      await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].showInactive());
      await page.waitForFunction(() => !document.hidden);
      const canvas = page.getByTestId('viewport-canvas');
      await canvas.focus();
      await page.waitForTimeout(150);
      const tool = await page
        .locator('.tool-palette button.active')
        .first()
        .getAttribute('aria-label');
      const before = await canvas.screenshot();
      await page.keyboard.down('w');
      await page.waitForTimeout(160);
      await page.keyboard.up('w');
      await page.waitForTimeout(100);
      const moved = await canvas.screenshot();
      assert.notDeepEqual(moved, before);
      assert.equal(
        await page.locator('.tool-palette button.active').first().getAttribute('aria-label'),
        tool,
      );
      await page.waitForTimeout(100);
      assert.deepEqual(await canvas.screenshot(), moved);
      await capture('desktop-fly-camera');
    },
  );
  await check('Three panel splitters resize, cancel and retain sizes after reload', async () => {
    await page.evaluate(() => {
      window.dragEvidence = [];
      for (const name of [
        'pointerdown',
        'pointermove',
        'pointerup',
        'pointercancel',
        'lostpointercapture',
        'dblclick',
        'blur',
      ])
        window.addEventListener(
          name,
          (e) => {
            if (name !== 'pointermove' || e.buttons)
              window.dragEvidence.push({
                type: name,
                target: e.target?.getAttribute?.('aria-label'),
                x: e.clientX,
                y: e.clientY,
                buttons: e.buttons,
              });
          },
          true,
        );
    });
    for (const [label, axis, d] of [
      ['Resize left panel', 'x', 60],
      ['Resize Inspector', 'x', -40],
      ['Resize bottom panel', 'y', -45],
    ]) {
      const sep = page.getByRole('separator', { name: label, exact: true });
      const before = Number(await sep.getAttribute('aria-valuenow'));
      const box = await sep.boundingBox(),
        x = box.x + box.width / 2,
        y = box.y + box.height / 2;
      await page.mouse.move(x, y);
      await page.mouse.down();
      await page.mouse.move(x + (axis === 'x' ? d : 0), y + (axis === 'y' ? d : 0), { steps: 8 });
      await page.mouse.up();
      const after = Number(await sep.getAttribute('aria-valuenow'));
      assert.ok(
        after > before + 25,
        `${label}: ${before} → ${after}; evidence ${JSON.stringify(await page.evaluate(() => window.dragEvidence))}; hit ${await page.evaluate(({ x, y }) => document.elementFromPoint(x, y)?.outerHTML.slice(0, 220), { x, y })}`,
      );
      const saved = Number(await sep.getAttribute('aria-valuenow'));
      await sep.press('ArrowRight');
      await sep.press('ArrowLeft');
      assert.equal(Number(await sep.getAttribute('aria-valuenow')), saved);
    }
    const stored = await page.evaluate(() => localStorage.getItem('mojive.layout'));
    await page.reload();
    await page.waitForSelector('[role=treeitem]');
    assert.equal(await page.evaluate(() => localStorage.getItem('mojive.layout')), stored);
    await openEditableJointScene();
  });
  await check(
    'Type filters and search return selectable rows without phantom expand controls',
    async () => {
      await page.getByRole('button', { name: 'Joint', exact: true }).click();
      assert.equal(await page.getByRole('treeitem').count(), 9);
      assert.equal(await page.locator('.tree-row button.tree-chevron').count(), 0);
      await page.getByRole('button', { name: 'All', exact: true }).click();
      await selectNode('03_ball_xyz');
      assert.equal(await page.getByRole('treeitem').count(), 1);
      assert.equal(await page.locator('.tree-row button.tree-chevron').count(), 0);
      const gap = await page
        .getByRole('treeitem')
        .evaluate(
          (el) =>
            el.querySelector('.tree-label').getBoundingClientRect().left -
            el.getBoundingClientRect().left,
        );
      assert.ok(gap < 65, `Filtered tree label inset ${gap}`);
      await capture('scene-search-flat');
      await page.getByRole('treeitem').dblclick();
      await page.waitForTimeout(400);
      assert.equal(await page.locator('.inspector-heading strong').textContent(), '03_ball_xyz');
    },
  );
  await check(
    'All four geometry modes, projection icons and persistent snap control respond',
    async () => {
      for (const mode of ['Default', 'Visual', 'Collision', 'Both']) {
        await page
          .locator('.geometry-selector')
          .getByRole('button', { name: mode, exact: true })
          .click();
        assert.equal(
          await page
            .locator('.geometry-selector')
            .getByRole('button', { name: mode, exact: true })
            .getAttribute('aria-pressed'),
          'true',
        );
      }
      await page
        .locator('.geometry-selector')
        .getByRole('button', { name: 'Default', exact: true })
        .click();
      await page.getByRole('button', { name: 'Snapping', exact: true }).click();
      assert.equal(
        await page
          .getByRole('button', { name: 'Snapping', exact: true })
          .getAttribute('aria-pressed'),
        'true',
      );
      await page.getByRole('button', { name: 'Projection', exact: true }).click();
      await page.getByRole('menuitem', { name: 'Orthographic', exact: true }).click();
      assert.match(await page.locator('.viewport-navigation').innerText(), /Orthographic/);
      await action('view.perspective');
      assert.match(await page.locator('.viewport-navigation').innerText(), /Perspective/);
    },
  );
  await check(
    'Ball and free joints expose three/six pose coordinates and update kinematically',
    async () => {
      await page.getByRole('button', { name: 'Joints', exact: true }).click();
      const ball = page
        .locator('.joint-entry')
        .filter({ has: page.locator('.entity-card-heading').filter({ hasText: '03_ball_xyz' }) });
      // The preceding scene selection opens this pose when the panel mounts.
      // Wait for that effect rather than racing it with a second toggle.
      await ball.locator('.entity-card-heading[aria-expanded="true"]').waitFor();
      await ball.getByLabel('Rotation X', { exact: true }).waitFor();
      await ball.getByLabel('Rotation X', { exact: true }).fill('20');
      await ball.getByLabel('Rotation X', { exact: true }).press('Enter');
      assert.ok(
        Math.abs(Number(await ball.getByLabel('Rotation X', { exact: true }).inputValue()) - 20) <
          0.01,
      );
      const free = page
        .locator('.joint-entry')
        .filter({ has: page.locator('.entity-card-heading').filter({ hasText: '04_free_6dof' }) });
      if (!(await free.getByLabel('Position Z', { exact: true }).count()))
        await free.getByRole('button', { name: /Expand/ }).click();
      await free.getByLabel('Position Z', { exact: true }).fill('0.25');
      await free.getByLabel('Position Z', { exact: true }).press('Enter');
      assert.equal(Number(await free.getByLabel('Position Z', { exact: true }).inputValue()), 0.25);
      await capture('joint-panel-revised');
      await page.getByRole('button', { name: 'Scene', exact: true }).click();
    },
  );
  await check(
    'Asset browser loads distinct scenes with unsaved replacement protection',
    async () => {
      await chooseSampleScene(page, 'Atelier arm');
      await page.getByRole('alertdialog', { name: 'Unsaved changes' }).waitFor();
      await page.getByRole('button', { name: 'Cancel', exact: true }).click();
      assert.match(await page.locator('.document-title').innerText(), /Joint/);
      await chooseSampleScene(page, 'Atelier arm');
      await page.getByRole('button', { name: "Don't Save", exact: true }).click();
      await page.waitForFunction(() =>
        document.querySelector('.document-title')?.textContent?.includes('Atelier'),
      );
      await page.locator('.busy-overlay').waitFor({ state: 'detached' });
      await page.locator('.activity-rail').getByRole('button', { name: 'Scene', exact: true }).click();
      await selectNode('elbow');
      await page.getByRole('button', { name: 'Joint (J)', exact: true }).click();
      await capture('arm-revised');
    },
  );
  await check('Native Add creates cone and ellipsoid entities with reversible geometry', async () => {
    await page.locator('.panel-search input').fill('');
    for (const [kind, x, z] of [['cone', -1.2, 0.5], ['ellipsoid', 1.2, 0.35]]) {
      await action(`add.${kind}`);
      const name = await page.locator('.inspector-heading strong').textContent();
      primitiveNames.push(name);
      assert.match(name.toLowerCase(), new RegExp(`^${kind} `));
      assert.equal(await page.locator(`.tree-row[data-tooltip="${name}"]`).count(), 1);
      await action('edit.undo');
      assert.equal(await page.locator(`.tree-row[data-tooltip="${name}"]`).count(), 0);
      await action('edit.redo');
      await selectNode(name);
      await typeValue('Position X', x);
      await typeValue('Position Y', 0);
      await typeValue('Position Z', z);
      await page.locator('.panel-search input').fill('');
    }
    await action('tool.select');
    await action('view.focusAll');
    await page.waitForTimeout(500);
    await capture('primitive-additions');
  });
  await check(
    'Add primitive, inspector vector fields, undo/redo and native text input ownership',
    async () => {
      await action('add.box');
      const initialX = Number(await page.getByLabel('Position X', { exact: true }).inputValue());
      await typeValue('Position X', 1.25);
      assert.equal(Number(await page.getByLabel('Position X', { exact: true }).inputValue()), 1.25);
      await page.locator('.inspector-heading').click();
      await action('edit.undo');
      assert.equal(Number(await page.getByLabel('Position X', { exact: true }).inputValue()), initialX);
      await action('edit.redo');
      assert.equal(Number(await page.getByLabel('Position X', { exact: true }).inputValue()), 1.25);
      await page.locator('.panel-search input').fill('f');
      assert.equal(await page.locator('.panel-search input').inputValue(), 'f');
      await page.locator('.panel-search input').fill('');
    },
  );
  await check(
    'Inspector numeric scrubbing, native axis badges and custom material color work together',
    async () => {
      const field = page.getByLabel('Position X', { exact: true }),
        initial = Number(await field.inputValue()),
        box = await field.boundingBox();
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.down();
      await page.mouse.move(box.x + box.width / 2 + 30, box.y + box.height / 2, { steps: 8 });
      await page.mouse.up();
      assert.ok(Number(await field.inputValue()) > initial + 0.1);
      await action('edit.undo');
      assert.equal(Number(await field.inputValue()), initial);
      const right = page.getByRole('separator', { name: 'Resize Inspector', exact: true });
      for (let i = 0; i < 10; i++) await right.press('ArrowLeft');
      await capture('inspector-wide');
      await page
        .locator('.inspector-tabs')
        .getByRole('button', { name: 'Material', exact: true })
        .click();
      const materialEdges = await page.locator('.material-property').evaluateAll((rows) =>
        rows.slice(0, 3).map((row) => {
          const bounds = row.lastElementChild.getBoundingClientRect();
          return [bounds.left, bounds.right];
        }),
      );
      assert.equal(materialEdges.length, 3);
      for (const edges of materialEdges) {
        assert.ok(Math.abs(edges[0] - materialEdges[0][0]) < 1);
        assert.ok(Math.abs(edges[1] - materialEdges[0][1]) < 1);
      }
      await page.getByRole('button', { name: 'Material 1 Color', exact: true }).click();
      assert.equal(await page.locator('input[type=color]').count(), 0);
      const corners = await page.evaluate(() => {
        const radius = (element) =>
          element.dataset.smoothRadii
            ? JSON.parse(element.dataset.smoothRadii)[0]
            : parseFloat(getComputedStyle(element).borderTopLeftRadius);
        const trigger = document.querySelector('.inspector .color-trigger');
        const shell = document.querySelector('.color-picker-surface');
        const button = shell.querySelector('.color-comparison-item');
        return [
          [trigger, trigger.querySelector('.color-swatch')],
          [shell, button],
          [button, button.querySelector('.color-swatch')],
        ].map(([outer, inner]) => {
          const a = outer.getBoundingClientRect(),
            b = inner.getBoundingClientRect();
          const scale = a.width / parseFloat(getComputedStyle(outer).width);
          return {
            outer: radius(outer),
            inner: radius(inner),
            left: (b.left - a.left) / scale,
            top: (b.top - a.top) / scale,
          };
        });
      });
      assert.deepEqual(
        corners.map(({ outer, inner }) => [outer, inner]),
        [
          [8, 4],
          [16, 8],
          [8, 4],
        ],
      );
      for (const corner of corners) {
        assert.ok(Math.abs(corner.left - corner.top) < 0.1);
        assert.ok(Math.abs(corner.outer - corner.left - corner.inner) < 0.1);
      }
      await page.getByRole('textbox', { name: 'Hex color', exact: true }).fill('#336699');
      await page.getByRole('textbox', { name: 'Hex color', exact: true }).press('Enter');
      await page.getByRole('button', { name: 'Restore original color', exact: true }).hover();
      await capture('material-picker-revised');
      await page.getByRole('button', { name: 'RGB 255', exact: true }).click();
      assert.deepEqual(
        (await page.locator('.color-channel-fields .axis').allTextContents()).map((text) =>
          text.trim(),
        ),
        ['R', 'G', 'B'],
      );
      await typeValue('R', 128);
      assert.equal(await page.getByRole('textbox', { name: 'R', exact: true }).inputValue(), '128');
      await capture('material-rgb-revised');
      await page.getByRole('button', { name: 'RGB 1', exact: true }).click();
      await capture('material-rgb1-revised');
      await page.getByRole('button', { name: 'Cancel', exact: true }).click();
      await page
        .locator('.inspector-tabs')
        .getByRole('button', { name: 'Properties', exact: true })
        .click();
      await right.dblclick();
      await page
        .locator('.viewport-navigation')
        .getByRole('button', { name: 'Layers', exact: true })
        .click();
      const cube = page.getByRole('switch', { name: 'View cube', exact: true });
      await cube.click();
      assert.equal(await cube.getAttribute('aria-checked'), 'false');
      await cube.click();
      await capture('layers-revised');
      await cube.press('Escape');
      const metrics = await page.locator('.render-stats').innerText();
      assert.match(metrics, /FPS/);
      assert.doesNotMatch(metrics, /ms/);
      assert.equal(
        await page
          .locator('footer .stat-calls, footer .stat-triangles, footer button:has([data-icon=key])')
          .count(),
        0,
      );
      assert.equal(await page.locator('.activity-rail button:has([data-icon=key])').count(), 1);
    },
  );
  await check(
    'Three timeline lanes preserve named keys, snapshots and complete recorded frames',
    async () => {
      await typeValue('Current time', 0);
      await page
        .getByRole('button', { name: 'Capture scene pose at playhead (K)', exact: true })
        .click();
      await typeValue('Current time', 2);
      await typeValue('Position X', 3.25);
      await page
        .getByRole('button', { name: 'Capture scene pose at playhead (K)', exact: true })
        .click();
      await typeValue('Current time', 1);
      assert.equal(Number(await page.getByLabel('Position X', { exact: true }).inputValue()), 3.25);
      await page.locator('.keyframe').first().dblclick();
      assert.equal(Number(await page.getByLabel('Position X', { exact: true }).inputValue()), 1.25);
      await typeValue('Position X', 2.25);
      await page.getByRole('button', { name: 'Capture snapshot', exact: true }).click();
      await typeValue('Position X', 4.25);
      await page.locator('.snapshot-marker').first().dblclick();
      assert.equal(Number(await page.getByLabel('Position X', { exact: true }).inputValue()), 2.25);
      await page.getByRole('button', { name: 'Record scene frames', exact: true }).click();
      await page.waitForTimeout(80);
      await typeValue('Position X', 3.25);
      await page.waitForTimeout(120);
      await page.getByRole('button', { name: 'Stop recording', exact: true }).click();
      await page.getByRole('button', { name: 'Go to start', exact: true }).click();
      assert.equal(Number(await page.getByLabel('Position X', { exact: true }).inputValue()), 2.25);
      await page.getByRole('button', { name: 'Go to end', exact: true }).click();
      assert.equal(Number(await page.getByLabel('Position X', { exact: true }).inputValue()), 3.25);
      await page.getByRole('button', { name: 'Play (Space)', exact: true }).click();
      await page.waitForTimeout(120);
      await page.getByRole('button', { name: 'Pause (Space)', exact: true }).click();
      assert.ok(Number(await page.getByLabel('Current time', { exact: true }).inputValue()) > 0);
      assert.equal(await page.locator('[data-lane]').count(), 3);
      assert.equal(await page.locator('.keyframe').count(), 2);
      assert.equal(await page.locator('.snapshot-marker').count(), 1);
      assert.ok((await page.locator('.take-frame-tick').count()) > 1);
      await capture('timeline-revised');
    },
  );
  await check('Custom file browser saves and reopens the complete scene and timeline', async () => {
    const filename = 'revised-roundtrip.mojive.json';
    await action('file.save');
    await page.getByRole('dialog', { name: 'Save scene as', exact: true }).waitFor();
    await page.getByRole('textbox', { name: 'Location', exact: true }).fill(out);
    await page.getByRole('textbox', { name: 'Location', exact: true }).press('Enter');
    await page.waitForTimeout(200);
    await page.getByRole('textbox', { name: 'File name', exact: true }).fill(filename);
    await capture('file-browser-save');
    await page.getByRole('button', { name: 'Save', exact: true }).click();
    await page.waitForFunction(
      () =>
        !document.querySelector('.file-dialog') ||
        document.querySelector('[role=alertdialog]') ||
        document.querySelector('.file-dialog-error'),
    );
    if (
      await page.getByRole('alertdialog', { name: 'Replace existing file?', exact: true }).count()
    )
      await page.getByRole('button', { name: 'Replace', exact: true }).click();
    await page.locator('.file-dialog').waitFor({ state: 'detached' });
    const saved = JSON.parse(await readFile(path.join(out, filename), 'utf8'));
    assert.deepEqual(saved.document.nodes.filter((node) => primitiveNames.includes(node.name))
      .map((node) => node.kind), ['cone', 'ellipsoid']);
    await action('file.openScene');
    await page.getByRole('textbox', { name: 'Location', exact: true }).fill(out);
    await page.getByRole('textbox', { name: 'Location', exact: true }).press('Enter');
    await page.getByRole('option', { name: filename, exact: true }).dblclick();
    await page.locator('.file-dialog').waitFor({ state: 'detached' });
    assert.match(await page.locator('.document-title').innerText(), /Atelier/);
    assert.equal(await page.locator('.key-lane').first().getByRole('button').count(), 2);
    assert.equal(await page.locator('.snapshot-marker').count(), 1);
    assert.ok((await page.locator('.take-frame-tick').count()) > 1);
    for (const [index, name] of primitiveNames.entries()) {
      await selectNode(name);
      assert.equal(Number(await page.getByLabel('Position X', { exact: true }).inputValue()),
        index === 0 ? -1.2 : 1.2);
    }
    await page.locator('.panel-search input').fill('');
    await action('view.focusAll');
    await page.waitForTimeout(500);
    await capture('primitive-roundtrip');
  });
  await check(
    'Output search, severity filters, sorting and clear are actual controls',
    async () => {
      await page
        .locator('.bottom-tabs')
        .getByRole('button', { name: /^Output/ })
        .click();
      await page.getByRole('button', { name: 'Search output', exact: true }).click();
      await page.locator('.output-search input').fill('Loaded');
      assert.ok((await page.locator('.output-message').count()) > 0);
      assert.ok(
        (await page.locator('.output-message').allTextContents()).every((s) =>
          s.includes('Loaded'),
        ),
      );
      await page.getByLabel('Sort output', { exact: true }).click();
      await capture('output-revised');
      await page.locator('.output-search input').fill('');
    },
  );
  await check(
    'External URDF and local project resources load through the sandbox bridge',
    async () => {
      const directory = path.resolve(root, '../assets');
      await action('file.openUrdf');
      await page.getByRole('textbox', { name: 'Location', exact: true }).fill(directory);
      await page.getByRole('textbox', { name: 'Location', exact: true }).press('Enter');
      await page.getByRole('option', { name: 'test_scene.urdf', exact: true }).dblclick();
      await page.waitForFunction(() =>
        document.querySelector('.document-title')?.textContent?.includes('test_scene'),
      );
      await page.locator('.busy-overlay').waitFor({ state: 'detached' });
      // The document swaps before the load transaction releases the native menu.
      if (process.platform === 'darwin') {
        const deadline = Date.now() + 10000;
        while (!(await app.evaluate(({ Menu }) => Menu.getApplicationMenu()?.getMenuItemById('file.openProject')?.enabled))) {
          assert.ok(Date.now() < deadline, 'File menu must become available after loading');
          await page.waitForTimeout(20);
        }
      }
      await action('file.openProject');
      await page.getByRole('textbox', { name: 'Location', exact: true }).fill(directory);
      await page.getByRole('button', { name: 'Select folder', exact: true }).click();
      await page.getByRole('dialog').waitFor({ state: 'hidden' });
      await page.getByRole('button', { name: 'Project files', exact: true }).click();
      assert.equal(
        await page.locator('.left-content .group-eyebrow').getAttribute('data-tooltip'),
        directory,
      );
      assert.ok((await page.locator('.file-row').count()) > 5);
      await capture('project-revised');
    },
  );
  await check(
    'Light/camera additions and narrow/high-scale layouts retain usable controls',
    async () => {
      await action('add.light');
      await action('add.camera');
      assert.ok(await page.getByLabel('Vertical field of view', { exact: true }).count());
      await page
        .locator('.camera-properties')
        .getByRole('button', { name: 'Capture image', exact: true })
        .click();
      await page.getByRole('dialog', { name: 'Save camera image', exact: true }).waitFor();
      await page.getByRole('textbox', { name: 'Location', exact: true }).fill(out);
      await page.getByRole('textbox', { name: 'Location', exact: true }).press('Enter');
      await page.waitForFunction(
        () => document.querySelector('.file-dialog-list')?.getAttribute('aria-busy') === 'false',
      );
      await page.getByRole('textbox', { name: 'File name', exact: true }).fill('camera-export.png');
      await page.getByRole('button', { name: 'Save', exact: true }).click();
      await page.waitForFunction(
        () =>
          !document.querySelector('.file-dialog') ||
          document.querySelector('[role=alertdialog]') ||
          document.querySelector('.file-dialog-error'),
      );
      if (
        await page.getByRole('alertdialog', { name: 'Replace existing file?', exact: true }).count()
      )
        await page.getByRole('button', { name: 'Replace', exact: true }).click();
      await page.locator('.file-dialog').waitFor({ state: 'detached' });
      const png = await readFile(path.join(out, 'camera-export.png'));
      assert.deepEqual([...png.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
      assert.equal(png.readUInt32BE(16), 1920);
      assert.equal(png.readUInt32BE(20), 1080);
      await app.evaluate(({ BrowserWindow }) =>
        BrowserWindow.getAllWindows()[0].setContentSize(1120, 800),
      );
      await wait();
      await capture('workspace-revised-compact');
      const fields = await page
        .locator('.inspector .axis-field input')
        .evaluateAll((inputs) => inputs.map((i) => i.getBoundingClientRect().width));
      assert.ok(fields.every((w) => w > 25));
      await app.evaluate(({ BrowserWindow }) =>
        BrowserWindow.getAllWindows()[0].setContentSize(1540, 980),
      );
      await action('app.settings');
      await typeValue('UI scale', 1.25);
      await page
        .locator('dialog[open]')
        .getByRole('button', { name: 'Close', exact: true })
        .click();
      await wait();
      const shellBounds = await page.locator('.app-shell').evaluate((shell) => {
        const bounds = shell.getBoundingClientRect();
        return {
          width: bounds.width,
          height: bounds.height,
          windowWidth: innerWidth,
          windowHeight: innerHeight,
        };
      });
      assert.ok(Math.abs(shellBounds.width - shellBounds.windowWidth) < 1);
      assert.ok(Math.abs(shellBounds.height - shellBounds.windowHeight) < 1);
      await capture('workspace-revised-125');
      await action('app.settings');
      await typeValue('UI scale', 1);
      await page
        .locator('dialog[open]')
        .getByRole('button', { name: 'Close', exact: true })
        .click();
    },
  );
  await check(
    'Camera preview, projection choices and viewport return work in the native Electron window',
    async () => {
      await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].showInactive());
      const inspector = page.locator('.inspector');
      await inspector.getByRole('switch', { name: 'Preview', exact: true }).click();
      await page.locator('.camera-preview-image').waitFor();
      await page.waitForFunction(
        () => Number(document.querySelector('.camera-preview-image')?.dataset.renderCount) > 0,
      );
      const preview = await page.locator('.camera-preview').evaluate((panel) => {
        const image = panel.querySelector('canvas');
        return {
          height: panel.getBoundingClientRect().height,
          image: image.getBoundingClientRect().height,
          bytes: image.toDataURL().length,
        };
      });
      assert.ok(preview.bytes > 10000);
      assert.ok(Math.abs(preview.height - preview.image - 52) < 1);
      const projection = inspector.getByRole('radiogroup', {
        name: 'Camera projection',
        exact: true,
      });
      assert.equal(await projection.getByRole('radio').count(), 2);
      await projection.getByRole('radio', { name: 'Orthographic', exact: true }).click();
      await inspector.getByLabel('Orthographic height', { exact: true }).waitFor();
      await projection.getByRole('radio', { name: 'Perspective', exact: true }).click();
      await inspector.getByRole('button', { name: 'View Camera', exact: true }).click();
      await page.locator('.scene-camera-return button').waitFor();
      await page.waitForTimeout(350);
      await capture('electron-camera-preview');
      await page.locator('.scene-camera-return button').click();
      await page.locator('.scene-camera-return').waitFor({ state: 'detached' });
      await page.waitForTimeout(350);
      await inspector.getByRole('switch', { name: 'Preview', exact: true }).click();
    },
  );
  await check(
    'Ground selection preferences and action notices preserve real undo and redo',
    async () => {
      await action('add.ground');
      const preference = page.getByRole('switch', { name: 'Select in viewport', exact: true });
      assert.equal(await preference.getAttribute('aria-checked'), 'false');
      await preference.click();
      assert.equal(await preference.getAttribute('aria-checked'), 'true');
      await page
        .locator('.operation-notice')
        .getByRole('button', { name: 'Undo', exact: true })
        .click();
      assert.equal(await preference.getAttribute('aria-checked'), 'false');
      await page
        .locator('.operation-notice')
        .getByRole('button', { name: 'Redo', exact: true })
        .click();
      assert.equal(await preference.getAttribute('aria-checked'), 'true');
      await preference.click();
      const place = await page.locator('.operation-notice').evaluate((node) => {
        const box = node.getBoundingClientRect(),
          viewport = node.parentElement.getBoundingClientRect();
        return { left: box.left - viewport.left, bottom: viewport.bottom - box.bottom };
      });
      assert.equal(place.left, 12);
      assert.equal(place.bottom, 12);
      const header = page.locator('.inspector .section-heading:has(.section-actions)');
      const reset = await header.locator('.section-actions > button').boundingBox(),
        disclosure = await header.locator('.section-disclosure').boundingBox();
      assert.equal(reset.width, 28);
      assert.equal(reset.height, 28);
      assert.equal(disclosure.x - reset.x - reset.width, 4);
      await capture('electron-ground-notice');
    },
  );
  await check(
    'Settings, clear buttons and navigation rail use their actual production bounds',
    async () => {
      await action('app.settings');
      const dialog = page.locator('.settings-dialog'),
        general = await dialog.boundingBox();
      const row = await page.locator('.settings-search').evaluate((node) => {
        const bounds = node.getBoundingClientRect(),
          parent = node.parentElement,
          outer = parent.getBoundingClientRect(),
          style = getComputedStyle(parent),
          scale = outer.width / parent.offsetWidth;
        return {
          left: bounds.left,
          right: bounds.right,
          contentLeft: outer.left + (parent.clientLeft + parseFloat(style.paddingLeft)) * scale,
          contentRight:
            outer.left +
            (parent.clientLeft + parent.clientWidth - parseFloat(style.paddingRight)) * scale,
          navigationLeft: node
            .closest('.settings-surface')
            .querySelector('.settings-navigation')
            .getBoundingClientRect().left,
          contentRightEdge: node
            .closest('.settings-surface')
            .querySelector('.settings-page')
            .getBoundingClientRect().right,
        };
      });
      assert.ok(Math.abs(row.left - row.contentLeft) < 0.5, JSON.stringify(row));
      assert.ok(Math.abs(row.right - row.contentRight) < 0.5, JSON.stringify(row));
      assert.equal(row.left, row.navigationLeft);
      assert.ok(row.right <= row.contentRightEdge + 1);
      await page.getByRole('textbox', { name: 'Search all settings', exact: true }).fill('123');
      const clear = await page.locator('.settings-search').evaluate((node) => {
        const a = node.getBoundingClientRect(),
          button = node.querySelector('button'),
          b = button.getBoundingClientRect();
        return {
          top: b.top - a.top,
          bottom: a.bottom - b.bottom,
          right: a.right - b.right,
          width: b.width,
          height: b.height,
          radius: getComputedStyle(button).borderRadius,
        };
      });
      const { right: endGap, ...clearFrame } = clear;
      assert.ok(Math.abs(endGap - 4.24723) < 0.03, JSON.stringify(clear));
      assert.deepEqual(clearFrame, {
        top: 4,
        bottom: 4,
        width: 20,
        height: 20,
        radius: '50%',
      });
      await page.locator('.settings-search button').click();
      await page
        .locator('.settings-navigation')
        .getByRole('button', { name: 'Rendering', exact: true })
        .click();
      assert.deepEqual(await dialog.boundingBox(), general);
      await capture('electron-settings-aligned');
      await dialog.getByRole('button', { name: 'Close', exact: true }).click();
      const rail = await page.locator('.activity-rail').evaluate((node) => {
        const r = node.getBoundingClientRect(),
          buttons = [...node.querySelectorAll(':scope > button')],
          first = buttons[0].getBoundingClientRect(),
          last = buttons.at(-1).getBoundingClientRect();
        return {
          top: first.top - r.top,
          bottom: r.bottom - last.bottom,
          left: first.left - r.left,
          right: r.right - first.right,
        };
      });
      for (const value of Object.values(rail))
        assert.ok(Math.abs(value - 7) < 1, JSON.stringify(rail));
      assert.equal(
        await page.locator('.viewport-navigation > button:has([data-icon=fit])').count(),
        1,
      );
    },
  );
  await check('No renderer exceptions or shader errors', async () => assert.deepEqual(errors, []));
  await writeFile(
    path.join(out, 'workspace-acceptance.json'),
    JSON.stringify({ passed, errors }, null, 2),
  );
} catch (error) {
  if (page) await capture('workspace-revised-failure').catch(() => {});
  await writeFile(
    path.join(out, 'workspace-acceptance.json'),
    JSON.stringify({ passed, errors, failure: String(error) }, null, 2),
  );
  throw error;
} finally {
  await app
    .evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().forEach((w) => w.destroy()))
    .catch(() => {});
  await app.close();
}
