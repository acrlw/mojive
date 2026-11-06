import { openEditableJointScene } from './helpers/editable-scene.mjs';
import { installPaintStyle, settlePaint } from './helpers/surface-style.mjs';
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir, mkdtemp, realpath, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
const require = createRequire(import.meta.url);
const { FileAccess } = require('../main/file-access.cjs');
const { Projects } = require('../main/resource-paths.cjs');
let server, browser, base;
before(async () => {
  server = await createServer({
    root: process.cwd(),
    server: { host: '127.0.0.1', port: 0, strictPort: false, watch: null },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
async function workspace(appearance, uiScale = 1, language = 'en', width = 1700, uiFontSize = 14) {
  const page = await browser.newPage({ viewport: { width, height: 1050 } });
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
  page.setDefaultTimeout(5000);
  await page.addInitScript(
    (settings) => localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings)),
    { appearance, uiScale, language, uiFontSize },
  );
  await page.goto(base);
  await openEditableJointScene(page);
  await page.evaluate(() => document.fonts.ready);
  return page;
}
async function assertClear(field, scale) {
  const box = await field.evaluate((node) => {
    const a = node.getBoundingClientRect(),
      button = node.querySelector(':scope > button'),
      b = button.getBoundingClientRect();
    return {
      top: b.top - a.top,
      bottom: a.bottom - b.bottom,
      right: a.right - b.right,
      width: b.width,
      height: b.height,
      radius: window.paintStyle(button).borderRadius,
    };
  });
  for (const edge of ['top', 'bottom'])
    assert.ok(Math.abs(box[edge] - 4 * scale) < 0.1, JSON.stringify(box));
  // Native feasibility's H28/s0.382 mean-clearance center is 14.24723px.
  assert.ok(Math.abs(box.right - 4.24723 * scale) < 0.1, JSON.stringify(box));
  assert.equal(box.width, box.height);
  assert.equal(box.radius, '50%');
  await field.locator('button').click();
  assert.equal(await field.locator('input').inputValue(), '');
  assert.ok(await field.locator('input').evaluate((input) => document.activeElement === input));
}
test('production settings fill aligned content columns and keep their bounds through category/search changes', async () => {
  for (const [theme, scale, language, width] of [
    ['dark', 1, 'zh', 1700],
    ['light', 1.25, 'en', 1700],
    ['dark', 2, 'en', 1100],
  ]) {
    const page = await workspace(theme, scale, language, width);
    await page.locator('.title-actions button:has([data-icon=settings])').click();
    const dialog = page.locator('.settings-dialog'),
      initial = await dialog.boundingBox();
    const layout = await page.locator('.settings-page').evaluate((node) => {
      const search = document.querySelector('.settings-search').getBoundingClientRect(),
        navigation = document.querySelector('.settings-navigation').getBoundingClientRect(),
        heading = node.querySelector('.settings-page-heading').getBoundingClientRect();
      return {
        search: search.toJSON(),
        heading: heading.toJSON(),
        navigation: navigation.toJSON(),
        overflow: node.scrollWidth - node.clientWidth,
        rows: [...node.querySelectorAll('.settings-row')].map((row) => ({
          row: row.getBoundingClientRect().toJSON(),
          field: row.lastElementChild.getBoundingClientRect().toJSON(),
          labelTop: row.firstElementChild.getBoundingClientRect().top,
        })),
      };
    });
    assert.equal(layout.overflow, 0);
    assert.ok(Math.abs(layout.search.left - layout.navigation.left) < 0.1);
    assert.ok(Math.abs(layout.search.right - layout.heading.right) < 0.1);
    assert.ok(
      layout.search.bottom < layout.navigation.top && layout.search.bottom < layout.heading.top,
    );
    const fields = layout.rows.filter((r) => r.field.left > r.row.left + 1);
    assert.ok(fields.length >= 5);
    assert.ok(fields.every((r) => Math.abs(r.field.left - fields[0].field.left) < 0.1));
    // All interactive controls reach the same right edge; readonly font names retain intrinsic width.
    for (const row of layout.rows) assert.ok(Math.abs(row.field.right - row.row.right) < 0.1);
    await dialog.screenshot({
      path: `../output/electron/verification/settings-aligned-${theme}-${language}-${scale}.png`,
    });
    for (const button of await page.locator('.settings-navigation > button').all()) {
      await button.click();
      assert.deepEqual(await dialog.boundingBox(), initial);
    }
    await page.locator('.settings-search input').fill('123');
    await page.locator('.settings-search button').hover();
    assert.deepEqual(await dialog.boundingBox(), initial);
    await dialog.screenshot({
      path: `../output/electron/verification/settings-clear-${theme}-${scale}.png`,
    });
    await assertClear(page.locator('.settings-search'), scale);
    await page.keyboard.press('Escape');
    await page.locator('.tree-row[data-tooltip="05_multi_slide_x"]').click();
    const tabs = await page.locator('.inspector-tabs').evaluate((node) => ({
      border: window.paintStyle(node).borderBottomWidth,
      overflow: node.scrollWidth - node.clientWidth,
      buttons: [...node.children].map((button) => ({
        title: button.dataset.tooltip,
        label: button.textContent,
        overflow: window.paintStyle(button).overflow,
        textOverflow: window.paintStyle(button).textOverflow,
        border: window.paintStyle(button).borderBottomWidth,
      })),
    }));
    assert.equal(tabs.border, '0px');
    assert.equal(tabs.overflow, 0);
    for (const tab of tabs.buttons) {
      assert.equal(tab.border, '0px');
      assert.equal(tab.title, tab.label);
      assert.equal(tab.overflow, 'hidden');
      assert.equal(tab.textOverflow, 'ellipsis');
    }
    await page.locator('.inspector').screenshot({
      path: `../output/electron/verification/inspector-tabs-${language}-${scale}.png`,
    });
    await page.close();
  }
});
test('all production search consumers share an optically centered clear circle at fractional scale', async () => {
  const scale = 1.25,
    page = await workspace('light', scale),
    project = await realpath(await mkdtemp(path.join(os.tmpdir(), 'mojive-search-layout-'))),
    files = new FileAccess([{ label: 'Home', path: project }]),
    projects = new Projects();
  const metrics = [];
  const checkSearch = async (field, panel) => {
    await field.locator('input').fill('test');
    await settlePaint(page);
    const shape = await field.evaluate((node) => {
      const style = window.paintStyle(node),
        input = node.querySelector('input');
      const text = getComputedStyle(input),
        icon = getComputedStyle(node.querySelector(':scope > .icon'));
      return {
        height: style.height,
        padding: style.padding,
        radius: style.borderRadius,
        font: text.font,
        color: text.color,
        placeholder: getComputedStyle(input, '::placeholder').color,
        iconWidth: icon.width,
        iconHeight: icon.height,
        iconColor: icon.color,
        outline: style.outlineStyle,
        outlineOffset: style.outlineOffset,
        kind: node.dataset.smoothKind,
        smoothing: node.dataset.smoothAmount,
      };
    });
    assert.equal(shape.iconWidth, '16px');
    assert.equal(shape.iconHeight, '16px');
    assert.equal(shape.height, '28px');
    assert.equal(shape.kind, 'capsule');
    if (metrics.length) assert.deepEqual(shape, metrics[0].shape, panel);
    metrics.push({ panel, shape });
    await field.screenshot({
      path: `../output/electron/verification/search-${panel.replaceAll(' ', '-').toLowerCase()}.png`,
    });
    await assertClear(field, scale);
  };
  try {
    await writeFile(
      path.join(project, 'test.urdf'),
      '<robot name="Search fixture"><link name="base"/></robot>',
    );
    await page.exposeFunction('browseSearchProject', (directory, hidden) =>
      files.browse(directory, hidden),
    );
    await page.exposeFunction('openSearchProject', async (directory) => {
      const root = await realpath(directory);
      assert.equal(root, project);
      await projects.register(root);
      return { path: root, name: path.basename(root), files: await projects.list(root) };
    });
    await page.exposeFunction('searchProjectFiles', async (directory, query) => {
      await projects.authorized(directory);
      return projects.search(directory, query);
    });
    await page.evaluate(() => {
      window.mojive = {
        browseFiles: window.browseSearchProject,
        openProjectPath: window.openSearchProject,
        searchProject: window.searchProjectFiles,
        hasNativeMenu: false,
        platform: 'linux',
        onMenuAction: () => () => {},
        updateMenuState: async () => {},
      };
    });
    const searches = [
      ['Scene', 'Search', '.panel-search'],
      ['Project files', null, '.panel-search'],
      ['Joints', 'Search joints', '.joint-panel-search'],
      ['Cameras', 'Search cameras', '.camera-panel-search'],
      ['Output', 'Search output', '.output-search'],
    ];
    for (const [panel, trigger, selector] of searches) {
      await page
        .locator('.activity-rail')
        .getByRole('button', { name: panel, exact: true })
        .click();
      if (panel === 'Project files') {
        await page.getByRole('button', { name: 'Open project folder', exact: true }).click();
        const dialog = page.getByRole('dialog', { name: 'Open project folder', exact: true });
        await dialog.waitFor();
        await dialog.locator('.file-dialog-list[aria-busy="false"]').waitFor();
        await checkSearch(dialog.locator('.file-dialog-filter'), 'File dialog');
        await dialog.getByRole('button', { name: 'Select folder', exact: true }).click();
        await dialog.waitFor({ state: 'hidden' });
        assert.equal(
          await page.locator('.left-content .group-eyebrow').getAttribute('data-tooltip'),
          project,
        );
      }
      if (trigger)
        await page
          .locator(panel === 'Scene' ? '.left-panel' : 'body')
          .getByRole('button', { name: trigger, exact: true })
          .click();
      const field = page.locator(selector);
      await checkSearch(field, panel);
    }
    await page.getByRole('button', { name: 'Settings', exact: true }).first().click();
    await checkSearch(page.locator('.settings-search'), 'Settings');
    await writeFile(
      '../output/electron/verification/search-consumers.json',
      JSON.stringify(metrics, null, 2),
    );
  } finally {
    await page.close();
    await rm(project, { recursive: true, force: true });
  }
});

test('settings share the measured longest label across sections with an exact gap and a flexible control column', async () => {
  for (const [theme, scale, language, width, fontSize] of [
    ['dark', 1, 'zh', 1700, 14],
    ['light', 1.25, 'en', 1700, 14],
    ['light', 1.25, 'zh', 1700, 20],
    ['dark', 1.25, 'en', 1700, 20],
    ['dark', 1.25, 'en', 500, 14],
  ]) {
    const page = await workspace(theme, scale, language, width, fontSize),
      errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    try {
      await page.locator('.title-actions button:has([data-icon=settings])').click();
      const dialog = page.locator('.settings-dialog'),
        initial = await dialog.boundingBox();
      for (let category = 0; category < 4; category++) {
        await page.locator('.settings-navigation > button').nth(category).click();
        const metrics = await page.locator('.settings-page').evaluate((node) => {
          const rect = (node) => node.getBoundingClientRect().toJSON(),
            root = node.querySelector('.settings-fields');
          const rows = [...root.querySelectorAll('.settings-row')].map((row) => {
            const label = row.firstElementChild,
              field = row.lastElementChild,
              style = window.paintStyle(label);
            const canvas = document.createElement('canvas'),
              ctx = canvas.getContext('2d');
            ctx.font = style.font;
            return {
              label: rect(label),
              field: rect(field),
              row: rect(row),
              caption: label.textContent,
              measured: ctx.measureText(label.textContent).width,
              toggle: field.matches('.custom-switch'),
              stacked: window.paintStyle(row).gridTemplateColumns.split(' ').length === 1,
            };
          });
          return {
            page: rect(node),
            navigationWidth: document.querySelector('.settings-navigation').getBoundingClientRect()
              .width,
            navigationFont: parseFloat(
              window.paintStyle(document.querySelector('.settings-navigation')).fontSize,
            ),
            fields: rect(root),
            search: rect(document.querySelector('.settings-search')),
            rows,
            overflow: node.scrollWidth - node.clientWidth,
            segments: [...root.querySelectorAll('.settings-segment .button')].map((button) => ({
              width: button.clientWidth,
              overflow: button.scrollWidth - button.clientWidth,
            })),
            navigation: [...document.querySelectorAll('.settings-navigation .button')].map(
              (button) => ({
                width: button.clientWidth,
                overflow: button.scrollWidth - button.clientWidth,
              }),
            ),
          };
        });
        const context = `${language}/${theme}/${scale}/${fontSize}/${width}/${category}`;
        assert.equal(metrics.overflow, 0, context);
        if (width >= 1000)
          assert.ok(
            Math.abs(
              metrics.navigationWidth / scale - Math.max(120, metrics.navigationFont * 8.5),
            ) < 0.1,
            `${context}: stable language-independent navigation width`,
          );
        assert.deepEqual(await dialog.boundingBox(), initial, context);
        assert.ok(Math.abs(metrics.search.right - metrics.fields.right) < 0.1, context);
        const inline = metrics.rows.filter((row) => !row.stacked),
          normal = inline.filter((row) => !row.toggle);
        if (width >= 1000) {
          const expected =
            Math.min(
              Math.max(...metrics.rows.map((row) => row.measured)),
              metrics.fields.width / scale - 140,
            ) * scale;
          for (const row of inline)
            assert.ok(
              Math.abs(row.label.width - expected) < 1,
              `${context}: shared intrinsic ${JSON.stringify(row)} expected ${expected}`,
            );
          for (const row of normal)
            assert.ok(
              Math.abs(row.field.left - row.label.right - 12 * scale) < 0.1,
              `${context}: 12pt gap`,
            );
        }
        for (const row of metrics.rows) {
          assert.ok(row.field.right <= metrics.fields.right + 0.1, `${context}: right edge`);
          if (!row.toggle)
            assert.ok(
              Math.abs(row.field.right - metrics.fields.right) < 0.1,
              `${context}: shared right edge`,
            );
        }
        for (const row of normal)
          assert.ok(
            Math.abs(row.field.left - normal[0].field.left) < 0.1,
            `${context}: shared left edge`,
          );
        for (const segment of metrics.segments)
          assert.equal(segment.overflow, 0, `${context}: appearance text must fit`);
        for (const button of metrics.navigation)
          assert.equal(button.overflow, 0, `${context}: category text must fit`);
        await dialog.screenshot({
          path: `../output/electron/verification/settings-intrinsic-${language}-${theme}-${scale}-${fontSize}-${width}-${category}.png`,
        });
      }
      await page.locator('.settings-search input').fill('缩放速度');
      const filtered = await page.locator('.settings-row').evaluate((row) => {
        const label = row.firstElementChild,
          field = row.lastElementChild,
          style = window.paintStyle(label);
        const ctx = document.createElement('canvas').getContext('2d');
        ctx.font = style.font;
        return {
          label: label.getBoundingClientRect().toJSON(),
          field: field.getBoundingClientRect().toJSON(),
          measured: ctx.measureText(label.textContent).width,
        };
      });
      if (width >= 1000) {
        assert.ok(
          Math.abs(filtered.label.width - filtered.measured * scale) < 1,
          'Search recomputes the shared intrinsic label column',
        );
        assert.ok(Math.abs(filtered.field.left - filtered.label.right - 12 * scale) < 0.1);
      }
      assert.deepEqual(await dialog.boundingBox(), initial);
      assert.deepEqual(errors, []);
    } finally {
      await page.close();
    }
  }
});

test('global settings search spans navigation and content, finds other categories and clears to the previous page', async () => {
  for (const [language, appearance, scale, width] of [
    ['en', 'light', 1, 1700],
    ['zh', 'dark', 1.25, 1700],
    ['en', 'dark', 2, 1100],
    ['zh', 'light', 1.25, 500],
  ]) {
    const page = await workspace(appearance, scale, language, width),
      errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    try {
      await page.locator('.title-actions button:has([data-icon=settings])').click();
      const dialog = page.locator('.settings-dialog'),
        search = page.locator('.settings-search input'),
        navigation = page.locator('.settings-navigation');
      const initial = await dialog.boundingBox(),
        searchBox = await page.locator('.settings-search').boundingBox();
      assert.equal(await dialog.getByRole('search').count(), 1);
      assert.equal(
        await dialog
          .locator('.settings-surface > .settings-global-search .settings-search')
          .count(),
        1,
      );
      assert.equal(await dialog.locator('.settings-page .settings-search').count(), 0);
      const categories = await navigation.locator('button').allTextContents();
      await navigation.locator('button').last().click();
      const navBox = await navigation.boundingBox();
      await search.fill(language === 'en' ? 'camera' : '相机');
      await page.mouse.move(searchBox.x + 20, searchBox.y + searchBox.height / 2);
      const provenance = await page
        .locator('[data-settings-category]')
        .evaluateAll((groups) => [
          ...new Set(groups.map((group) => group.dataset.settingsCategory)),
        ]);
      assert.deepEqual(provenance, ['Camera', 'Rendering']);
      const headings = await page.locator('.settings-group > h3').allTextContents();
      assert.ok(
        headings.some((text) => text.startsWith(language === 'en' ? 'Camera ›' : '相机 ›')),
      );
      assert.ok(
        headings.some((text) => text.startsWith(language === 'en' ? 'Rendering ›' : '渲染 ›')),
      );
      assert.equal(
        await page
          .getByRole('textbox', {
            name: language === 'en' ? 'Camera FOV' : '相机视场角',
            exact: true,
          })
          .count(),
        1,
      );
      assert.deepEqual(await navigation.locator('button').allTextContents(), categories);
      assert.deepEqual(await navigation.boundingBox(), navBox);
      assert.deepEqual(await dialog.boundingBox(), initial);
      await page.locator('.settings-page').evaluate((node) => (node.scrollTop = node.scrollHeight));
      assert.deepEqual(await page.locator('.settings-search').boundingBox(), searchBox);
      await dialog.screenshot({
        path: `../output/electron/verification/settings-global-results-${language}-${appearance}-${scale}-${width}.png`,
      });
      await search.fill(language === 'en' ? 'Font family' : '字体家族');
      assert.equal(await page.locator('.settings-font-group').count(), 2);
      assert.equal(await page.locator('.settings-font-group .settings-row').count(), 2);
      assert.deepEqual(
        await page
          .locator('[data-settings-category]')
          .evaluateAll((groups) => groups.map((group) => group.dataset.settingsCategory)),
        ['General'],
      );
      assert.equal(
        await page
          .getByRole('combobox', { name: language === 'en' ? 'Language' : '语言', exact: true })
          .count(),
        0,
      );
      await page.locator('.settings-font-group').first().getByRole('combobox').click();
      await page.getByRole('option', { name: 'MiSans', exact: true }).click();
      const beforeReset = await page.evaluate(() =>
        JSON.parse(localStorage.getItem('mojive.electron.settings.v1')),
      );
      assert.equal(beforeReset.numericFont, 'body');
      await page.locator('.settings-page-heading .button').click();
      const afterReset = await page.evaluate(() =>
        JSON.parse(localStorage.getItem('mojive.electron.settings.v1')),
      );
      assert.deepEqual(
        afterReset,
        { ...beforeReset, numericFont: 'mono', outputFont: 'mono' },
        'Reset applies only the matching global result fields',
      );
      await search.fill('no-such-setting-8472');
      assert.equal(await page.locator('.settings-empty .empty').count(), 1);
      assert.equal(await page.locator('.settings-page-heading .button').isDisabled(), true);
      assert.deepEqual(await navigation.boundingBox(), navBox);
      await dialog.screenshot({
        path: `../output/electron/verification/settings-global-empty-${language}-${appearance}-${scale}-${width}.png`,
      });
      await page
        .locator('.settings-empty')
        .getByRole('button', { name: language === 'en' ? 'Clear search' : '清除搜索', exact: true })
        .click();
      assert.equal(await search.inputValue(), '');
      assert.equal(await search.evaluate((node) => document.activeElement === node), true);
      assert.equal(await navigation.locator('button').last().getAttribute('aria-current'), 'page');
      assert.equal(
        await page.locator('.settings-page-heading h2').textContent(),
        language === 'en' ? 'Rendering' : '渲染',
      );
      await search.fill('orbit');
      await navigation.locator('button').first().click();
      assert.equal(await search.inputValue(), '');
      assert.equal(await navigation.locator('button').first().getAttribute('aria-current'), 'page');
      assert.deepEqual(await dialog.boundingBox(), initial);
      assert.equal(
        await page
          .locator('.settings-page')
          .evaluate((node) => node.scrollWidth - node.clientWidth),
        0,
      );
      await dialog.screenshot({
        path: `../output/electron/verification/settings-global-general-${language}-${appearance}-${scale}-${width}.png`,
      });
      assert.deepEqual(errors, []);
    } finally {
      await page.close();
    }
  }
});
