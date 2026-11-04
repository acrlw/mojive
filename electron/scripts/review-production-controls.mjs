import { createServer as createTcpServer } from 'node:net';
import { createServer } from 'vite';
import { _electron } from 'playwright';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';
const root = fileURLToPath(new URL('../', import.meta.url)),
  out = path.join(root, '../output/electron/verification');
const socket = createTcpServer();
await new Promise((resolve) => socket.listen(0, '127.0.0.1', resolve));
const port = socket.address().port;
await new Promise((resolve) => socket.close(resolve));
const server = await createServer({
    root,
    server: { host: '127.0.0.1', port, strictPort: true },
    logLevel: 'error',
  }),
  profile = await mkdtemp(path.join(out, 'production-controls-profile-'));
let app;
const errors = [],
  geometry = [],
  buttons = [];
try {
  await server.listen();
  app = await _electron.launch({
    args: [path.join(root, 'main/main.cjs'), '--hidden', '--user-data-dir=' + profile],
    env: { ...process.env, MOJIVE_DEV_URL: `http://127.0.0.1:${port}` },
  });
  const page = await app.firstWindow();
  page.setDefaultTimeout(15000);
  page.on('pageerror', (error) => errors.push(String(error)));
  await page.waitForSelector('[role=treeitem]');
  await page.evaluate(() => document.fonts.ready);
  await app.evaluate(({ Menu, BrowserWindow }) => {
    const action = Menu.getApplicationMenu().getMenuItemById('app.settings');
    action.click(action, BrowserWindow.getAllWindows()[0], {});
  });
  await page.getByRole('dialog', { name: '设置' }).waitFor();
  const initial = await page.getByRole('dialog', { name: '设置' }).boundingBox();
  for (const theme of ['dark', 'light']) {
    await page.getByRole('button', { name: '常规', exact: true }).click();
    await page
      .getByRole('button', { name: theme === 'dark' ? '深色' : '浅色', exact: true })
      .click();
    await page.getByRole('button', { name: '渲染', exact: true }).click();
    const current = await page.getByRole('dialog', { name: '设置' }).boundingBox();
    assert.deepEqual(current, initial);
    const toggle = page.getByRole('switch').first();
    geometry.push(
      await toggle.evaluate((node) => {
        const track = node.querySelector('.custom-switch'),
          thumb = track.querySelector('span');
        return {
          theme: document.documentElement.dataset.theme,
          track: getComputedStyle(track).cornerShape,
          thumb: getComputedStyle(thumb).cornerShape,
          thumbBounds: {
            width: thumb.getBoundingClientRect().width,
            height: thumb.getBoundingClientRect().height,
          },
          outer: getComputedStyle(node.closest('dialog')).boxShadow,
          surface: getComputedStyle(node.closest('.settings-surface')).clipPath,
        };
      }),
    );
    await page.mouse.move(10, 10);
    await page.screenshot({ path: path.join(out, `app-settings-${theme}-rest.png`) });
    await toggle.hover();
    await page.screenshot({ path: path.join(out, `app-settings-${theme}-hover.png`) });
    await page.mouse.down();
    await page.screenshot({ path: path.join(out, `app-settings-${theme}-held.png`) });
    await page.mouse.up();
  }
  // Use the production FileDialog and menu command without writing a scene file.
  for (const theme of ['dark', 'light']) {
    await page.getByRole('button', { name: '常规', exact: true }).click();
    await page
      .getByRole('button', { name: theme === 'dark' ? '深色' : '浅色', exact: true })
      .click();
    await page.keyboard.press('Escape');
    await app.evaluate(({ Menu, BrowserWindow }) => {
      const action = Menu.getApplicationMenu().getMenuItemById('file.save');
      action.click(action, BrowserWindow.getAllWindows()[0], {});
    });
    const dialog = page.getByRole('dialog', { name: '场景另存为' });
    await dialog.waitFor();
    const location = dialog.getByRole('textbox', { name: '位置', exact: true });
    await location.fill(profile);
    await location.press('Enter');
    await page.waitForTimeout(100);
    const save = dialog.getByRole('button', { name: '保存', exact: true });
    await save.waitFor();
    for (const state of ['rest', 'hover', 'held']) {
      if (state === 'rest') await page.mouse.move(5, 5);
      else await save.hover();
      if (state === 'held') await page.mouse.down();
      // Wait for any existing CSS transition before comparing the resolved color.
      await page.waitForTimeout(180);
      const ink = await save.evaluate((node) => ({
        background: getComputedStyle(node).backgroundColor,
        foreground: getComputedStyle(node).color,
        variant: node.dataset.buttonVariant,
      }));
      const expected = {
        dark: {
          rest: 'rgb(156, 191, 141)',
          hover: 'rgb(103, 135, 90)',
          held: 'rgb(184, 210, 172)',
        },
        light: { rest: 'rgb(76, 112, 64)', hover: 'rgb(69, 104, 58)', held: 'rgb(62, 97, 53)' },
      }[theme][state];
      assert.equal(ink.background, expected);
      assert.equal(ink.variant, 'primary');
      buttons.push({ theme, state, ...ink });
      await page.screenshot({ path: path.join(out, `app-save-${theme}-${state}.png`) });
    }
    await page.mouse.move(5, 5);
    await page.mouse.up();
    await dialog.getByRole('button', { name: '取消', exact: true }).click();
    await app.evaluate(({ Menu, BrowserWindow }) => {
      const action = Menu.getApplicationMenu().getMenuItemById('app.settings');
      action.click(action, BrowserWindow.getAllWindows()[0], {});
    });
    await page.getByRole('dialog', { name: '设置' }).waitFor();
  }
  assert.ok(
    geometry.every(
      (record) =>
        /round|superellipse\(1\)/.test(record.track) &&
        /round|superellipse\(1\)/.test(record.thumb) &&
        record.thumbBounds.width === record.thumbBounds.height &&
        record.outer === 'none' &&
        record.surface === 'none',
    ),
  );
  await writeFile(
    path.join(out, 'production-controls-review.json'),
    JSON.stringify({ geometry, buttons, errors }, null, 2) + '\n',
  );
  assert.deepEqual(errors, []);
  console.log(
    'Captured actual Electron App settings: dark/light, rest/hover/held; one corner painter and stable category geometry.',
  );
} finally {
  await app?.close();
  await server.close();
  await rm(profile, { recursive: true, force: true });
}
