import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { mkdtemp, mkdir, writeFile, rm, readFile, realpath } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { _electron } from 'playwright';

test('desktop preload opens local URDF, streams authorized resources, and saves atomically', async () => {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const temporary = await mkdtemp(path.join(os.tmpdir(), 'mojive-desktop-'));
  const project = path.join(temporary, 'robot_description');
  await mkdir(path.join(project, 'urdf'), { recursive: true });
  await mkdir(path.join(project, 'meshes'));
  const urdfPath = path.join(project, 'urdf', 'robot.urdf');
  await writeFile(urdfPath, '<robot name="desktop"><link name="base"/></robot>');
  await writeFile(path.join(project, 'meshes', 'part.stl'), 'solid sample\nendsolid sample');
  const outside = path.join(temporary, 'private.urdf');
  await writeFile(outside, '<robot name="private"/>');
  const server = createServer((_req, res) => {
    res.setHeader('content-type', 'text/html');
    res.end('<!doctype html><title>Desktop bridge test</title>');
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  let app;
  try {
    app = await _electron.launch({
      args: [path.join(root, 'main/main.cjs'), '--hidden'],
      env: { ...process.env, MOJIVE_DEV_URL: `http://127.0.0.1:${server.address().port}` },
    });
    const page = await app.firstWindow();
    await page.waitForFunction(() => Boolean(window.mojive));
    assert.equal(await page.evaluate(() => typeof require), 'undefined');
    const listing = await page.evaluate(
      (directory) => window.mojive.browseFiles(directory),
      project,
    );
    assert.ok(listing.entries.some((entry) => entry.name === 'urdf' && entry.directory));
    const resource = await page.evaluate(
      (filename) => window.mojive.openUrdfPath(filename),
      urdfPath,
    );
    assert.equal(resource.name, 'robot.urdf');
    assert.match(resource.content, /desktop/);
    const mesh = await page.evaluate(async (base) => {
      const response = await fetch(new URL('../meshes/part.stl', base));
      return { status: response.status, text: await response.text() };
    }, resource.resourceBase);
    assert.equal(mesh.status, 200);
    assert.match(mesh.text, /sample/);
    const packageMesh = await page.evaluate(async (base) => {
      const token = new URL(base).pathname.split('/')[1];
      const response = await fetch(
        `mojive-resource://project/${token}/@package/robot_description/meshes/part.stl`,
      );
      return response.status;
    }, resource.resourceBase);
    assert.equal(packageMesh, 200);
    const blocked = await page.evaluate(async (filename) => {
      try {
        await window.mojive.readUrdf(filename);
        return '';
      } catch (error) {
        return error.message;
      }
    }, outside);
    assert.match(blocked, /outside the opened project/);
    assert.equal(
      await page.evaluate(() => window.mojive.hasNativeMenu),
      process.platform === 'darwin',
    );
    await page.evaluate(() => {
      window.menuActions = [];
      window.unsubscribeMenu = window.mojive.onMenuAction((action) =>
        window.menuActions.push(action),
      );
      return window.mojive.updateMenuState({
        locale: 'zh-CN',
        theme: 'light',
        canUndo: true,
        canRedo: false,
        hasSelection: true,
        showLeft: true,
        showInspector: false,
        showTimeline: true,
        sceneId: 'joint_gizmo',
        textInput: false,
      });
    });
    if (process.platform === 'darwin') {
      const menuState = await app.evaluate(({ Menu, nativeTheme }) => {
        const menu = Menu.getApplicationMenu();
        menu.getMenuItemById('app.settings').click();
        return {
          labels: menu.items.map((item) => item.label),
          theme: nativeTheme.themeSource,
          undo: menu.getMenuItemById('edit.undo').enabled,
          redo: menu.getMenuItemById('edit.redo').enabled,
          inspector: menu.getMenuItemById('view.toggleInspector').checked,
          icon: (() => {
            const image = menu.getMenuItemById('add.box').icon;
            return {
              empty: image.isEmpty(),
              template: image.isTemplateImage(),
              scales: image.getScaleFactors(),
              size: image.getSize(),
            };
          })(),
          helpRole: menu.items.find((item) => item.label === '帮助').role,
        };
      });
      await page.waitForFunction(() => window.menuActions.length === 1);
      assert.deepEqual(await page.evaluate(() => window.menuActions), ['app.settings']);
      assert.ok(menuState.labels.includes('文件'));
      assert.equal(menuState.theme, 'light');
      assert.equal(menuState.undo, true);
      assert.equal(menuState.redo, false);
      assert.equal(menuState.inspector, false);
      assert.equal(menuState.icon.empty, false);
      assert.equal(menuState.icon.template, true);
      assert.deepEqual(menuState.icon.scales, [1, 2]);
      assert.deepEqual(menuState.icon.size, { width: 16, height: 16 });
      assert.equal(menuState.helpRole, 'help');
      await page.evaluate(() => {
        const input = document.createElement('input');
        input.id = 'native-search';
        document.body.append(input);
        input.focus();
        return window.mojive.updateMenuState({
          locale: 'zh-CN',
          theme: 'light',
          canUndo: false,
          canRedo: false,
          hasSelection: true,
          showLeft: true,
          showInspector: false,
          showTimeline: true,
          sceneId: 'joint_gizmo',
          textInput: true,
        });
      });
      await page.keyboard.press('f');
      await page.keyboard.press('Shift+F');
      assert.equal(await page.locator('#native-search').inputValue(), 'fF');
      const accelerator = await app.evaluate(
        ({ Menu }) => Menu.getApplicationMenu().getMenuItemById('view.focusSelection').accelerator,
      );
      assert.ok(accelerator == null);
      await app.evaluate(({ Menu }) =>
        Menu.getApplicationMenu().getMenuItemById('edit.undo').click(),
      );
      await page.waitForFunction(() => document.querySelector('#native-search').value.length < 2);
      assert.deepEqual(await page.evaluate(() => window.menuActions), ['app.settings']);
      await page.evaluate(() => window.unsubscribeMenu());
      await app.evaluate(({ Menu }) =>
        Menu.getApplicationMenu().getMenuItemById('file.save').click(),
      );
      assert.equal(await page.evaluate(() => window.menuActions.length), 1);
    }
    const clipboardBefore = await app.evaluate(({ clipboard }) => clipboard.readText());
    try {
      await page.evaluate(() => window.mojive.writeClipboardText('#9CBF8D'));
      assert.equal(await page.evaluate(() => window.mojive.readClipboardText()), '#9CBF8D');
      const invalidClipboard = await page.evaluate(async () => {
        try {
          await window.mojive.writeClipboardText({ html: 'not plain text' });
          return '';
        } catch (error) {
          return error.message;
        }
      });
      assert.match(invalidClipboard, /must be a string/);
    } finally {
      await app.evaluate(({ clipboard }, text) => clipboard.writeText(text), clipboardBefore);
    }
    const savePath = path.join(project, 'scene.mojive.json');
    assert.equal(
      await page.evaluate(
        (filename) => window.mojive.saveScenePath(filename, '{\"test\":true}'),
        savePath,
      ),
      await realpath(savePath),
    );
    assert.deepEqual(JSON.parse(await readFile(savePath, 'utf8')), { test: true });
  } finally {
    await app?.close();
    await new Promise((resolve) => server.close(resolve));
    await rm(temporary, { recursive: true, force: true });
  }
});
