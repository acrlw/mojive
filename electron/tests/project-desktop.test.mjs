import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'vite';
import { _electron } from 'playwright';
import { mkdtemp, mkdir, realpath, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';

test('actual desktop project root and native Close Project action retain the edited scene across close/reopen', async () => {
  const temporary = await realpath(
    await mkdtemp(path.join(os.tmpdir(), 'mojive-project-desktop-')),
  );
  const folder = path.join(temporary, 'mojive');
  await mkdir(path.join(folder, '3rdparty'), { recursive: true });
  await writeFile(path.join(folder, '3rdparty', 'resource.txt'), 'Project resource');
  let server, app, page;
  const errors = [];
  try {
    server = await createServer({
      server: { host: '127.0.0.1', port: 0, strictPort: false },
      logLevel: 'error',
    });
    await server.listen();
    app = await _electron.launch({
      args: [
        path.resolve('main/main.cjs'),
        '--hidden',
        '--user-data-dir=' + path.join(temporary, 'profile'),
      ],
      env: {
        ...process.env,
        MOJIVE_DEV_URL: `http://127.0.0.1:${server.httpServer.address().port}`,
      },
    });
    page = await app.firstWindow();
    page.setDefaultTimeout(10000);
    page.on('pageerror', (error) => errors.push(error.message));
    await page.getByRole('treeitem').first().waitFor();
    const action = async (id) => {
      for (let index = 0; index < 80; index++) {
        if (
          await app.evaluate(
            ({ Menu }, id) => !!Menu.getApplicationMenu().getMenuItemById(id)?.enabled,
            id,
          )
        )
          break;
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
      return app.evaluate(({ Menu, BrowserWindow }, id) => {
        const item = Menu.getApplicationMenu().getMenuItemById(id);
        if (!item?.enabled) throw Error('Unavailable menu action: ' + id);
        item.click(item, BrowserWindow.getAllWindows()[0], {});
      }, id);
    };
    const enabled = async () =>
      app.evaluate(
        ({ Menu }) => Menu.getApplicationMenu().getMenuItemById('file.closeProject').enabled,
      );
    const waitEnabled = async (value) => {
      for (let i = 0; i < 40; i++) {
        if ((await enabled()) === value) return;
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
      assert.equal(await enabled(), value);
    };
    await action('locale.en');
    await page.getByTitle('Project files', { exact: true }).waitFor();
    await action('add.box');
    const x = page.getByLabel('Position X', { exact: true });
    await x.dblclick();
    await x.fill('2');
    await x.press('Enter');
    const title = await page.locator('.document-title').innerText();
    assert.equal(await enabled(), false);
    await action('file.openProject');
    await page.getByRole('textbox', { name: 'Location', exact: true }).fill(folder);
    await page.getByRole('button', { name: 'Select folder', exact: true }).click();
    await page.getByRole('dialog').waitFor({ state: 'hidden' });
    assert.equal(await page.locator('.left-content .group-eyebrow').getAttribute('title'), folder);
    assert.equal(await page.locator('.file-row').count(), 1);
    await waitEnabled(true);
    await page.locator('.file-row').click();
    await page
      .locator(`.file-row[title="${path.join(folder, '3rdparty', 'resource.txt')}"]`)
      .waitFor();
    await page.locator('.panel-search input').fill('resource');
    await page.waitForFunction(
      () =>
        document.querySelectorAll('.file-row').length === 1 &&
        document.querySelector('.file-row').textContent.includes('resource.txt'),
    );
    await action('file.closeProject');
    await waitEnabled(false);
    await page.locator('.left-panel .empty').getByText('No folder open', { exact: true }).waitFor();
    assert.equal(await page.locator('.left-content .group-eyebrow').count(), 0);
    assert.equal(await page.locator('.file-row').count(), 0);
    assert.equal(await page.locator('.panel-search input').count(), 0);
    assert.equal(await page.locator('.document-title').innerText(), title);
    assert.equal(await page.locator('.dirty-dot').count(), 1);
    assert.equal(Number(await x.inputValue()), 2);
    await page.evaluate(() => document.activeElement?.blur());
    await page.waitForFunction(() => !document.activeElement?.matches('input,textarea'));
    for (let index = 0; index < 80; index++) {
      if (
        await app.evaluate(
          ({ Menu }) =>
            Menu.getApplicationMenu().getMenuItemById('view.focusAll').accelerator === 'Shift+F',
        )
      )
        break;
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    assert.equal(
      await app.evaluate(
        ({ Menu }) => Menu.getApplicationMenu().getMenuItemById('view.focusAll').accelerator,
      ),
      'Shift+F',
    );
    await action('edit.undo');
    await page.waitForFunction(
      () => Number(document.querySelector('input[aria-label="Position X"]')?.value) === 0,
    );
    await action('edit.redo');
    await page.waitForFunction(
      () => Number(document.querySelector('input[aria-label="Position X"]')?.value) === 2,
    );
    await action('file.openProject');
    await page.getByRole('textbox', { name: 'Location', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Select folder', exact: true }).click();
    await page.getByRole('dialog').waitFor({ state: 'hidden' });
    assert.equal(await page.locator('.left-content .group-eyebrow').getAttribute('title'), folder);
    await waitEnabled(true);
    await page.getByTitle('Close project folder', { exact: true }).click();
    await waitEnabled(false);
    assert.equal(Number(await x.inputValue()), 2);
    await mkdir('../output/electron/verification', { recursive: true });
    await page.screenshot({ path: '../output/electron/verification/project-desktop-closed.png' });
    assert.deepEqual(errors, []);
  } catch (error) {
    console.error(error);
    if (page) {
      console.error(await page.locator('body').innerText());
      await page.screenshot({ path: '../output/electron/verification/project-desktop-failure.png' });
    }
    throw error;
  } finally {
    await app
      ?.evaluate(({ BrowserWindow }) =>
        BrowserWindow.getAllWindows().forEach((window) => window.destroy()),
      )
      .catch(() => {});
    await app?.close();
    await server?.close();
    await rm(temporary, { recursive: true, force: true });
  }
});
