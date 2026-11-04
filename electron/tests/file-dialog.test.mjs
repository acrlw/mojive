import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'vite';
import { _electron } from 'playwright';
import { mkdtemp, mkdir, writeFile, readFile, realpath, rm } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
test('custom desktop file dialog browses, opens URDF/folders, saves with an owned overwrite modal and cancels safely', async () => {
  const temporary = await realpath(await mkdtemp(path.join(os.tmpdir(), 'mojive-dialog-')));
  let server, app;
  try {
    await mkdir(path.join(temporary, 'robots'));
    await writeFile(
      path.join(temporary, 'robots', 'test.urdf'),
      '<robot name="Test"><link name="base"/></robot>',
    );
    await writeFile(path.join(temporary, 'existing.mojive.json'), '{"original":true}');
    await writeFile(path.join(temporary, '.hidden.json'), '{}');
    server = await createServer({
      plugins: [
        {
          name: 'file-dialog-fixture',
          configureServer(server) {
            server.middlewares.use((req, res, next) => {
              if (req.url === '/') {
                res.statusCode = 302;
                res.setHeader(
                  'Location',
                  '/tests/harness/file-dialog-review.html?directory=' + encodeURIComponent(temporary),
                );
                res.end();
                return;
              }
              next();
            });
          },
        },
      ],
      server: { host: '127.0.0.1', port: 0, strictPort: false },
      logLevel: 'error',
    });
    await server.listen();
    const base = `http://127.0.0.1:${server.httpServer.address().port}`;
    app = await _electron.launch({
      args: [
        path.resolve('main/main.cjs'),
        '--hidden',
        '--user-data-dir=' + path.join(temporary, 'profile'),
      ],
      env: { ...process.env, MOJIVE_DEV_URL: base },
    });
    const page = await app.firstWindow();
    page.setDefaultTimeout(8000);
    await page.getByRole('dialog', { name: 'Open URDF', exact: true }).waitFor();
    await page.getByRole('option', { name: 'robots', exact: true }).dblclick();
    await page.getByRole('textbox', { name: 'Filter files', exact: true }).fill('test');
    await page.getByRole('option', { name: 'test.urdf', exact: true }).click();
    await page.getByRole('button', { name: 'Open', exact: true }).click();
    await page.waitForFunction(() => window.fileReview.choices.length === 1);
    await page.getByRole('dialog').waitFor({ state: 'hidden' });
    assert.equal(await page.evaluate(() => window.fileReview.choices[0].result.name), 'test.urdf');
    await page.evaluate(() => window.showFileDialog('project'));
    await page.getByRole('textbox', { name: 'Location', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Select folder', exact: true }).click();
    await page.waitForFunction(() => window.fileReview.choices.length === 2);
    await page.getByRole('dialog').waitFor({ state: 'hidden' });
    assert.equal(await page.evaluate(() => window.fileReview.choices[1].kind), 'project');
    assert.equal(await page.evaluate(() => window.fileReview.choices[1].path), temporary);
    assert.equal(await page.evaluate(() => window.fileReview.choices[1].result.path), temporary);
    await page.evaluate(() => window.showFileDialog('project'));
    await page.getByRole('option', { name: 'robots', exact: true }).dblclick();
    await page.waitForFunction(
      (expected) => document.querySelector('input[aria-label="Location"]').value === expected,
      path.join(temporary, 'robots'),
    );
    // Do not press Enter: confirmation must resolve the visible path, not the old listing.
    await page.getByRole('textbox', { name: 'Location', exact: true }).fill(temporary);
    await page.getByRole('button', { name: 'Select folder', exact: true }).click();
    await page.waitForFunction(() => window.fileReview.choices.length === 3);
    await page.getByRole('dialog').waitFor({ state: 'hidden' });
    assert.equal(await page.evaluate(() => window.fileReview.choices[2].path), temporary);
    assert.equal(await page.evaluate(() => window.fileReview.choices[2].result.path), temporary);
    await page.evaluate(() => window.showFileDialog('save'));
    await page.getByRole('dialog', { name: 'Save scene as', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Save', exact: true }).click();
    await page.getByRole('alertdialog', { name: 'Replace existing file?', exact: true }).waitFor();
    await page
      .getByRole('alertdialog')
      .getByRole('button', { name: 'Cancel', exact: true })
      .click();
    assert.equal(
      await readFile(path.join(temporary, 'existing.mojive.json'), 'utf8'),
      '{"original":true}',
    );
    assert.equal(await page.getByRole('dialog', { name: 'Save scene as', exact: true }).count(), 1);
    await page.getByRole('button', { name: 'Save', exact: true }).click();
    await mkdir('../output/electron/verification', { recursive: true });
    await page.screenshot({ path: '../output/electron/verification/file-dialog-overwrite.png' });
    await page
      .getByRole('alertdialog')
      .getByRole('button', { name: 'Replace', exact: true })
      .click();
    await page.waitForFunction(() => window.fileReview.choices.length === 4);
    await page.getByRole('dialog').waitFor({ state: 'hidden' });
    assert.equal(
      await readFile(path.join(temporary, 'existing.mojive.json'), 'utf8'),
      '{"saved":true}',
    );
    await page.evaluate(() => window.showFileDialog('scene'));
    await page.getByRole('textbox', { name: 'Location', exact: true }).waitFor();
    await page
      .getByRole('textbox', { name: 'Location', exact: true })
      .fill('/path-that-does-not-exist');
    await page.getByRole('textbox', { name: 'Location', exact: true }).press('Enter');
    await page.getByRole('alert').waitFor();
    assert.equal(await page.getByRole('dialog').count(), 1);
    await page.getByRole('textbox', { name: 'Location', exact: true }).fill(temporary);
    await page.getByRole('textbox', { name: 'Location', exact: true }).press('Enter');
    await page.getByRole('option', { name: 'existing.mojive.json', exact: true }).waitFor();
    await page.screenshot({ path: '../output/electron/verification/file-dialog-scene.png' });
    await page.keyboard.press('Escape');
    assert.equal(await page.getByRole('dialog').count(), 0);
    assert.equal(await page.evaluate(() => typeof window.mojive.openUrdf), 'undefined');
  } catch (error) {
    if (app) {
      const page = await app.firstWindow();
      console.error(await page.locator('body').innerText());
      await page.screenshot({ path: '../output/electron/verification/file-dialog-failure.png' });
    }
    throw error;
  } finally {
    await app?.close();
    await server?.close();
    await rm(temporary, { recursive: true, force: true });
  }
});
