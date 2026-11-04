import assert from 'node:assert/strict';
import { test, before, after, afterEach } from 'node:test';
import { createRequire } from 'node:module';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdtemp, mkdir, writeFile, readFile, realpath, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
const require = createRequire(import.meta.url);
const { FileAccess } = require('../main/file-access.cjs');
const { Projects } = require('../main/resource-paths.cjs');
let server, browser, page, temporary, project, child, empty, access;
const errors = [],
  opened = [],
  projects = new Projects();
before(async () => {
  temporary = await realpath(await mkdtemp(path.join(os.tmpdir(), 'mojive-project-ui-')));
  project = path.join(temporary, 'mojive');
  child = path.join(project, '3rdparty');
  empty = path.join(temporary, 'empty');
  await mkdir(child, { recursive: true });
  await mkdir(path.join(project, 'robots'));
  await mkdir(empty);
  await writeFile(
    path.join(project, 'robots', 'arm.urdf'),
    '<robot name="Project Robot"><link name="base"><visual><geometry><box size="1 1 1"/></geometry></visual></link></robot>',
  );
  access = new FileAccess([{ label: 'Home', path: project }]);
  server = await createServer({
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    logLevel: 'error',
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 1460, height: 950 } });
  page.setDefaultTimeout(10000);
  page.on('pageerror', (error) => errors.push(error.message));
  await page.exposeFunction('browseLocal', (directory, hidden) => access.browse(directory, hidden));
  await page.exposeFunction('openLocalProject', async (directory) => {
    const root = await realpath(directory);
    await projects.register(root);
    opened.push(root);
    return { path: root, name: path.basename(root), files: await projects.list(root) };
  });
  await page.exposeFunction('readLocalDirectory', async (directory) => {
    await projects.authorized(directory);
    return projects.list(directory);
  });
  await page.exposeFunction('searchLocalProject', async (directory, query) => {
    await projects.authorized(directory);
    return projects.search(directory, query);
  });
  await page.exposeFunction('readLocalUrdf', async (filename) => {
    await projects.authorized(filename);
    return {
      content: await readFile(filename, 'utf8'),
      name: path.basename(filename),
      resourceBase: '',
      files: await projects.tree(path.dirname(filename)),
    };
  });
  await page.addInitScript(() => {
    localStorage.setItem(
      'mojive.electron.settings.v1',
      JSON.stringify({ language: 'en', appearance: 'dark', uiScale: 1 }),
    );
    const handlers = new Set();
    window.projectMenu = {
      state: null,
      invoke: (action) => {
        for (const callback of [...handlers]) callback(action);
      },
    };
    window.mojive = {
      browseFiles: window.browseLocal,
      openProjectPath: window.openLocalProject,
      readProjectDirectory: window.readLocalDirectory,
      searchProject: window.searchLocalProject,
      readUrdf: window.readLocalUrdf,
      hasNativeMenu: false,
      platform: 'linux',
      onMenuAction: (callback) => {
        handlers.add(callback);
        return () => handlers.delete(callback);
      },
      updateMenuState: async (state) => {
        window.projectMenu.state = state;
      },
    };
  });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}`);
  await page.getByRole('treeitem').first().waitFor();
  await page.getByTitle('Project files', { exact: true }).click();
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
  await rm(temporary, { recursive: true, force: true });
});
afterEach(async (context) => {
  if (!context.passed) {
    console.error(await page.locator('body').innerText());
    await page.screenshot({ path: '../output/electron/verification/project-workflow-failure.png' });
  }
});
const location = () => page.getByRole('textbox', { name: 'Location', exact: true });
const confirm = () => page.getByRole('button', { name: 'Select folder', exact: true });
const heading = () => page.locator('.left-content .group-eyebrow');
const open = async () => {
  await page.getByTitle('Open project folder', { exact: true }).click();
  await page.getByRole('dialog', { name: 'Open project folder', exact: true }).waitFor();
  await page.waitForFunction(
    () => document.querySelector('.file-dialog-list')?.getAttribute('aria-busy') === 'false',
  );
  await page.waitForFunction(
    () => !!document.querySelector('.file-dialog .primary-button:not(:disabled)'),
  );
};
const navigate = async (directory) => {
  await location().fill(directory);
  await location().press('Enter');
  await page.waitForFunction(
    (directory) =>
      document.querySelector('input[aria-label="Location"]').value === directory &&
      document.querySelector('.file-dialog-list').getAttribute('aria-busy') === 'false',
    directory,
  );
};
const chosen = async (expected) => {
  await confirm().click();
  await page.getByRole('dialog').waitFor({ state: 'hidden' });
  assert.equal(await heading().getAttribute('title'), expected);
  assert.equal(opened.at(-1), expected);
  await page.waitForFunction(() => window.projectMenu.state.hasProject);
};

test('project picker commits current directory, explicit child and typed location without silently choosing the first folder', async () => {
  await open();
  assert.equal(await location().inputValue(), project);
  assert.equal(await page.getByRole('option', { selected: true }).count(), 0);
  await chosen(project);
  await open();
  await page.getByRole('option', { name: '3rdparty', exact: true }).click();
  await chosen(child);
  await open();
  assert.equal(await location().inputValue(), child);
  // Regression: editing Location and immediately confirming used to return the old child.
  await location().fill(project);
  await chosen(project);
  await open();
  await page.getByRole('option', { name: '3rdparty', exact: true }).dblclick();
  await page.waitForFunction(
    (child) => document.querySelector('input[aria-label="Location"]').value === child,
    child,
  );
  assert.equal(await page.getByRole('option', { selected: true }).count(), 0);
  await chosen(child);
  await open();
  await page.getByRole('listbox').focus();
  await page.keyboard.press('Backspace');
  await page.getByRole('option', { name: '3rdparty', exact: true }).waitFor();
  await page.keyboard.press('Home');
  assert.equal(
    await page.getByRole('option', { selected: true }).getAttribute('aria-label'),
    '3rdparty',
  );
  await page.keyboard.press('Enter');
  await page.waitForFunction(
    (child) => document.querySelector('input[aria-label="Location"]').value === child,
    child,
  );
  await page.keyboard.press('Backspace');
  await page.getByRole('option', { name: 'robots', exact: true }).waitFor();
  await chosen(project);
  const count = opened.length;
  await open();
  await page.getByRole('option', { name: '3rdparty', exact: true }).click();
  await page.getByRole('button', { name: 'Cancel', exact: true }).click();
  assert.equal(opened.length, count);
  assert.equal(await heading().getAttribute('title'), project);
  await open();
  await location().fill(path.join(temporary, 'missing'));
  await confirm().click();
  await page.getByRole('alert').waitFor();
  assert.equal(opened.length, count, 'invalid typed location never commits the old directory');
  await page.keyboard.press('Escape');
  await open();
  await navigate(project);
  await page.screenshot({ path: '../output/electron/verification/project-current-directory.png' });
  await chosen(project);
});

test('project root survives folder navigation and scene loads; close clears files/search while preserving scene, edits and undo', async () => {
  await open();
  await location().fill(project);
  await chosen(project);
  const robots = page.locator(`.file-row[title="${path.join(project, 'robots')}"]`);
  await robots.dblclick();
  assert.equal(
    await heading().getAttribute('title'),
    project,
    'explorer folders never replace the project root',
  );
  await page.locator(`.file-row[title="${path.join(project, 'robots', 'arm.urdf')}"]`).dblclick();
  await page.waitForFunction(() =>
    document.querySelector('.document-title').textContent.includes('Project Robot'),
  );
  assert.equal(
    await heading().getAttribute('title'),
    project,
    'loading a model does not replace the explorer with model siblings',
  );
  await page
    .getByRole('navigation', { name: 'Application menu' })
    .getByRole('button', { name: 'Add', exact: true })
    .click();
  await page.getByRole('menuitem', { name: 'Box', exact: true }).click();
  const x = page.getByLabel('Position X', { exact: true });
  await x.dblclick();
  await x.fill('2');
  await x.press('Enter');
  await page.locator('.left-panel .panel-search input').fill('arm');
  await page.locator('.file-row').filter({ hasText: 'arm.urdf' }).waitFor();
  await page.screenshot({
    path: '../output/electron/verification/project-open-scene-preserved.png',
  });
  await page.getByTitle('Close project folder', { exact: true }).click();
  await page.locator('.left-panel .empty').getByText('No folder open', { exact: true }).waitFor();
  assert.equal(await heading().count(), 0);
  assert.equal(await page.locator('.file-row').count(), 0);
  assert.equal(await page.locator('.left-panel .panel-search input').count(), 0);
  assert.equal(await x.inputValue(), '2.000');
  assert.match(await page.locator('.document-title').innerText(), /Project Robot/);
  assert.equal(await page.locator('.dirty-dot').count(), 1);
  await page.waitForFunction(() => !window.projectMenu.state.hasProject);
  assert.equal(await page.getByTitle('Close project folder', { exact: true }).isDisabled(), true);
  await page.evaluate(() => window.projectMenu.invoke('edit.undo'));
  await page.waitForFunction(
    () => Number(document.querySelector('input[aria-label="Position X"]')?.value) === 0,
  );
  await page.evaluate(() => window.projectMenu.invoke('edit.redo'));
  await page.waitForFunction(
    () => Number(document.querySelector('input[aria-label="Position X"]')?.value) === 2,
  );
  await page.screenshot({
    path: '../output/electron/verification/project-closed-scene-preserved.png',
  });
  await open();
  await location().fill(project);
  await chosen(project);
  assert.equal(await x.inputValue(), '2.000');
  await page.evaluate(() => window.projectMenu.invoke('file.closeProject'));
  await page.waitForFunction(() => !window.projectMenu.state.hasProject);
  await page.locator('.left-panel .empty').getByText('No folder open', { exact: true }).waitFor();
  assert.equal(await heading().count(), 0);
  assert.equal(await x.inputValue(), '2.000');
  // Empty open projects are not confused with the no-project state.
  await open();
  await location().fill(empty);
  await chosen(empty);
  await page.getByText('This folder is empty', { exact: true }).waitFor();
  await page
    .getByRole('navigation', { name: 'Application menu' })
    .getByRole('button', { name: 'File', exact: true })
    .click();
  await page.getByRole('menuitem', { name: 'Close project folder', exact: true }).click();
  await page.locator('.left-panel .empty').getByText('No folder open', { exact: true }).waitFor();
  assert.equal(await heading().count(), 0);
  assert.equal(await x.inputValue(), '2.000');
  assert.deepEqual(errors, []);
});

test('the actual Mojive repository opens with every root entry and searches files in unopened directories', async () => {
  const repository = await realpath(path.resolve('..'));
  await open();
  await location().fill(repository);
  await chosen(repository);
  const expected = await projects.list(repository);
  assert.deepEqual(
    await page.locator('.file-row').evaluateAll((rows) => rows.map((row) => row.title)),
    expected.map((file) => file.path),
  );
  assert.ok(expected.some((file) => file.name === '3rdparty'));
  assert.ok(expected.some((file) => file.name === 'assets'));
  assert.equal(await page.locator('.file-row[aria-expanded="true"]').count(), 0);
  await page.screenshot({ path: '../output/electron/verification/project-mojive-root.png' });
  await page.locator('.panel-search input').fill('joint_gizmo.xml');
  await page
    .locator(`.file-row[title="${path.join(repository, 'assets', 'joint_gizmo.xml')}"]`)
    .waitFor();
  await page.screenshot({
    path: '../output/electron/verification/project-mojive-recursive-search.png',
  });
  await page.getByTitle('Close project folder', { exact: true }).click();
  assert.equal(await page.locator('.file-row').count(), 0);
  assert.equal(await page.locator('.panel-search input').count(), 0);
  assert.deepEqual(errors, []);
});
