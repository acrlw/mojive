import { installPaintStyle } from './helpers/surface-style.mjs';
import assert from 'node:assert/strict';
import { after, afterEach, before, beforeEach, test } from 'node:test';
import { createRequire } from 'node:module';
import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const require = createRequire(import.meta.url);
const { FileAccess } = require('../main/file-access.cjs');
const { Projects } = require('../main/resource-paths.cjs');
let server, browser, page, base, temporary, project, urdf, access;
const projects = new Projects();
const openedProjects = [],
  openedUrdfs = [],
  errors = [];

before(async () => {
  temporary = await realpath(await mkdtemp(path.join(os.tmpdir(), 'mojive-empty-states-')));
  project = path.join(temporary, 'mojive');
  urdf = path.join(project, 'arm.urdf');
  await mkdir(path.join(project, '3rdparty'), { recursive: true });
  await writeFile(
    urdf,
    '<robot name="Empty state robot"><link name="base"><visual><geometry><box size="1 1 1"/></geometry></visual></link></robot>',
  );
  access = new FileAccess([{ label: 'Home', path: project }]);
  server = await createServer({
    configFile: false,
    root: process.cwd(),
    cacheDir: 'node_modules/.vite-empty-states-test',
    server: { host: '127.0.0.1', port: 0, hmr: false },
    optimizeDeps: {
      include: ['react', 'react-dom', 'react-dom/client', 'react/jsx-runtime', 'three'],
    },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
  page.setDefaultTimeout(10000);
  page.on('pageerror', (error) => errors.push(error.message));
  await page.exposeFunction('browseLocalFiles', (directory, hidden) =>
    access.browse(directory, hidden),
  );
  await page.exposeFunction('openLocalProject', async (directory) => {
    const root = await realpath(directory);
    await projects.register(root);
    openedProjects.push(root);
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
  await page.exposeFunction('openLocalUrdf', async (filename) => {
    assert.equal(await realpath(filename), urdf);
    openedUrdfs.push(filename);
    return {
      content: await readFile(filename, 'utf8'),
      name: path.basename(filename),
      resourceBase: '',
      files: [],
    };
  });
  await page.addInitScript(() => {
    localStorage.setItem(
      'mojive.electron.settings.v1',
      JSON.stringify({ language: 'en', appearance: 'dark', uiScale: 1 }),
    );
    localStorage.setItem('mojive.layout', JSON.stringify({ left: 300, right: 380, bottom: 280 }));
    window.mojive = {
      browseFiles: window.browseLocalFiles,
      openProjectPath: window.openLocalProject,
      readProjectDirectory: window.readLocalDirectory,
      searchProject: window.searchLocalProject,
      openUrdfPath: window.openLocalUrdf,
      hasNativeMenu: false,
      platform: 'linux',
      onMenuAction: () => () => {},
      updateMenuState: async () => {},
    };
  });
  await mkdir('../output/electron/verification', { recursive: true });
});

beforeEach(async () => {
  errors.length = 0;
  await page.goto(base);
  await page.getByRole('treeitem').first().waitFor();
});
afterEach(async (context) => {
  if (!context.passed) {
    console.error(await page.locator('body').innerText());
    await page.screenshot({ path: '../output/electron/verification/empty-states-failure.png' });
  }
  assert.deepEqual(errors, []);
});
after(async () => {
  await browser?.close();
  await server?.close();
  await rm(temporary, { recursive: true, force: true });
});

const panel = () => page.locator('.left-panel');
const emptyAction = (label) =>
  panel().locator('.empty-actions').getByRole('button', { name: label, exact: true });
async function openFolderFromEmpty() {
  await page.getByRole('button', { name: 'Project files', exact: true }).click();
  await emptyAction('Open folder').click();
  await page.getByRole('dialog', { name: 'Open project folder', exact: true }).waitFor();
  await page.locator('.file-dialog-list[aria-busy="false"]').waitFor();
  assert.equal(
    await page.getByRole('textbox', { name: 'Location', exact: true }).inputValue(),
    project,
  );
  await page.getByRole('button', { name: 'Select folder', exact: true }).click();
  await page
    .getByRole('dialog', { name: 'Open project folder', exact: true })
    .waitFor({ state: 'hidden' });
}

test('scene no-results action clears the query and object type together', async () => {
  await panel().getByRole('button', { name: 'Joint', exact: true }).click();
  await panel().getByRole('button', { name: 'Search', exact: true }).click();
  const search = page.getByRole('textbox', { name: 'Search Scene', exact: true });
  await search.fill('no-such-object');
  await panel().getByText('No matching objects', { exact: true }).waitFor();
  assert.equal(await panel().getByRole('treeitem').count(), 0);
  await emptyAction('Clear filters').click();
  assert.equal(await search.inputValue(), '');
  assert.equal(
    await panel().getByRole('button', { name: 'All', exact: true }).getAttribute('aria-pressed'),
    'true',
  );
  await panel().getByRole('treeitem').first().waitFor();
  assert.ok((await panel().getByRole('treeitem').count()) > 9);
});

test('project empty action opens the exact current folder and remains reversible', async () => {
  await page.getByRole('button', { name: 'Project files', exact: true }).click();
  await panel().getByText('No folder open', { exact: true }).waitFor();
  assert.equal(await panel().locator('.group-eyebrow').count(), 0);
  assert.equal(
    await panel().getByRole('textbox', { name: 'Search Project', exact: true }).count(),
    0,
  );
  await panel().screenshot({ path: '../output/electron/verification/empty-project-dark.png' });
  await emptyAction('Open folder').click();
  await page.getByRole('dialog', { name: 'Open project folder', exact: true }).waitFor();
  await page.locator('.file-dialog-list[aria-busy="false"]').waitFor();
  await page.getByRole('button', { name: 'Select folder', exact: true }).click();
  await page
    .getByRole('dialog', { name: 'Open project folder', exact: true })
    .waitFor({ state: 'hidden' });
  assert.equal(openedProjects.at(-1), project);
  assert.equal(await panel().locator('.group-eyebrow').getAttribute('data-tooltip'), project);
  await panel().getByRole('textbox', { name: 'Search Project', exact: true }).waitFor();
  await page.getByRole('button', { name: 'Close project folder', exact: true }).click();
  await emptyAction('Open folder').waitFor();
  assert.equal(await panel().locator('.group-eyebrow').count(), 0);
  assert.equal(
    await panel().getByRole('textbox', { name: 'Search Project', exact: true }).count(),
    0,
  );
});

test('project search recovery restores files after no results', async () => {
  await openFolderFromEmpty();
  const search = page.getByRole('textbox', { name: 'Search Project', exact: true });
  await search.fill('nothing-matches-this');
  await panel().getByText('No matching files', { exact: true }).waitFor();
  await emptyAction('Clear search').click();
  assert.equal(await search.inputValue(), '');
  await panel().getByRole('treeitem', { name: 'arm.urdf', exact: true }).waitFor();
});

test('actuator empty state describes the scene source without a format-specific action', async () => {
  // A previously expanded scene search must not leak into a panel without search behavior.
  await panel().getByRole('button', { name: 'Search', exact: true }).click();
  await page.getByRole('button', { name: 'Actuators', exact: true }).click();
  await panel().getByText('No actuators defined', { exact: true }).waitFor();
  await panel().getByText('Actuators are provided by the scene source.', { exact: true }).waitFor();
  assert.equal(await panel().getByRole('textbox').count(), 0);
  assert.equal(await panel().getByRole('button', { name: 'Add object', exact: true }).count(), 0);
  assert.equal(await panel().getByRole('button', { name: 'Search', exact: true }).count(), 0);
  await panel().screenshot({ path: '../output/electron/verification/empty-actuators-dark.png' });
  const dialog = page.getByRole('dialog', { name: 'Open URDF', exact: true });
  assert.equal(await panel().getByRole('button', { name: 'Open URDF', exact: true }).count(), 0);
  await page.locator('[data-menu=File]').click();
  await page
    .locator('.menu-file')
    .getByRole('menuitem', { name: /^Open URDF/ })
    .click();
  await dialog.waitFor();
  await page.locator('.file-dialog-list[aria-busy="false"]').waitFor();
  await dialog.getByRole('option', { name: 'arm.urdf', exact: true }).click();
  await dialog.getByRole('button', { name: 'Open', exact: true }).click();
  await dialog.waitFor({ state: 'hidden' });
  await page.locator('.document-title').getByText('Empty state robot', { exact: true }).waitFor();
  assert.equal(openedUrdfs.at(-1), urdf);
});

test('joint and camera no-results actions restore their lists', async () => {
  for (const [tab, searchTitle, emptyTitle, card] of [
    ['Joints', 'Search joints', 'No matching joints', '.joint-entry'],
    ['Cameras', 'Search cameras', 'No matching cameras', '.camera-entry'],
  ]) {
    await page.getByRole('button', { name: tab, exact: true }).click();
    await panel().getByRole('button', { name: searchTitle, exact: true }).click();
    const search = panel().getByRole('textbox', { name: searchTitle, exact: true });
    const originalCount = await panel().locator(card).count();
    assert.ok(originalCount > 0);
    await search.fill('no-matching-name');
    await panel().getByText(emptyTitle, { exact: true }).waitFor();
    await emptyAction('Clear search').click();
    assert.equal(await search.inputValue(), '');
    assert.equal(await panel().locator(card).count(), originalCount);
  }
});

test('material empty guidance is centered inside the actual inspector at both widths', async () => {
  await page.getByRole('button', { name: 'Cameras', exact: true }).click();
  await panel()
    .locator('.camera-entry')
    .getByRole('button', { name: 'joint_gizmos', exact: true })
    .click();
  await page.locator('.inspector').getByRole('button', { name: 'Material', exact: true }).click();
  const empty = page.locator('.inspector .empty');
  await empty.getByText('No surface material', { exact: true }).waitFor();
  const inspector = page.locator('.inspector');
  const splitter = page.getByRole('separator', { name: 'Resize Inspector', exact: true });
  for (const width of [480, 280]) {
    const handle = await splitter.boundingBox(),
      previous = await inspector.boundingBox();
    await page.mouse.move(handle.x + handle.width / 2, handle.y + 100);
    await page.mouse.down();
    await page.mouse.move(handle.x + handle.width / 2 - (width - previous.width), handle.y + 100, {
      steps: 8,
    });
    await page.mouse.up();
    assert.ok(Math.abs((await inspector.boundingBox()).width - width) <= 1);
    const geometry = await empty.evaluate((element) => {
      const root = element.getBoundingClientRect();
      const detail = element.querySelector('.empty-detail');
      const bounds = detail.getBoundingClientRect();
      return {
        center: root.x + root.width / 2,
        detailCenter: bounds.x + bounds.width / 2,
        align: window.paintStyle(detail).textAlign,
      };
    });
    assert.equal(geometry.align, 'center');
    assert.ok(Math.abs(geometry.center - geometry.detailCenter) <= 1);
  }
  await page
    .locator('.inspector')
    .screenshot({ path: '../output/electron/verification/empty-material-centered.png' });
});
