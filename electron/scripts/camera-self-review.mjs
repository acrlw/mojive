import assert from 'node:assert/strict';
import { _electron } from 'playwright';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url)),
  out = path.join(root, '../output/electron/verification'),
  hash = (text) => createHash('sha256').update(text).digest('hex');
await mkdir(out, { recursive: true });
const profile = await mkdtemp(path.join(out, 'camera-self-review-profile-')),
  env = { ...process.env };
delete env.MOJIVE_DEV_URL;
const report = {
  application: 'Production Electron App',
  indexSha256: hash(await readFile(path.join(root, 'dist/index.html'))),
  states: [],
  errors: [],
};
let app,
  page,
  phase = 'launch';
try {
  app = await _electron.launch({
    args: [root, '--hidden', '--user-data-dir=' + profile],
    cwd: root,
    env,
    timeout: 45000,
  });
  page = await app.firstWindow();
  page.setDefaultTimeout(15000);
  page.on('pageerror', (error) => report.errors.push({ phase, error: error.stack }));
  await app.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows()[0].setContentSize(1540, 980),
  );
  await page.locator('.tree-row').first().waitFor();
  assert.equal(
    new URL(page.url()).protocol,
    'mojive-app:',
    'Review must load the production bundle',
  );
  report.url = page.url();
  await page.evaluate(() => {
    localStorage.setItem(
      'mojive.electron.settings.v1',
      JSON.stringify({ language: 'en', appearance: 'dark', uiScale: 1 }),
    );
    localStorage.removeItem('mojive.layout');
  });
  await page.reload();
  await page.locator('.tree-row').first().waitFor();
  await page.evaluate(() => document.fonts.ready);
  const action = async (id) => {
    await app.evaluate(({ Menu, BrowserWindow }, id) => {
      const item = Menu.getApplicationMenu().getMenuItemById(id);
      if (!item || !item.enabled) throw Error(`Unavailable native menu action: ${id}`);
      item.click(item, BrowserWindow.getAllWindows()[0], {});
    }, id);
  };
  const settle = async () => {
    await page.mouse.move(20, 20);
    await page.waitForTimeout(500);
  };
  phase = 'load primitive scene and camera';
  await action('scene.primitive_gallery');
  await page.waitForFunction(() =>
    document.querySelector('.document-title')?.textContent.includes('Scene objects'),
  );
  await page.locator('.tree-row.type-camera').first().click();
  report.camera = (await page.locator('.inspector-title').count())
    ? await page.locator('.inspector-title').innerText()
    : await page.locator('.tree-row.type-camera.selected').getAttribute('title');
  await page.locator('.tool-palette').getByTitle('Rotation (E)', { exact: true }).click();
  await action('view.focusSelection');
  await settle();
  const inspector = page.locator('.inspector'),
    previewToggle = inspector.getByRole('switch', { name: 'Preview', exact: true });
  await previewToggle.click();
  await page.locator('.camera-preview-image').waitFor();
  await page.waitForFunction(
    () => Number(document.querySelector('.camera-preview-image')?.dataset.renderCount) > 0,
  );
  await settle();

  const capture = async (state) => {
    phase = state;
    await settle();
    assert.equal(await previewToggle.getAttribute('aria-checked'), 'true');
    assert.equal(
      await page
        .locator('.tool-palette')
        .getByTitle('Rotation (E)', { exact: true })
        .evaluate((node) => node.classList.contains('active')),
      true,
    );
    const pixels = await page.getByTestId('viewport-canvas').evaluate(async (canvas) => {
      const overlay = [...canvas.parentElement.children].find(
        (node) => node.tagName === 'CANVAS' && !node.dataset.testid,
      );
      const rgba = overlay.getContext('2d').getImageData(0, 0, overlay.width, overlay.height).data;
      let overlayInk = 0;
      for (let i = 3; i < rgba.length; i += 4) if (rgba[i]) overlayInk++;
      const preview = document.querySelector('.camera-preview-image'),
        previewData = preview.toDataURL(),
        image = new Image();
      image.src = previewData;
      await image.decode();
      const copy = document.createElement('canvas');
      copy.width = image.width;
      copy.height = image.height;
      const context = copy.getContext('2d');
      context.drawImage(image, 0, 0);
      const pixels = context.getImageData(0, 0, copy.width, copy.height).data,
        colors = new Set();
      for (let i = 0; i < pixels.length; i += 4)
        colors.add(`${pixels[i]},${pixels[i + 1]},${pixels[i + 2]},${pixels[i + 3]}`);
      return {
        overlayInk,
        overlayData: overlay.toDataURL(),
        previewData,
        preview: {
          width: copy.width,
          height: copy.height,
          colors: colors.size,
          renderCount: Number(preview.dataset.renderCount),
        },
        selected: document.querySelector('.tree-row.selected')?.getAttribute('title'),
        transform: [
          ...document.querySelectorAll('.inspector-content > .section:first-child input'),
        ].map((input) => input.value),
        sceneCamera: document.querySelector('.scene-camera-return')?.textContent ?? null,
      };
    });
    assert.ok(
      pixels.preview.width > 100 &&
        pixels.preview.height > 50 &&
        pixels.preview.colors > 32 &&
        pixels.preview.renderCount > 0,
      JSON.stringify(pixels.preview),
    );
    const { overlayData, previewData, ...metrics } = pixels;
    report.states.push({
      state,
      ...metrics,
      overlaySha256: hash(overlayData),
      previewSha256: hash(previewData),
    });
    await page.screenshot({ path: path.join(out, `camera-self-electron-${state}.png`) });
    return pixels;
  };
  const editor = await capture('editor');
  assert.ok(editor.overlayInk > 300, 'Selected camera has its rotation handles in Editor');
  phase = 'enter View';
  await inspector.getByRole('button', { name: 'View Camera', exact: true }).click();
  await page.locator('.scene-camera-return').waitFor();
  const view = await capture('view');
  assert.equal(view.overlayInk, 0, 'Active camera must not show its own rotation handles in View');
  assert.equal(view.selected, editor.selected);
  assert.equal(
    view.previewData,
    editor.previewData,
    'View does not affect the separate camera preview',
  );
  assert.deepEqual(view.transform, editor.transform);
  phase = 'switch to Pilot';
  await inspector.getByRole('button', { name: 'Pilot camera', exact: true }).click();
  const pilot = await capture('pilot');
  assert.equal(
    pilot.overlayInk,
    0,
    'Active camera must not show its own rotation handles in Pilot',
  );
  assert.equal(pilot.selected, editor.selected);
  assert.equal(pilot.previewData, editor.previewData);
  assert.deepEqual(pilot.transform, editor.transform);
  phase = 'return Editor';
  await page.locator('.scene-camera-return button').click();
  await page.locator('.scene-camera-return').waitFor({ state: 'detached' });
  const returned = await capture('returned');
  assert.equal(
    returned.overlayData,
    editor.overlayData,
    'Returning to Editor restores exactly the original rotation overlay',
  );
  assert.equal(returned.previewData, editor.previewData);
  assert.deepEqual(returned.transform, editor.transform);
  assert.deepEqual(report.errors, []);
  report.passed = true;
  console.log(
    JSON.stringify(
      {
        passed: true,
        camera: report.camera,
        states: report.states.map(({ state, overlayInk, preview }) => ({
          state,
          overlayInk,
          preview,
        })),
        errors: report.errors,
      },
      null,
      2,
    ),
  );
} catch (error) {
  report.failure = { phase, message: String(error), stack: error.stack };
  await page?.screenshot({ path: path.join(out, 'camera-self-electron-failure.png') });
  throw error;
} finally {
  if (app) {
    await app.evaluate(({ BrowserWindow }) =>
      BrowserWindow.getAllWindows().forEach((window) => window.destroy()),
    );
    await app.close();
  }
  await rm(profile, { recursive: true, force: true });
  await writeFile(
    path.join(out, 'camera-self-electron.json'),
    JSON.stringify(report, null, 2) + '\n',
  );
}
