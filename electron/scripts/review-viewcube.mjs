import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createServer } from 'vite';
const run = promisify(execFile),
  root = fileURLToPath(new URL('../', import.meta.url)),
  repository = fileURLToPath(new URL('../../', import.meta.url));
const python = async (script) => {
  const { stdout, stderr } = await run(
    'uv',
    ['run', '--no-sync', 'python', fileURLToPath(new URL(script, import.meta.url))],
    { cwd: repository },
  );
  if (stdout) process.stdout.write(stdout);
  if (stderr) process.stderr.write(stderr);
};
const output = new URL('../../output/electron/verification/', import.meta.url);
await mkdir(output, { recursive: true });
await python('./capture-native-viewcube.py');
const { dpr } = JSON.parse(await readFile(new URL('viewcube-native-capture.json', output), 'utf8'));
const server = await createServer({
  root,
  server: { host: '127.0.0.1', port: 0, strictPort: false },
  logLevel: 'error',
});
let browser;
try {
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage({
    viewport: { width: 900, height: 180 },
    deviceScaleFactor: dpr,
  });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/tests/harness/viewcube-review.html`);
  await page.waitForFunction(() => window.ready);
  await page.screenshot({ path: new URL('viewcube-web-gallery.png', output).pathname });
  for (const name of ['oblique', 'front', 'top', 'back-fade', 'neck-limit'])
    await page
      .locator(`[data-view="${name}"]`)
      .screenshot({ path: new URL(`viewcube-web-${name}.png`, output).pathname });
  // Center in an oblique view belongs to projection; aligned center belongs to nearest axis.
  await page.mouse.click(90, 90);
  assert.deepEqual(await page.evaluate(() => window.events), [['projection']]);
  await page.mouse.click(270, 90);
  assert.deepEqual((await page.evaluate(() => window.events)).at(-1), ['axis', 1, -1]);
  await page.evaluate(() => (window.events = []));
  await page.mouse.move(90, 90);
  await page.mouse.down();
  await page.mouse.move(100, 90);
  await page.mouse.up();
  const drag = await page.evaluate(() => window.events);
  assert.ok(drag.some((e) => e[0] === 'orbit'));
  assert.ok(!drag.some((e) => e[0] === 'projection'));
  // Backdrop is visual only and cannot acquire a widget gesture.
  await page.evaluate(() => (window.events = []));
  await page.mouse.click(120, 115);
  assert.deepEqual(await page.evaluate(() => window.events), []);
  await page.mouse.move(90, 90);
  await page.mouse.down();
  await page.locator('canvas').first().dispatchEvent('pointercancel');
  await page.mouse.up();
  assert.deepEqual(await page.evaluate(() => window.events), []);
  assert.deepEqual(errors, []);
  await writeFile(
    new URL('viewcube-interactions.json', output),
    JSON.stringify(
      {
        dpr,
        passed: [
          'origin projection click',
          'aligned nearest-axis ownership',
          'held drag without projection toggle',
          'transparent backdrop has no widget ownership',
          'pointer cancellation',
        ],
        errors,
      },
      null,
      2,
    ),
  );
  await python('./compare-viewcube.py');
  console.log(
    'ViewCube native/browser captures, raster comparison and 5 interaction checks passed.',
  );
} finally {
  await browser?.close();
  await server.close();
}
