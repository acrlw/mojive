import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, symlink, rm } from 'node:fs/promises';
import path from 'node:path';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { routeReviewResources } from '../scripts/review-resources.mjs';

test('model review transport preserves reserved filename characters and binary bytes', async (t) => {
  const output = path.resolve('../output/electron');
  await mkdir(output, { recursive: true });
  const directory = await mkdtemp(path.join(output, 'review-resources-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const assets = path.join(directory, 'assets');
  await mkdir(assets);
  const names = ['Asm-MRH-HB1-Visual,00-Plastic.stl', 'mesh #1? 50%.stl', 'mesh%2Cname.stl', '模型.stl'];
  const bytes = Buffer.from([0, 255, 128, 13, 10, 42]);
  for (const name of names) await writeFile(path.join(assets, name), bytes);
  await writeFile(path.join(directory, 'outside.stl'), bytes);
  await symlink(path.join(directory, 'outside.stl'), path.join(assets, 'outside-link.stl'));
  const server = await createServer({ configFile: false, root: path.resolve('.'),
    optimizeDeps: { noDiscovery: true, include: [] }, server: { host: '127.0.0.1', port: 0 } });
  await server.listen();
  t.after(() => server.close());
  const browser = await chromium.launch({ headless: true });
  t.after(() => browser.close());
  const page = await browser.newPage();
  await routeReviewResources(page, assets);
  await page.route('**/__review', (route) => route.fulfill({ contentType: 'text/html', body: '<html></html>' }));
  await page.goto(server.resolvedUrls.local[0] + '__review');
  const results = await page.evaluate(async (names) => {
    const results = [];
    for (const name of names) {
      const response = await fetch('/__model_resource?path=' + encodeURIComponent(name));
      results.push({ status: response.status, bytes: Array.from(new Uint8Array(await response.arrayBuffer())) });
    }
    return results;
  }, [...names, 'missing.stl', '../outside.stl', 'outside-link.stl']);
  for (const result of results.slice(0, names.length)) {
    assert.equal(result.status, 200);
    assert.deepEqual(result.bytes, Array.from(bytes));
  }
  assert.deepEqual(results.slice(names.length).map((result) => result.status), [404, 403, 403]);
});
