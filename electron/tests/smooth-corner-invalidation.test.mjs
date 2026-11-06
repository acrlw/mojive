import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { chromium } from 'playwright';
import { createServer } from 'vite';

let server, browser, base;
before(async () => {
  server = await createServer({
    server: { host: '127.0.0.1', port: 0, strictPort: false, watch: null },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});

const fixture = `<!doctype html><html><head><style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; font: 14px sans-serif; }
  .surface { width: 240px; height: 100px; padding: 8px; margin: 0 0 16px;
    border-radius: var(--radius, 20px); background: #30363b; }
  .inner { width: 100%; height: 100%; padding: 12px;
    border-radius: calc(var(--radius, 20px) - 8px); background: #252a30; }
  .rail { width: 160px; height: 8px; border-radius: 4px; background: #454a50; }
  .fill { width: 32px; height: 8px; border-radius: 4px; background: #9fbf8f; }
  .narrow { width: 180px; }
  button { display: block; width: 100%; height: 28px; margin-top: 8px;
    border: 1px solid #556; border-radius: 6px; background: #556; }
  button:focus { border-width: 2px; }
  .portal { position: absolute; left: 320px; top: 24px; }
</style></head><body>
  <section id="a" class="surface"><div id="a-inner" class="inner">
    <div id="a-rail" class="rail"><div id="a-fill" class="fill"></div></div>
    <button id="a-button">Focus</button>
  </div></section>
  <section id="b" class="surface"><div id="b-inner" class="inner">
    <div id="b-rail" class="rail"><div id="b-fill" class="fill"></div></div>
  </div></section>
  <script type="module">
    import { setCornerSmoothing } from '/src/smooth-corners.ts';
    window.setCornerSmoothing = setCornerSmoothing;
    const original = Element.prototype.getBoundingClientRect;
    window.cornerReads = {};
    Element.prototype.getBoundingClientRect = function () {
      if (this.id) window.cornerReads[this.id] = (window.cornerReads[this.id] || 0) + 1;
      return original.call(this);
    };
  </script>
</body></html>`;

async function settle(page) {
  await page.evaluate(async () => {
    for (let index = 0; index < 5; index++)
      await new Promise((resolve) => requestAnimationFrame(resolve));
  });
}
async function snapshot(page) {
  return page.evaluate(() =>
    Object.fromEntries(
      [...document.querySelectorAll('[id][data-smooth-geometry]')].map((element) => [
        element.id,
        {
          radii: element.dataset.smoothRadii,
          reference: element.dataset.smoothReferences,
          amount: element.dataset.smoothAmount,
          paint: element.style.getPropertyValue('--smooth-paint'),
        },
      ]),
    ),
  );
}
async function matchesFullRebuild(page, smoothing, message) {
  await settle(page);
  const local = await snapshot(page);
  await page.evaluate((value) => {
    window.setCornerSmoothing(value === 0 ? 0.01 : 0);
    window.setCornerSmoothing(value);
  }, smoothing);
  await settle(page);
  assert.deepEqual(await snapshot(page), local, message);
}

for (const smoothing of [0, 0.6]) {
  test(`corner invalidation stays local and preserves nested geometry (${smoothing})`, async () => {
    const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
    try {
      await page.route('**/__corner-invalidation', (route) =>
        route.fulfill({ contentType: 'text/html', body: fixture }),
      );
      await page.goto(`${base}/__corner-invalidation`);
      await page.waitForFunction(() => !!window.setCornerSmoothing);
      await page.evaluate((value) => window.setCornerSmoothing(value), smoothing);
      await settle(page);
      await page.evaluate(() => {
        window.cornerReads = {};
        document.getElementById('a-fill').style.width = '96px';
      });
      await settle(page);
      const reads = await page.evaluate(() => window.cornerReads);
      assert.ok(reads.a > 0 && reads['a-fill'] > 0, JSON.stringify(reads));
      assert.ok(!Object.keys(reads).some((id) => id.startsWith('b')), JSON.stringify(reads));
      await matchesFullRebuild(page, smoothing, 'slider fill geometry differs from a full rebuild');

      await page.evaluate(() => document.getElementById('a').classList.add('narrow'));
      await matchesFullRebuild(
        page,
        smoothing,
        'resizing a rounded parent loses child constraints',
      );
      await page.locator('#a-button').focus();
      await matchesFullRebuild(page, smoothing, 'focus border changes lose nested constraints');

      await page.evaluate(() => {
        document.getElementById('a-inner').remove();
        const inner = document.createElement('div');
        inner.id = 'a-replacement';
        inner.className = 'inner';
        document.getElementById('a').append(inner);
      });
      await matchesFullRebuild(page, smoothing, 'replaced descendants retain stale constraints');
      assert.ok(!(await snapshot(page))['a-inner']);

      await page.evaluate(() => {
        const portal = document.createElement('section');
        portal.id = 'portal';
        portal.className = 'surface portal';
        portal.innerHTML = '<div id="portal-inner" class="inner"></div>';
        document.body.append(portal);
      });
      await matchesFullRebuild(page, smoothing, 'new portal surfaces miss nested constraints');
      assert.ok((await snapshot(page))['portal-inner']);

      await page.evaluate(() => {
        document.documentElement.style.setProperty('--radius', '24px');
        document.documentElement.style.fontSize = '16px';
      });
      await matchesFullRebuild(
        page,
        smoothing,
        'inherited root styles do not invalidate all surfaces',
      );
    } finally {
      await page.close();
    }
  });
}
