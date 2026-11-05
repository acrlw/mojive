import { installPaintStyle } from './helpers/surface-style.mjs';
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { mkdir } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser, page;
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    root: path.resolve('.'),
    cacheDir: 'node_modules/.vite-precise-test',
    server: { host: '127.0.0.1', port: 0, hmr: false },
    // The fixture imports scene catalogs after mounting. Prebundle their shared
    // dependencies before the page loads so late discovery cannot duplicate Three.
    optimizeDeps: {
      include: [
        'react',
        'react-dom/client',
        'react/jsx-runtime',
        'three',
        'i18next',
        'react-i18next',
      ],
    },
  });
  server.middlewares.use('/__precise_test', (_, res) => {
    res.setHeader('content-type', 'text/html');
    res.end(
      '<!doctype html><base href="/"><link rel="icon" href="data:,"><div id="fixture" style="position:relative;width:900px;height:600px"></div>',
    );
  });
  await server.listen();
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage({ viewport: { width: 1000, height: 700 } });
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__precise_test`);
  await page.evaluate(async () => {
    const { default: React } = await import('/node_modules/.vite-precise-test/deps/react.js'),
      { default: ReactDOM } = await import(
        '/node_modules/.vite-precise-test/deps/react-dom_client.js'
      ),
      { Viewport } = await import('/src/Viewport.tsx'),
      { createPrimitive } = await import('/src/scene.ts'),
      T = await import('/node_modules/three/build/three.module.js');
    const node = createPrimitive('box', 'Precise box'),
      doc = {
        name: 'test',
        root: new T.Group(),
        nodes: [node],
        joints: [],
        files: [],
        warnings: [],
      };
    doc.root.add(node.object);
    let revision = 0;
    window.fixture = {
      node,
      doc,
      commits: 0,
      begins: 0,
      tool: 'position',
      frame: 'world',
      api: React.createRef(),
      projectionChanges: [],
      stats: [],
    };
    const root = ReactDOM.createRoot(document.getElementById('fixture'));
    const render = () =>
      root.render(
        React.createElement(Viewport, {
          ref: window.fixture.api,
          layers: window.fixture.layers,
          onStats: (stats) => window.fixture.stats.push(stats),
          projection: window.fixture.projection,
          settings: window.fixture.settings,
          translate: window.fixture.translate,
          onProjectionChange: (mode) => window.fixture.projectionChanges.push(mode),
          document: window.fixture.doc,
          selected: window.fixture.selected === undefined ? node.id : window.fixture.selected,
          activeJoint: window.fixture.activeJoint,
          onActiveJoint: (id) => {
            window.fixture.activeJoint = id;
            render();
          },
          tool: window.fixture.tool,
          frame: window.fixture.frame,
          geometry: 'visual',
          revision: revision++,
          onSelect: (id) => {
            window.fixture.selected = id;
            render();
          },
          onChange: render,
          onBegin: () => window.fixture.begins++,
          onCommit: () => window.fixture.commits++,
        }),
      );
    window.fixture.render = render;
    render();
  });
  await page.waitForSelector('[data-testid="viewport-canvas"]');
  await page.evaluate(async () => {
    await import('/src/design-tokens.css');
    await import('/src/styles.css');
    await document.fonts.ready;
  });
  await mkdir('../output/electron/verification', { recursive: true });
  await page.waitForTimeout(150);
});
after(async () => {
  await browser?.close();
  await server?.close();
});
async function handle(color = 'red') {
  await page.mouse.move(980, 680);
  await page.waitForTimeout(80);
  return page.evaluate((color) => {
    const host = document.querySelector('.viewport-stage'),
      canvas = host.querySelectorAll('canvas')[1],
      ctx = canvas.getContext('2d'),
      data = ctx.getImageData(0, 0, canvas.width, canvas.height).data,
      rect = canvas.getBoundingClientRect();
    let best = null,
      dist = 0;
    for (let y = 0; y < canvas.height; y++)
      for (let x = 0; x < canvas.width; x++) {
        const i = (y * canvas.width + x) * 4,
          r = data[i],
          g = data[i + 1],
          b = data[i + 2],
          a = data[i + 3];
        const match =
          color === 'red'
            ? r >= 215 && r <= 225 && g >= 115 && g <= 123 && b >= 110 && b <= 120
            : r >= 105 && r <= 115 && g >= 143 && g <= 153 && b >= 224;
        if (a > 140 && match) {
          const d = (x - canvas.width / 2) ** 2 + (y - canvas.height / 2) ** 2;
          if (d > dist) {
            dist = d;
            best = {
              x: rect.x + (x / canvas.width) * rect.width,
              y: rect.y + (y / canvas.height) * rect.height,
            };
          }
        }
      }
    if (!best) throw new Error('No scalar handle found in production overlay');
    return best;
  }, color);
}
async function open(color) {
  const p = await handle(color);
  await page.mouse.dblclick(p.x, p.y, { delay: 80 });
  await page.getByRole('dialog').waitFor();
}
test('real handle double-click opens deferred numeric entry; Enter applies one relative position edit', async () => {
  await open('red');
  const before = await page.evaluate(() => ({
    x: fixture.node.object.position.x,
    commits: fixture.commits,
    begins: fixture.begins,
  }));
  await page.getByLabel('Gizmo value').fill('0.25');
  await page.screenshot({ path: '../output/electron/verification/precise-gizmo-input.png' });
  assert.equal(await page.evaluate(() => fixture.node.object.position.x), before.x);
  await page.getByLabel('Gizmo value').press('Enter');
  assert.ok(
    Math.abs((await page.evaluate(() => fixture.node.object.position.x)) - before.x - 0.25) < 1e-9,
  );
  assert.equal(await page.evaluate(() => fixture.commits), before.commits + 1);
  assert.equal(await page.evaluate(() => fixture.begins), before.begins + 1);
});
test('absolute entry conversion and Escape cancellation leave pose/history intact', async () => {
  await open('red');
  const before = await page.evaluate(() => ({
    x: fixture.node.object.position.x,
    commits: fixture.commits,
  }));
  await page.getByRole('button', { name: 'Absolute', exact: true }).click();
  assert.ok(Math.abs(Number(await page.getByLabel('Gizmo value').inputValue()) - before.x) < 1e-8);
  await page.getByLabel('Gizmo value').fill('123');
  await page.getByLabel('Gizmo value').press('Escape');
  assert.equal(await page.getByRole('dialog').count(), 0);
  assert.equal(await page.evaluate(() => fixture.node.object.position.x), before.x);
  assert.equal(await page.evaluate(() => fixture.commits), before.commits);
});
test('rotation numeric unit switch preserves amount and accepts radians', async () => {
  await page.evaluate(() => {
    localStorage.clear();
    fixture.tool = 'rotation';
    fixture.render();
  });
  await open('blue');
  const initial = await page.evaluate(() => fixture.node.object.quaternion.toArray());
  await page.getByLabel('Gizmo value').fill('90');
  await page.getByLabel('Gizmo value').press('u');
  assert.ok(
    Math.abs(Number(await page.getByLabel('Gizmo value').inputValue()) - Math.PI / 2) < 1e-9,
  );
  await page.getByLabel('Gizmo value').press('Enter');
  const q = await page.evaluate(() => fixture.node.object.quaternion.toArray());
  const dot = q.reduce((sum, value, index) => sum + value * initial[index], 0);
  const angle = 2 * Math.acos(Math.min(1, Math.abs(dot)));
  assert.ok(Math.abs(angle - Math.PI / 2) < 1e-8, JSON.stringify({ q, initial, angle }));
});

test('projection commands are idempotent and controlled state applies native projection transitions', async () => {
  await page.evaluate(() => {
    window.fixture.api.current.setProjection('orthographic');
    window.fixture.api.current.setProjection('orthographic');
  });
  await page.waitForTimeout(340);
  const first = await page.evaluate(() => ({
    changes: window.fixture.projectionChanges,
    image: window.fixture.api.current.capture(),
  }));
  assert.deepEqual(first.changes, ['orthographic']);
  await page.evaluate(() => window.fixture.api.current.view('orthographic'));
  await page.waitForTimeout(50);
  const second = await page.evaluate(() => ({
    changes: window.fixture.projectionChanges,
    image: window.fixture.api.current.capture(),
  }));
  assert.deepEqual(second.changes, ['orthographic']);
  assert.equal(second.image, first.image);
  await page.evaluate(() => {
    window.fixture.projection = 'perspective';
    window.fixture.render();
  });
  await page.waitForTimeout(340);
  assert.deepEqual(await page.evaluate(() => window.fixture.projectionChanges), [
    'orthographic',
    'perspective',
  ]);
});
test('viewport display settings hide the live view cube and preserve authored camera framing', async () => {
  const before = await page.evaluate(() => window.fixture.api.current.capture());
  await page.evaluate(() => {
    window.fixture.settings = {
      showGrid: false,
      showViewCube: false,
      selectionOutline: false,
      selectionHighlight: false,
    };
    window.fixture.render();
  });
  await page.waitForTimeout(70);
  assert.equal(await page.locator('[data-testid="view-cube"]').isVisible(), false);
  assert.notEqual(await page.evaluate(() => window.fixture.api.current.capture()), before);
});

test('viewport layers hide actual geometry, grid, cube and gizmos; object display flags independently gate handles', async () => {
  await page.evaluate(() => {
    const f = window.fixture;
    f.settings = {};
    f.tool = 'position';
    f.layers = {
      geometry: true,
      grid: true,
      helpers: true,
      viewCube: true,
      gizmos: true,
      selection: true,
    };
    f.render();
  });
  await page.waitForTimeout(80);
  const ink = () =>
    page.evaluate(() => {
      const canvas = document.querySelector('.viewport-stage').querySelectorAll('canvas')[1],
        pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let count = 0;
      for (let i = 3; i < pixels.length; i += 4) if (pixels[i]) count++;
      return count;
    });
  assert.ok((await ink()) > 50);
  await page.evaluate(() => {
    window.fixture.layers.gizmos = false;
    window.fixture.render();
  });
  await page.waitForTimeout(60);
  assert.equal(await ink(), 0);
  await page.evaluate(() => {
    const f = window.fixture;
    f.layers.gizmos = true;
    f.node.object.userData.gizmoEnabled = false;
    f.render();
  });
  await page.waitForTimeout(60);
  assert.equal(await ink(), 0);
  await page.evaluate(() => {
    window.fixture.node.object.userData.gizmoEnabled = true;
    window.fixture.render();
  });
  await page.waitForTimeout(60);
  assert.ok((await ink()) > 50);
  await page.evaluate(() => {
    const f = window.fixture;
    for (const key of Object.keys(f.layers)) f.layers[key] = false;
    f.render();
  });
  await page.waitForTimeout(80);
  assert.equal(await ink(), 0);
  assert.equal(await page.locator('[data-testid="view-cube"]').isVisible(), false);
  const pixel = await page.evaluate(() => {
    const source = document.querySelector('[data-testid="viewport-canvas"]'),
      canvas = document.createElement('canvas');
    canvas.width = source.width;
    canvas.height = source.height;
    const c = canvas.getContext('2d');
    c.drawImage(source, 0, 0);
    return Array.from(c.getImageData(canvas.width / 2, canvas.height / 2, 1, 1).data);
  });
  assert.deepEqual(pixel, [34, 39, 45, 255]);
});
test('live viewport telemetry reports actual pass workload and becomes idle without triggering extra renders', async () => {
  await page.evaluate(async () => {
    for (let i = 0; i < 16; i++) {
      window.fixture.render();
      await new Promise(requestAnimationFrame);
    }
  });
  await page.waitForTimeout(280);
  const active = await page.evaluate(() => window.fixture.stats.at(-1));
  assert.ok(active.fps > 0);
  assert.equal(active.idle, false);
  assert.equal(active.backend, 'Three.js · WebGL 2');
  assert.ok(active.frameMs >= 0);
  assert.equal(active.drawCalls, 0);
  assert.equal(active.triangles, 0);
  await page.waitForTimeout(800);
  const idle = await page.evaluate(() => window.fixture.stats.at(-1));
  assert.equal(idle.idle, true);
  assert.equal(idle.fps, 0);
  assert.equal(idle.drawCalls, 0);
});

for (const theme of ['light', 'dark'])
  test(`precise popup uses shared ${theme} colors and nested continuous corners`, async () => {
    await page.evaluate(async (theme) => {
      const { applySettings, DEFAULT_SETTINGS } = await import('/src/settings.tsx');
      applySettings({ ...DEFAULT_SETTINGS, appearance: theme, language: 'en' });
      const f = window.fixture;
      f.tool = 'rotation';
      f.layers = undefined;
      f.settings = {};
      f.render();
    }, theme);
    await open('blue');
    await page.getByLabel('Gizmo value').fill('45');
    await page.getByRole('button', { name: 'Relative', exact: true }).click();
    await page.getByLabel('Gizmo value').focus();
    const styles = await page.evaluate(() => {
      const field = document.querySelector('.precise-gizmo-number input'),
        segment = document.querySelector('.precise-gizmo-segments button.active'),
        panel = document.querySelector('.precise-gizmo-input'),
        root = window.paintStyle(document.documentElement);
      return {
        field: window.paintStyle(field).backgroundColor,
        segment: window.paintStyle(segment).backgroundColor,
        ink: window.paintStyle(field).color,
        outline: window.paintStyle(field).outlineWidth,
        border: window.paintStyle(field).borderWidth,
        paths: [...panel.querySelectorAll('input,.precise-gizmo-segments,button'), panel].map(
          (e) => window.paintStyle(e).clipPath,
        ),
        rect: panel.getBoundingClientRect().toJSON(),
        segments: [...panel.querySelectorAll('.precise-gizmo-segments')].map((group) => ({
          padding: window.paintStyle(group).padding,
          gap: window.paintStyle(group).gap,
          frame: group.getBoundingClientRect().toJSON(),
          buttons: [...group.children].map((button) => ({
            rect: button.getBoundingClientRect().toJSON(),
            corners: [
              'borderTopLeftRadius',
              'borderTopRightRadius',
              'borderBottomRightRadius',
              'borderBottomLeftRadius',
            ]
              .map((key) => parseFloat(window.paintStyle(button)[key]))
              .join(','),
          })),
        })),
      };
    });
    assert.equal(styles.field, theme === 'light' ? 'rgb(201, 212, 194)' : 'rgb(64, 71, 78)');
    assert.equal(styles.segment, styles.field);
    assert.equal(styles.ink, theme === 'light' ? 'rgb(36, 44, 35)' : 'rgb(230, 232, 235)');
    assert.equal(styles.outline, '0px');
    assert.equal(styles.border, '0px');
    assert.ok(styles.paths.every((p) => p === 'none'));
    for (const segment of styles.segments) {
      assert.equal(segment.padding, '0px');
      assert.equal(segment.gap, '0px');
      assert.equal(segment.buttons[0].rect.x, segment.frame.x);
      assert.equal(segment.buttons[1].rect.right, segment.frame.right);
      assert.equal(segment.buttons[0].rect.right, segment.buttons[1].rect.x);
      assert.equal(segment.buttons[0].corners, '4,0,0,4');
      assert.equal(segment.buttons[1].corners, '0,4,4,0');
    }
    assert.ok(styles.rect.x >= 0 && styles.rect.right <= 1000 && styles.rect.bottom <= 700);
    await page.screenshot({ path: `../output/electron/verification/precise-popup-${theme}.png` });
    const before = await page.evaluate(() => fixture.node.object.quaternion.toArray());
    await page.getByLabel('Gizmo value').press('Escape');
    assert.deepEqual(await page.evaluate(() => fixture.node.object.quaternion.toArray()), before);
  });

test('native multi-joint picker is contextual, movable and preserves explicit choice after dismissal', async () => {
  const ids = await page.evaluate(async () => {
    const { createNativeJointScene } = await import('/src/SceneCatalog.ts');
    const doc = createNativeJointScene(await (await fetch('/demo/joint_gizmo.xml')).text());
    const joints = doc.joints.filter((j) => j.name.startsWith('05_multi'));
    const body = doc.nodes.find((n) => n.id === joints[0].bodyId);
    const f = window.fixture;
    f.doc = doc;
    f.selected = body.id;
    f.tool = 'joint';
    f.layers = undefined;
    f.render();
    return {
      body: body.id,
      joints: joints.map((j) => ({ id: j.id, name: j.name })),
      single: doc.joints.find((j) => j.name === '02_prismatic_x').id,
    };
  });
  const picker = page.getByRole('dialog', { name: 'Choose joint' });
  await picker.waitFor();
  assert.equal(await picker.locator('.viewport-joint-picker-row').count(), 3);
  const initial = await picker.boundingBox(),
    host = await page.locator('.viewport-stage').boundingBox();
  assert.equal(initial.x - host.x, 18);
  assert.equal(initial.y - host.y, 18);
  for (const joint of ids.joints) {
    await picker.getByRole('button', { name: new RegExp(joint.name) }).click();
    assert.equal(await page.evaluate(() => fixture.activeJoint), joint.id);
    assert.equal(await page.evaluate(() => fixture.selected), ids.body);
  }
  const heading = await picker.locator('.viewport-joint-picker-heading strong').boundingBox();
  await page.mouse.move(heading.x + 10, heading.y + 10);
  await page.mouse.down();
  await page.mouse.move(heading.x + 105, heading.y + 70, { steps: 5 });
  await page.mouse.up();
  const moved = await picker.boundingBox();
  assert.ok(moved.x > initial.x + 90 && moved.y > initial.y + 55);
  await page.screenshot({ path: '../output/electron/verification/contextual-joint-picker.png' });
  await picker.getByRole('button', { name: 'Close', exact: true }).click();
  await page.evaluate(() => fixture.render());
  await page.waitForTimeout(50);
  assert.equal(await picker.count(), 0);
  assert.equal(await page.evaluate(() => fixture.activeJoint), ids.joints[2].id);
  await page.evaluate((id) => {
    fixture.selected = id;
    fixture.render();
  }, ids.single);
  await page.waitForTimeout(40);
  await page.evaluate((id) => {
    fixture.selected = id;
    fixture.render();
  }, ids.body);
  await picker.waitFor();
  assert.equal(
    await picker
      .getByRole('button', { name: new RegExp(ids.joints[2].name) })
      .getAttribute('aria-pressed'),
    'true',
  );
  await picker.getByRole('button', { name: new RegExp(ids.joints[0].name) }).dblclick();
  assert.equal(await picker.count(), 0);
  assert.equal(await page.evaluate(() => fixture.activeJoint), ids.joints[0].id);
  await page.waitForTimeout(400);
  await page.evaluate(() => {
    fixture.selected = null;
    fixture.tool = 'select';
    fixture.render();
  });
  await page.waitForTimeout(50);
  const bounds = await page.locator('.viewport-stage').boundingBox();
  await page.mouse.click(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  const clicked = await page.evaluate(() => fixture.selected);
  if (!clicked)
    await page.screenshot({
      path: '../output/electron/verification/precise-scene-pick-failure.png',
    });
  assert.ok(clicked, 'Scene click selects the focused compound body geometry');
  await page.evaluate(() => {
    fixture.tool = 'joint';
    fixture.render();
  });
  await picker.waitFor();
  const context = await picker.boundingBox();
  assert.equal(context.x - host.x, bounds.width / 2 + 14);
  assert.equal(context.y - host.y, bounds.height / 2 + 14);
  await page.screenshot({
    path: '../output/electron/verification/contextual-joint-picker-click.png',
  });
});

for (const theme of ['light', 'dark'])
  test(`native rich ball scene exposes complete trackball during real ${theme} pointer drag`, async () => {
    await page.evaluate(async (theme) => {
      const { applySettings, DEFAULT_SETTINGS } = await import('/src/settings.tsx');
      applySettings({ ...DEFAULT_SETTINGS, appearance: theme, language: 'zh' });
      const f = window.fixture,
        joint = f.doc.joints.find((j) => j.name === '03_ball_xyz');
      f.translate = (await import('/src/i18n.ts')).t;
      f.selected = joint.id;
      f.tool = 'joint';
      f.settings = {};
      f.render();
      f.api.current.focus(joint.id);
    }, theme);
    await page.waitForTimeout(450);
    const center = await page.evaluate(() => {
      const canvas = document.querySelector('.viewport-stage').querySelectorAll('canvas')[1],
        rect = canvas.getBoundingClientRect(),
        pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let x0 = Infinity,
        x1 = -Infinity,
        y0 = Infinity,
        y1 = -Infinity;
      for (let y = 0; y < canvas.height; y++)
        for (let x = 0; x < canvas.width; x++) {
          const i = (y * canvas.width + x) * 4;
          if (
            pixels[i + 3] >= 8 &&
            pixels[i + 3] <= 10 &&
            Math.abs(pixels[i] - pixels[i + 1]) < 3 &&
            Math.abs(pixels[i + 1] - pixels[i + 2]) < 3
          ) {
            x0 = Math.min(x0, x);
            x1 = Math.max(x1, x);
            y0 = Math.min(y0, y);
            y1 = Math.max(y1, y);
          }
        }
      return {
        x: rect.x + (((x0 + x1) / 2) * rect.width) / canvas.width,
        y: rect.y + (((y0 + y1) / 2) * rect.height) / canvas.height,
      };
    });
    assert.ok(Number.isFinite(center.x) && Number.isFinite(center.y));
    const before = await page.evaluate(() =>
      fixture.doc.joints.find((j) => j.name === '03_ball_xyz').rotation.toArray(),
    );
    await page.mouse.move(center.x, center.y);
    await page.mouse.down();
    await page.mouse.move(center.x - 90, center.y - 19, { steps: 8 });
    assert.notDeepEqual(
      await page.evaluate(() =>
        fixture.doc.joints.find((j) => j.name === '03_ball_xyz').rotation.toArray(),
      ),
      before,
    );
    await page.screenshot({ path: `../output/electron/verification/trackball-rich-${theme}.png` });
    await page.keyboard.press('Escape');
    await page.mouse.up();
    assert.deepEqual(
      await page.evaluate(() =>
        fixture.doc.joints.find((j) => j.name === '03_ball_xyz').rotation.toArray(),
      ),
      before,
    );
  });
