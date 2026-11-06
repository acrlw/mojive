import { openEditableJointScene } from './helpers/editable-scene.mjs';
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser, page, base;
const errors = [];
before(async () => {
  server = await createServer({
    configFile: false,
    cacheDir: 'node_modules/.vite-selection-ownership',
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
    server: { host: '127.0.0.1', port: 0, hmr: false },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  page.on('pageerror', (e) => errors.push(e.message));
  await mkdir('../output/electron/verification', { recursive: true });
  await page.goto(base + '/tests/harness/selection-review.html');
  await page.waitForFunction(() => window.selectionReview?.api.current);
});
after(async () => {
  await browser?.close();
  await server?.close();
});

test('selection owner follows geom/body/joint identities and never crosses into a child body or model container', async () => {
  const result = await page.evaluate(async () => {
    const { createNativeJointScene } = await import('/src/SceneCatalog.ts'),
      { selectionGeometry, selectionGeometryOwner, setViewportPickable } = await import(
        '/src/scene-selection.ts'
      ),
      doc = createNativeJointScene(await (await fetch('/demo/joint_gizmo.xml')).text()),
      selected = (id) => [...selectionGeometry(doc, id)].map((mesh) => mesh.userData.nodeId).sort(),
      anchor = doc.nodes.find((n) => n.name === '01_revolute_anchor'),
      child = doc.nodes.find((n) => n.name === '01_revolute'),
      joint = doc.joints.find((j) => j.name === '01_revolute_y'),
      own = doc.nodes.filter((n) => n.parentId === anchor.id && n.kind === 'geom'),
      childGeoms = doc.nodes.filter((n) => n.parentId === child.id && n.kind === 'geom');
    const before = {
      anchor: selected(anchor.id),
      child: selected(child.id),
      joint: selected(joint.id),
      geom: selected(childGeoms[0].id),
      model: selected('robot'),
      world: selected('world'),
      jointOwner: selectionGeometryOwner(doc, joint.id)?.id,
    };
    setViewportPickable(childGeoms[0].object, false);
    const partial = selected(joint.id);
    setViewportPickable(anchor.object, false);
    setViewportPickable(child.object, true);
    const disabled = selected(joint.id);
    return {
      before,
      partial,
      disabled,
      own: own.map((n) => n.id).sort(),
      childGeoms: childGeoms.map((n) => n.id).sort(),
      child: child.id,
    };
  });
  assert.deepEqual(result.before.anchor, result.own);
  assert.deepEqual(result.before.child, result.childGeoms);
  assert.deepEqual(result.before.joint, result.childGeoms);
  assert.equal(result.before.jointOwner, result.child);
  assert.equal(result.before.geom.length, 1);
  assert.deepEqual(result.before.model, []);
  assert.deepEqual(result.before.world, []);
  assert.equal(result.partial.length, result.childGeoms.length - 1);
  assert.deepEqual(result.disabled, []);
});

test('actual shader masks and fill match one owner across all four geometry modes', async () => {
  const results = await page.evaluate(async () => {
    const T = window.selectionReview.THREE,
      { ViewportPipeline } = await import('/src/viewport/pipeline.ts'),
      { setViewportPickable } = await import('/src/scene-selection.ts');
    const renderer = new T.WebGLRenderer({ preserveDrawingBuffer: true, antialias: true });
    renderer.setSize(360, 160);
    const pipeline = new ViewportPipeline(renderer);
    pipeline.resize(360, 160);
    const scene = new T.Scene(),
      root = new T.Group(),
      model = new T.Group(),
      a = new T.Group(),
      j = new T.Group(),
      b = new T.Group(),
      nodes = [];
    scene.add(root);
    root.add(model);
    model.add(a);
    a.add(j);
    j.add(b);
    nodes.push(
      { id: 'world', kind: 'world', object: root },
      { id: 'model', kind: 'robot', object: model },
      { id: 'a', kind: 'link', object: a },
      { id: 'j', kind: 'joint', object: j },
      { id: 'b', name: 'b', kind: 'link', object: b },
    );
    const add = (id, parent, x, role) => {
      const group = new T.Group(),
        mesh = new T.Mesh(
          new T.BoxGeometry(0.9, 0.9, 0.9),
          new T.MeshStandardMaterial({ color: '#7e9db9' }),
        );
      group.position.x = x;
      mesh.userData = { nodeId: id, role };
      group.userData = { role, roleGroup: true };
      group.visible = role === 'visual';
      group.add(mesh);
      parent.add(group);
      nodes.push({ id, kind: 'geom', object: group });
      return group;
    };
    const av = add('av', a, -1.5, 'visual'),
      ac = add('ac', a, -1.5, 'collision'),
      bv = add('bv', b, 1.5, 'visual'),
      bc = add('bc', b, 1.5, 'collision');
    const doc = { root, nodes, joints: [{ id: 'j', bodyId: 'b', child: 'b' }] },
      camera = new T.OrthographicCamera(-3, 3, 1.33, -1.33, 0.01, 100);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    scene.add(new T.AmbientLight('white', 2));
    const gl = renderer.getContext(),
      render = (selected, mode) => {
        pipeline.render(scene, camera, doc, selected, mode);
        const p = new Uint8Array(360 * 160 * 4);
        gl.readPixels(0, 0, 360, 160, gl.RGBA, gl.UNSIGNED_BYTE, p);
        return p;
      },
      count = (a, b, lo, hi) => {
        let changed = 0;
        for (let y = 0; y < 160; y++)
          for (let x = lo; x < hi; x++) {
            const i = (y * 360 + x) * 4;
            if (a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2]) changed++;
          }
        return changed;
      },
      results = [];
    for (const mode of ['default', 'visual', 'collision', 'both']) {
      const base = render(null, mode),
        parent = render('a', mode),
        child = render('b', mode),
        joint = render('j', mode),
        model = render('model', mode),
        world = render('world', mode);
      setViewportPickable(av, false);
      setViewportPickable(ac, false);
      const disabled = render('a', mode);
      setViewportPickable(av, true);
      setViewportPickable(ac, true);
      results.push({
        mode,
        parent: [count(base, parent, 0, 180), count(base, parent, 180, 360)],
        child: [count(base, child, 0, 180), count(base, child, 180, 360)],
        jointDifference: count(child, joint, 0, 360),
        model: count(base, model, 0, 360),
        world: count(base, world, 0, 360),
        disabled: count(base, disabled, 0, 360),
      });
    }
    pipeline.dispose();
    renderer.dispose();
    return results;
  });
  for (const r of results) {
    assert.ok(r.parent[0] > 1000, JSON.stringify(r));
    assert.equal(r.parent[1], 0);
    assert.equal(r.child[0], 0);
    assert.ok(r.child[1] > 1000);
    assert.equal(r.jointDifference, 0);
    assert.equal(r.model, 0);
    assert.equal(r.world, 0);
    assert.equal(r.disabled, 0);
  }
  await writeFile(
    '../output/electron/verification/selection-owner-pixels.json',
    JSON.stringify(results, null, 2) + '\n',
  );
});

test('actual workspace anchor and child highlights occupy distinct pixels; model and world have no descendant highlight', async () => {
  await page.addInitScript(() =>
    localStorage.setItem(
      'mojive.electron.settings.v1',
      JSON.stringify({ language: 'en', appearance: 'dark' }),
    ),
  );
  await page.goto(base);
  await openEditableJointScene(page);
  const exact = (name) => page.locator(`[role=treeitem][data-tooltip="${name}"]`);
  const anchor = exact('01_revolute_anchor');
  await anchor.waitFor();
  await anchor.dblclick();
  await page.waitForTimeout(450);
  await page
    .locator('.left-panel .panel-header')
    .getByRole('button', { name: 'Search', exact: true })
    .click();
  const search = page.getByRole('textbox', { name: 'Search Scene', exact: true });
  const select = async (name) => {
    await search.fill(name);
    await exact(name).click();
    await page.waitForTimeout(80);
  };
  const capture = async (name) => {
    await page.evaluate((name) => {
      const canvas = document.querySelector('[data-testid="viewport-canvas"]'),
        gl = canvas.getContext('webgl2'),
        pixels = new Uint8Array(canvas.width * canvas.height * 4);
      gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      (window.ownerPixels ??= {})[name] = pixels;
      window.ownerWidth = canvas.width;
      window.ownerHeight = canvas.height;
    }, name);
    await page
      .getByTestId('viewport-canvas')
      .screenshot({ path: `../output/electron/verification/selection-scope-${name}.png` });
  };
  await select('World');
  await capture('baseline');
  await select('01_revolute_anchor');
  await capture('anchor');
  await select('01_revolute');
  await capture('child');
  await select('joint_gizmo_acceptance');
  await capture('model');
  const result = await page.evaluate(() => {
    const { baseline, anchor, child, model } = window.ownerPixels;
    let a = 0,
      b = 0,
      overlap = 0,
      interiorOverlap = 0,
      m = 0;
    const intersection = new Uint8Array(baseline.length / 4);
    for (let i = 0; i < baseline.length; i += 4) {
      const changed = (p) =>
          p[i] !== baseline[i] || p[i + 1] !== baseline[i + 1] || p[i + 2] !== baseline[i + 2],
        ca = changed(anchor),
        cb = changed(child);
      a += ca;
      b += cb;
      overlap += ca && cb;
      intersection[i / 4] = ca && cb;
      m += changed(model);
    }
    // Native outline masks intentionally ignore scene depth. The child's thin
    // silhouette can cross the anchor; its surface fill must not select it.
    // Remove the source kernel's four-pixel footprint before comparing interiors.
    for (let y = 4; y < window.ownerHeight - 4; y++)
      for (let x = 4; x < window.ownerWidth - 4; x++) {
        if (!intersection[y * window.ownerWidth + x]) continue;
        let interior = true;
        for (let dy = -4; dy <= 4 && interior; dy++)
          for (let dx = -4; dx <= 4; dx++)
            if (!intersection[(y + dy) * window.ownerWidth + x + dx]) {
              interior = false;
              break;
            }
        interiorOverlap += interior;
      }
    return { anchor: a, child: b, overlap, interiorOverlap, model: m };
  });
  assert.ok(result.anchor > 10000, JSON.stringify(result));
  assert.ok(result.child > 10000, JSON.stringify(result));
  assert.equal(result.interiorOverlap, 0, JSON.stringify(result));
  assert.equal(result.model, 0);
  await writeFile(
    '../output/electron/verification/selection-scope-app.json',
    JSON.stringify(result, null, 2) + '\n',
  );
  assert.deepEqual(errors, []);
});

test('body, joint, world and helper scopes match independently exported native render identities', async () => {
  const reference = JSON.parse(
      await readFile(
        new URL('./fixtures/selection-ownership-native.json', import.meta.url),
        'utf8',
      ),
    ),
    source = reference.scenes.find((scene) => scene.id === 'joint_gizmo'),
    comparable = source.selections.filter((node) => node.kind !== 'geom'),
    actual = await page.evaluate(async (selections) => {
      const { createNativeJointScene } = await import('/src/SceneCatalog.ts'),
        { selectionGeometry } = await import('/src/scene-selection.ts'),
        doc = createNativeJointScene(await (await fetch('/demo/joint_gizmo.xml')).text()),
        geometries = doc.nodes.filter((node) => node.kind === 'geom'),
        index = new Map(geometries.map((node, index) => [node.id, index]));
      return selections.map((selected) => {
        const node = doc.nodes.find(
          (node) => node.name === (selected.kind === 'world' ? 'World' : selected.name),
        );
        if (!node) throw Error('Missing production node ' + selected.name);
        return [
          ...new Set(
            [...selectionGeometry(doc, node.id)].map((mesh) => index.get(mesh.userData.nodeId)),
          ),
        ].sort((a, b) => a - b);
      });
    }, comparable);
  comparable.forEach((selected, index) => {
    const expected = [
      ...new Set(selected.matchingInstances.map((index) => source.instances[index].geomIndex)),
    ].sort((a, b) => a - b);
    assert.deepEqual(actual[index], expected, selected.name);
  });
  // Keep the adapter distinction explicit: these MODEL IDs own no geometry,
  // whereas the separate WorldInstances fixture intentionally owns render IDs.
  const attached = reference.scenes
    .flatMap((scene) => scene.selections)
    .find((node) => node.kind === 'model' && node.name === 'payload');
  assert.deepEqual(attached.matchingInstances, []);
  assert.deepEqual(
    await page.evaluate(async () => {
      const { createNativeJointScene } = await import('/src/SceneCatalog.ts'),
        { selectionGeometry } = await import('/src/scene-selection.ts');
      const doc = createNativeJointScene(await (await fetch('/demo/joint_gizmo.xml')).text());
      return [...selectionGeometry(doc, 'robot')];
    }),
    [],
  );
});
