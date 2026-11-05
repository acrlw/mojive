import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser, page, base;
const errors = [];
before(async () => {
  server = await createServer({
    configFile: false,
    cacheDir: 'node_modules/.vite-selection-test',
    server: { host: '127.0.0.1', port: 0, hmr: false },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 800, height: 700 } });
  page.on('pageerror', (error) => errors.push(error.message));
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
beforeEach(async () => {
  errors.length = 0;
  await page.goto(`${base}/tests/harness/selection-review.html`);
  await page.waitForFunction(() => window.selectionReview?.api.current);
});

test('ground defaults derive from authored type; selection inheritance survives project files and undo snapshots', async () => {
  const result = await page.evaluate(async () => {
    const f = window.selectionReview,
      { createPrimitive } = await import('/src/scene.ts'),
      { createNativeJointScene } = await import('/src/SceneCatalog.ts'),
      { isViewportPickable, setViewportPickable } = await import('/src/scene-selection.ts'),
      { encodeDocument, decodeDocument, saveWorkspace, loadWorkspace } = await import(
        '/src/editor.ts'
      );
    const disguisedBox = createPrimitive('box', 'Floor'),
      renamedGround = createPrimitive('ground', 'Table'),
      native = createNativeJointScene(
        '<mujoco><worldbody><geom name="Renamed surface" type="plane" size="0 0 .04"/><body name="Part"><geom name="floor" type="box" size=".2 .2 .2"/></body></worldbody></mujoco>',
      );
    const before = encodeDocument(f.doc);
    setViewportPickable(f.parent, false);
    setViewportPickable(f.front.object, true);
    setViewportPickable(f.ground.object, true);
    const after = encodeDocument(f.doc),
      restored = decodeDocument(before),
      redone = decodeDocument(after),
      loaded = loadWorkspace(saveWorkspace(f.doc, [], 10)).doc;
    const state = (doc) => ({
      ground: isViewportPickable(doc.nodes.find((n) => n.id === f.ground.id).object),
      front: isViewportPickable(doc.nodes.find((n) => n.id === f.front.id).object),
      parent: doc.nodes.find((n) => n.id === 'group').object.userData.viewportPickable,
    });
    return {
      initial: state(restored),
      redone: state(redone),
      loaded: state(loaded),
      floorByType: !isViewportPickable(renamedGround.object),
      boxByType: isViewportPickable(disguisedBox.object),
      nativePlane: !isViewportPickable(
        native.nodes.find((n) => n.name === 'Renamed surface').object,
      ),
      nativeBox: isViewportPickable(native.nodes.find((n) => n.name === 'floor').object),
      localChild: f.front.object.userData.viewportPickable,
    };
  });
  assert.equal(result.floorByType, true);
  assert.equal(result.boxByType, true);
  assert.equal(result.nativePlane, true);
  assert.equal(result.nativeBox, true);
  assert.deepEqual(result.initial, { ground: false, front: true, parent: undefined });
  assert.deepEqual(result.redone, { ground: true, front: false, parent: false });
  assert.deepEqual(result.loaded, result.redone);
  assert.equal(result.localChild, true);
});

async function selected() {
  return page.evaluate(() => window.selectionReview.selected);
}
async function change(fn) {
  await page.evaluate(fn);
  await page.waitForTimeout(50);
}
test('real viewport ray skips disabled ancestor subtrees, reaches objects behind them and clears on a disabled floor', async () => {
  await page.evaluate(() => {
    const f = window.selectionReview;
    f.api.current.view('top');
    f.api.current.setProjection('orthographic');
  });
  await page.waitForTimeout(450);
  await page.mouse.click(400, 350);
  const ids = await page.evaluate(() => ({
    front: window.selectionReview.front.id,
    back: window.selectionReview.back.id,
    ground: window.selectionReview.ground.id,
  }));
  assert.equal(await selected(), ids.front);
  await change(() => {
    const f = window.selectionReview;
    f.setViewportPickable(f.parent, false);
    f.setViewportPickable(f.front.object, true);
    f.refresh();
  });
  await page.mouse.click(400, 350);
  assert.equal(await selected(), ids.back);
  await change(() => {
    const f = window.selectionReview;
    f.setViewportPickable(f.back.object, false);
    f.refresh();
  });
  await page.mouse.click(400, 350);
  assert.equal(await selected(), null);
  await change(() => {
    const f = window.selectionReview;
    f.setViewportPickable(f.ground.object, true);
    f.refresh();
  });
  await page.mouse.click(100, 550);
  assert.equal(await selected(), ids.ground);
  await change(() => {
    const f = window.selectionReview;
    f.setViewportPickable(f.ground.object, false);
    f.refresh();
  });
  // Hierarchy selection remains legal, without introducing a surface highlight.
  await page.mouse.click(100, 550);
  assert.equal(await selected(), null);
  const unselected = await page.evaluate(() => window.selectionReview.api.current.capture());
  await change(() => {
    const f = window.selectionReview;
    f.select(f.ground.id);
  });
  assert.equal(await selected(), ids.ground);
  assert.equal(await page.evaluate(() => window.selectionReview.api.current.capture()), unselected);
  await page
    .getByTestId('viewport-canvas')
    .screenshot({ path: '../output/electron/verification/nonselectable-floor-tree-selection.png' });
  assert.deepEqual(errors, []);
});

test('selection masks and fill omit disabled descendants in every geometry mode without changing scene pixels', async () => {
  const results = await page.evaluate(async () => {
    const { THREE } = window.selectionReview,
      { ViewportPipeline } = await import('/src/viewport/pipeline.ts'),
      { setViewportPickable } = await import('/src/scene-selection.ts');
    const renderer = new THREE.WebGLRenderer({ preserveDrawingBuffer: true, antialias: true });
    renderer.setSize(240, 160);
    renderer.setClearColor('#22272d');
    const pipeline = new ViewportPipeline(renderer);
    pipeline.resize(240, 160);
    const scene = new THREE.Scene(),
      root = new THREE.Group(),
      parent = new THREE.Group();
    scene.add(root);
    root.add(parent);
    const left = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshStandardMaterial({ color: '#7e9db9' }),
      ),
      right = left.clone();
    right.material = left.material.clone();
    left.position.x = -0.8;
    right.position.x = 0.8;
    left.userData.role = 'visual';
    right.userData.role = 'visual';
    parent.add(left, right);
    scene.add(new THREE.AmbientLight('white', 2));
    const camera = new THREE.OrthographicCamera(-2, 2, 1.33, -1.33, 0.01, 100);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const doc = {
      root,
      nodes: [{ id: 'group', object: parent }],
      joints: [],
      files: [],
      warnings: [],
    };
    setViewportPickable(left, false);
    const gl = renderer.getContext(),
      pixels = () => {
        const p = new Uint8Array(240 * 160 * 4);
        gl.readPixels(0, 0, 240, 160, gl.RGBA, gl.UNSIGNED_BYTE, p);
        return p;
      };
    const difference = (a, b, lo, hi) => {
      let changed = 0;
      for (let y = 0; y < 160; y++)
        for (let x = lo; x < hi; x++) {
          const i = (y * 240 + x) * 4;
          if (a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2]) changed++;
        }
      return changed;
    };
    const results = [];
    for (const mode of ['default', 'visual', 'collision', 'both']) {
      left.userData.role = right.userData.role = mode === 'collision' ? 'collision' : 'visual';
      setViewportPickable(parent, true);
      pipeline.render(scene, camera, doc, null, mode);
      const baseline = pixels();
      pipeline.render(scene, camera, doc, 'group', mode);
      const selected = pixels();
      setViewportPickable(parent, false);
      pipeline.render(scene, camera, doc, 'group', mode);
      const allDisabled = pixels();
      results.push({
        mode,
        left: difference(baseline, selected, 0, 120),
        right: difference(baseline, selected, 120, 240),
        allDisabled: difference(baseline, allDisabled, 0, 240),
      });
    }
    pipeline.dispose();
    renderer.dispose();
    return results;
  });
  for (const result of results) {
    assert.equal(result.left, 0, JSON.stringify(result));
    assert.ok(result.right > 500, JSON.stringify(result));
    assert.equal(result.allDisabled, 0, JSON.stringify(result));
  }
});

test('camera and light markers keep visible artwork but ignore disabled targets and selection color', async () => {
  const result = await page.evaluate(async () => {
    const { THREE } = window.selectionReview,
      { SceneEntityMarkers } = await import('/src/viewport/entities.ts'),
      { createPrimitive } = await import('/src/scene.ts'),
      { setViewportPickable } = await import('/src/scene-selection.ts');
    const renderer = new THREE.WebGLRenderer({ alpha: true, preserveDrawingBuffer: true });
    renderer.setSize(100, 100);
    renderer.setClearColor(0, 0);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const root = new THREE.Group(),
      parent = new THREE.Group(),
      entity = createPrimitive('camera'),
      behind = createPrimitive('light');
    root.add(parent, behind.object);
    parent.add(entity.object);
    entity.object.position.set(0, 0, 0);
    behind.object.position.set(0, 0, -1);
    root.updateMatrixWorld(true);
    const doc = { root, nodes: [entity, behind] },
      markers = new SceneEntityMarkers();
    markers.render(renderer, camera, doc, null, 100, 100);
    const enabled = markers.pick([50, 50]);
    setViewportPickable(parent, false);
    const next = markers.pick([50, 50]);
    setViewportPickable(behind.object, false);
    const none = markers.pick([50, 50]);
    renderer.clear();
    markers.render(renderer, camera, doc, null, 100, 100);
    const idle = renderer.domElement.toDataURL();
    renderer.clear();
    markers.render(renderer, camera, doc, entity.id, 100, 100);
    const selected = renderer.domElement.toDataURL();
    markers.dispose();
    renderer.dispose();
    return {
      enabled: enabled === entity.id,
      next: next === behind.id,
      none,
      idleEqualsSelected: idle === selected,
    };
  });
  assert.deepEqual(result, { enabled: true, next: true, none: null, idleEqualsSelected: true });
});

test('encoded and restored history snapshots do not alias mutable object metadata or node properties', async () => {
  const result = await page.evaluate(async () => {
    const f = window.selectionReview,
      { encodeDocument, decodeDocument } = await import('/src/editor.ts');
    f.front.object.userData.editor = { nested: { enabled: true } };
    f.front.properties = { Size: 1 };
    const encoded = encodeDocument(f.doc),
      original = JSON.stringify(encoded);
    f.front.object.userData.editor.nested.enabled = false;
    f.front.properties.Size = 2;
    const isolatedFromLive = JSON.stringify(encoded) === original,
      restored = decodeDocument(encoded);
    const node = restored.nodes.find((n) => n.id === f.front.id);
    node.object.userData.editor.nested.enabled = false;
    node.properties.Size = 3;
    return {
      isolatedFromLive,
      isolatedFromRestore: JSON.stringify(encoded) === original,
      restoredAgain: decodeDocument(encoded).nodes.find((n) => n.id === f.front.id).object.userData
        .editor.nested.enabled,
    };
  });
  assert.deepEqual(result, {
    isolatedFromLive: true,
    isolatedFromRestore: true,
    restoredAgain: true,
  });
});

test('actual Inspector toggle records undo/redo while disabled floor remains tree-editable', async () => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.addInitScript(() =>
    localStorage.setItem(
      'mojive.electron.settings.v1',
      JSON.stringify({ language: 'en', appearance: 'dark' }),
    ),
  );
  await page.goto(base);
  await page.getByRole('treeitem').first().waitFor();
  const floor = page.locator('.tree-row[data-tooltip="floor"]');
  await floor.click();
  const toggle = page.getByRole('switch', { name: 'Select in viewport', exact: true });
  await toggle.waitFor();
  assert.equal(await toggle.getAttribute('aria-checked'), 'false');
  await toggle.click();
  assert.equal(await toggle.getAttribute('aria-checked'), 'true');
  await page.keyboard.press('ControlOrMeta+z');
  await page.waitForTimeout(100);
  assert.equal(await toggle.getAttribute('aria-checked'), 'false');
  await page.keyboard.press('ControlOrMeta+Shift+z');
  await page.waitForTimeout(100);
  assert.equal(await toggle.getAttribute('aria-checked'), 'true');
  await toggle.click();
  assert.equal(await toggle.getAttribute('aria-checked'), 'false');
  assert.equal(await floor.getAttribute('aria-selected'), 'true');
  const position = page
    .locator('.inspector')
    .getByRole('textbox', { name: 'Position Z', exact: true });
  await position.fill('0.1');
  await position.press('Enter');
  assert.equal(Number(await position.inputValue()), 0.1);
  await page.getByRole('button', { name: 'Position (G)', exact: true }).click();
  await page.waitForTimeout(60);
  assert.ok(
    await page
      .locator('[data-testid="viewport-canvas"] ~ canvas:not([data-testid])')
      .evaluate((canvas) => {
        const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        return pixels.some((value, index) => index % 4 === 3 && value > 0);
      }),
    'Hierarchy-selected nonselectable floor retains its transform gizmo',
  );
  await page
    .locator('.inspector')
    .screenshot({ path: '../output/electron/verification/nonselectable-floor-inspector.png' });
  await page
    .locator('.viewport-content')
    .screenshot({ path: '../output/electron/verification/nonselectable-floor-gizmo.png' });
  await page.locator('.tree-row.type-world').first().click();
  await toggle.click();
  await floor.click();
  assert.equal(await toggle.isDisabled(), true);
  assert.equal(await toggle.getAttribute('aria-checked'), 'false');
  assert.deepEqual(errors, []);
});
