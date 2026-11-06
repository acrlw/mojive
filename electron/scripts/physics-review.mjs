import { chooseSampleScene } from '../tests/helpers/editable-scene.mjs';
import { _electron as electron } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();
const packaged = process.argv.includes('--packaged');
const out = path.resolve(root, '../output/electron/physics', ...(packaged ? ['packaged'] : []));
await mkdir(out, { recursive: true });
await writeFile(path.join(out, 'environment.xml'), `<mujoco model="Environment review">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody><light pos="0 -2 4"/>
    <geom name="floor" type="plane" size="4 4 0.1" rgba="0.3 0.35 0.4 1"/>
    <body name="ball" pos="0 0 1"><freejoint name="ball_free"/><geom type="sphere" size="0.15" rgba="0.5 0.7 0.4 1"/></body>
    <body name="pendulum" pos="1 0 1"><joint name="hinge" axis="0 1 0" range="-90 90"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0.2 0 -0.5" rgba="0.9 0.5 0.2 1"/></body>
  </worldbody><actuator><motor name="hinge_motor" joint="hinge" ctrlrange="-2 2"/></actuator>
</mujoco>`);
const app = await electron.launch({
  ...(packaged ? { executablePath: path.resolve(root, '../output/electron/release/Mojive.app/Contents/MacOS/Mojive') } : {}),
  args: [...(packaged ? [] : [root]), '--hidden', '--user-data-dir=' + await mkdtemp(path.join(out, 'profile-'))],
  cwd: root, timeout: 45000,
});
const errors = [];
let page;
const action = (id) => app.evaluate(({ Menu, BrowserWindow }, id) => {
  const menu = Menu.getApplicationMenu()?.getMenuItemById(id);
  if (process.platform === 'darwin') { if (!menu?.enabled) throw new Error('Unavailable menu: ' + id); menu.click(); }
  else BrowserWindow.getAllWindows()[0].webContents.send('mojive:menu-action', id);
}, id);
const numeric = async (label, value) => {
  const input = page.getByLabel(label, { exact: true });
  await input.dblclick(); await input.fill(String(value)); await input.press('Enter');
};
const waitValue = async (label, value) => {
  await page.waitForFunction(({ label, value }) => Number(document.querySelector(`input[aria-label="${label}"]`)?.value) === value, { label, value });
};
const reviewTransport = async (theme) => {
  const original = await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].getBounds());
  for (const width of [1120, 640]) {
    const currentWidth = await page.locator('.viewport-panel').evaluate((element) => element.getBoundingClientRect().width);
    await app.evaluate(({ BrowserWindow }, delta) => {
      const window = BrowserWindow.getAllWindows()[0];
      const bounds = window.getBounds();
      window.setBounds({ ...bounds, width: Math.round(bounds.width + delta) });
    }, width - currentWidth);
    await page.waitForFunction((width) => Math.abs(document.querySelector('.viewport-panel').getBoundingClientRect().width - width) < 2, width);
    const layout = await page.locator('.viewport-topbar').evaluate((bar) => {
      const box = (element) => { const r = element.getBoundingClientRect(); return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, center: r.x + r.width / 2 }; };
      return { bar: box(bar), play: box(bar.querySelector('.simulation-transport .button:nth-child(2)')),
        groups: ['.viewport-navigation', '.simulation-actions', '.simulation-transport', '.simulation-speed', '.geometry-selector'].map((selector) => box(bar.querySelector(selector))),
        contentTop: document.querySelector('.viewport-content').getBoundingClientRect().top };
    });
    assert.ok(Math.abs(layout.bar.center - layout.play.center) < 1, 'Play stays centered in the viewport.');
    assert.ok(Math.abs(layout.contentTop - layout.bar.bottom) < 1, 'Transport occupies no extra viewport row.');
    layout.groups.forEach((group, index) => {
      assert.ok(group.top >= layout.bar.top && group.bottom <= layout.bar.bottom, 'Controls fit the shared row.');
      if (index) assert.ok(group.left >= layout.groups[index - 1].right, 'Independent groups do not overlap.');
    });
    await page.screenshot({ path: path.join(out, `transport-${theme}-${width}.png`) });
  }
  await app.evaluate(({ BrowserWindow }, bounds) => BrowserWindow.getAllWindows()[0].setBounds(bounds), original);
};
const reviewEnvironment = async (theme) => {
  await page.screenshot({ path: path.join(out, `environment-${theme}.png`) });
  const algorithms = page.locator('.environment-properties').getByRole('button', { name: '算法', exact: true });
  await algorithms.click();
  const geometry = await page.locator('.environment-property').evaluateAll((rows) => rows.map((row) => {
    const control = row.lastElementChild.getBoundingClientRect();
    return { left: control.left, right: control.right, overflow: row.scrollWidth - row.clientWidth };
  }));
  assert.ok(geometry.length >= 5, 'Basic and advanced Environment fields are available.');
  assert.ok(geometry.every((row) => row.overflow <= 1 &&
    Math.abs(row.left - geometry[0].left) < 1 && Math.abs(row.right - geometry[0].right) < 1),
    'Environment controls share Inspector columns: ' + JSON.stringify(geometry));
  await page.screenshot({ path: path.join(out, `environment-advanced-${theme}.png`) });
  await algorithms.click();
};
try {
  page = await app.firstWindow();
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  await page.waitForSelector('[role=treeitem]');
  assert.equal(await app.evaluate(({ app }) => app.isPackaged), packaged);
  // Startup and the asset example must load the same live MJCF as native make joint-gizmo.
  await page.getByRole('toolbar', { name: '仿真', exact: true }).waitFor();
  assert.match(await page.locator('.document-title').innerText(), /关节操作示例/);
  await page.locator('.activity-rail').getByRole('button', { name: '关节', exact: true }).click();
  await page.locator('.joint-entry').first().waitFor();
  assert.equal(await page.locator('.joint-entry').count(), 9);
  await page.getByRole('button', { name: '单步仿真', exact: true }).click();
  await page.waitForFunction(() => parseFloat(document.querySelector('.status-time')?.textContent ?? '0') > 0);
  await page.getByRole('button', { name: '上一帧仿真', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.status-time')?.textContent?.trim() === '0.000 s');
  await page.screenshot({ path: path.join(out, 'default-joint-mujoco.png') });
  // A history step back restores the clean state; make an explicit edit before testing discard.
  await page.getByRole('button', { name: '环境', exact: true }).click();
  await numeric('时间步长', 0.005);
  await waitValue('时间步长', 0.005);
  await chooseSampleScene(page, 'Joint gizmos', true);
  await page.getByRole('button', { name: '不保存', exact: true }).click();
  await page.getByRole('alertdialog').waitFor({ state: 'detached' });
  await page.getByRole('toolbar', { name: '仿真', exact: true }).waitFor();
  await page.waitForFunction(() => !document.querySelector('.busy-overlay'));
  await action('file.openSimulation');
  await page.getByRole('textbox', { name: '位置', exact: true }).fill(out);
  await page.getByRole('textbox', { name: '位置', exact: true }).press('Enter');
  await page.getByRole('option', { name: 'environment.xml', exact: true }).dblclick();
  await page.getByRole('toolbar', { name: '仿真', exact: true }).waitFor({ timeout: 45000 });
  assert.equal(await page.getByRole('button', { name: '上一帧仿真', exact: true }).isDisabled(), true);
  await page.getByRole('button', { name: '环境', exact: true }).click();
  await waitValue('时间步长', 0.002);
  await numeric('时间步长', 0.005);
  await page.getByRole('button', { name: '单步仿真', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.status-time')?.textContent?.trim() === '0.005 s');
  await page.getByRole('button', { name: '上一帧仿真', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.status-time')?.textContent?.trim() === '0.000 s');
  await page.getByRole('button', { name: '单步仿真', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.status-time')?.textContent?.trim() === '0.005 s');
  await page.getByRole('button', { name: '重置仿真', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.status-time')?.textContent?.trim() === '0.000 s');
  await waitValue('时间步长', 0.005);
  await page.getByRole('button', { name: '恢复模型参数', exact: true }).click();
  await waitValue('时间步长', 0.002);
  await action('edit.undo');
  await waitValue('时间步长', 0.005);
  await reviewEnvironment('dark');
  await reviewTransport('dark');
  await page.getByRole('button', { name: '运行仿真', exact: true }).click();
  await page.waitForFunction(() => parseFloat(document.querySelector('.status-time')?.textContent ?? '0') > 0.15);
  await page.getByRole('button', { name: '暂停仿真', exact: true }).click();
  const pausedTime = parseFloat(await page.locator('.status-time').innerText());
  await page.getByRole('button', { name: '上一帧仿真', exact: true }).click();
  await page.waitForFunction((before) => parseFloat(document.querySelector('.status-time')?.textContent ?? '0') < before, pausedTime);
  await page.getByRole('button', { name: '重置仿真', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.status-time')?.textContent?.trim() === '0.000 s');

  // Preview owns a separate document and Worker, including for physical models.
  await page.locator('.activity-rail').getByRole('button', { name: '资源', exact: true }).click();
  await page.locator('.asset-row-preview').filter({ hasText: '驱动摆' }).click();
  await page.locator('.asset-preview-stage canvas').first().waitFor();
  await page.screenshot({ path: path.join(out, 'assets-preview-dark.png') });
  await page.getByRole('button', { name: '关闭预览', exact: true }).click();
  assert.equal(parseFloat(await page.locator('.status-time').innerText()), 0);
  assert.match(await page.locator('.document-title').innerText(), /environment/);
  await page.getByRole('button', { name: '切换主题', exact: true }).click();
  await page.getByRole('button', { name: '环境', exact: true }).click();
  await reviewEnvironment('light');
  await reviewTransport('light');
  await page.locator('.activity-rail').getByRole('button', { name: '驱动器', exact: true }).click();
  await page.screenshot({ path: path.join(out, 'actuators-light.png') });

  // Capture past the former ten-second boundary and persist the settled terminal state.
  await page.locator('.bottom-tabs').getByRole('button', { name: /^时间轴/ }).click();
  await page.getByRole('button', { name: '录制场景帧', exact: true }).click();
  await page.waitForFunction(() => parseFloat(document.querySelector('.status-time')?.textContent ?? '0') > 10.1,
    undefined, { timeout: 45000 });
  await page.getByRole('button', { name: '停止录制', exact: true }).click();
  await page.getByRole('button', { name: '停止录制', exact: true }).waitFor({ state: 'detached' });
  await page.screenshot({ path: path.join(out, 'recording-light.png') });
  await action('file.save');
  await page.getByRole('textbox', { name: '位置', exact: true }).fill(out);
  await page.getByRole('textbox', { name: '位置', exact: true }).press('Enter');
  await page.locator('.file-dialog input').last().fill('physics-roundtrip.mojive.json');
  await page.getByRole('button', { name: '保存', exact: true }).click();
  const replace = page.getByRole('button', { name: '替换', exact: true });
  await page.waitForFunction(() => !document.querySelector('.file-dialog') || document.querySelector('[role=alertdialog]'));
  if (await replace.count()) await replace.click();
  await page.locator('.file-dialog').waitFor({ state: 'detached' });
  const saved = JSON.parse(await readFile(path.join(out, 'physics-roundtrip.mojive.json'), 'utf8'));
  await action('file.openScene');
  await page.getByRole('textbox', { name: '位置', exact: true }).fill(out);
  await page.getByRole('textbox', { name: '位置', exact: true }).press('Enter');
  await page.getByRole('option', { name: 'physics-roundtrip.mojive.json', exact: true }).dblclick();
  await page.locator('.file-dialog').waitFor({ state: 'detached' });
  await page.waitForFunction(() => parseFloat(document.querySelector('.status-time')?.textContent ?? '0') > 10);
  assert.equal(await page.locator('[data-lane]').count(), 3);
  assert.ok(await page.locator('.take-frame-tick').count() > 1);
  await writeFile(path.join(out, 'review.json'), JSON.stringify({ packaged, checks: ['live default joint MJCF and example reload', 'centered single-row transport at 640/1120px in both themes', 'previous frame and resumed single step', 'Environment edits/reset/undo',
    'isolated MJCF preview', 'recording beyond ten seconds', 'physical workspace roundtrip'],
    workspaceBytes: JSON.stringify(saved).length, errors }, null, 2));
  assert.deepEqual(errors, []);
  console.log('Production WASM, Environment, asset preview, long recording and workspace roundtrip passed.');
} catch (error) {
  console.error('Renderer errors:', errors);
  if (page) { await page.screenshot({ path: path.join(out, 'failure.png') }); console.error(await page.locator('body').innerText()); }
  throw error;
} finally { await app.evaluate(({ app }) => app.exit()).catch(() => {}); await app.close(); }
