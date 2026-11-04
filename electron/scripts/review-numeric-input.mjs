import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { _electron } from 'playwright';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const out = path.join(root, '../output/electron/verification');
await mkdir(out, { recursive: true });
const themes = JSON.parse(await readFile(path.join(root, 'src/locales/themes.json'), 'utf8'));
const server = await createServer({
  root,
  cacheDir: 'node_modules/.vite-numeric-production',
  server: { host: '127.0.0.1', port: 0, strictPort: false, watch: null },
  logLevel: 'error',
});
const evidence = [],
  errors = [];
let phase = 'start';
const rgb = (hex) =>
  `rgb(${[1, 3, 5].map((offset) => parseInt(hex.slice(offset, offset + 2), 16)).join(', ')})`;
try {
  await server.listen();
  const url = `http://127.0.0.1:${server.httpServer.address().port}`;
  for (const appearance of ['dark', 'light'])
    for (const uiScale of [1, 1.25]) {
      const profile = await mkdtemp(path.join(out, 'numeric-input-profile-'));
      let app, page;
      const entry = { appearance, uiScale, states: [], checks: {} };
      evidence.push(entry);
      try {
        phase = `${appearance}/${uiScale}: launch`;
        app = await _electron.launch({
          args: [path.join(root, 'main/main.cjs'), '--hidden', '--user-data-dir=' + profile],
          cwd: root,
          env: { ...process.env, MOJIVE_DEV_URL: url },
          timeout: 45000,
        });
        page = await app.firstWindow();
        page.setDefaultTimeout(10000);
        page.on('pageerror', (error) => errors.push({ phase, error: error.message }));
        await page.locator('.tree-row').first().waitFor();
        await page.evaluate(
          (settings) => {
            localStorage.setItem('mojive.electron.settings.v1', JSON.stringify(settings));
            localStorage.removeItem('mojive.layout');
          },
          { appearance, uiScale, language: 'en' },
        );
        await page.reload();
        await page.locator('.tree-row.type-camera').first().click();
        await page.evaluate(() => document.fonts.ready);
        const input = page
          .locator('.inspector-content > .section')
          .first()
          .getByRole('textbox', { name: 'Position X', exact: true });
        const field = input.locator('..');
        await input.scrollIntoViewIfNeeded();
        const initial = await input.inputValue();
        const point = async () => {
          const r = await input.boundingBox();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
        };
        const settle = () =>
          page.evaluate(
            () =>
              new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
          );
        const undoEnabled = () =>
          app.evaluate(
            ({ Menu }) => Menu.getApplicationMenu().getMenuItemById('edit.undo').enabled,
          );
        const expectUndo = async (expected) => {
          await settle();
          for (let i = 0; i < 50; i++) {
            if ((await undoEnabled()) === expected) return;
            await page.waitForTimeout(20);
          }
          assert.equal(await undoEnabled(), expected, `${phase}: native Undo availability`);
        };
        const undo = async () => {
          await expectUndo(true);
          await app.evaluate(({ Menu, BrowserWindow }) => {
            const action = Menu.getApplicationMenu().getMenuItemById('edit.undo');
            action.click(action, BrowserWindow.getAllWindows()[0], {});
          });
          await page.waitForFunction(
            ({ initial }) =>
              document.querySelector('.inspector-content > .section input[aria-label="Position X"]')
                .value === initial,
            { initial },
          );
          await expectUndo(false);
        };
        const idle = async () => {
          assert.equal(
            await field.evaluate(
              (node) =>
                node.classList.contains('editing') ||
                node.classList.contains('pressed') ||
                node.classList.contains('scrubbing'),
            ),
            false,
            `${phase}: field returns to display mode`,
          );
          assert.equal(
            await input.evaluate((node) => document.activeElement === node),
            false,
            `${phase}: single click must not enter text`,
          );
        };
        const paint = async (state) => {
          const data = await field.evaluate((node) => ({
            background: getComputedStyle(node).backgroundColor,
            classes: node.className,
            inputFocused: document.activeElement === node.querySelector('input'),
            value: node.querySelector('input').value,
            bounds: node.getBoundingClientRect().toJSON(),
          }));
          assert.equal(
            data.background,
            rgb(
              themes[appearance][
                { rest: 'field', hover: 'field_hovered', held: 'field_active' }[state]
              ],
            ),
          );
          entry.states.push({ state, ...data });
          await page.locator('.inspector').screenshot({
            path: path.join(out, `numeric-input-${appearance}-${uiScale}x-${state}.png`),
          });
        };
        phase = `${appearance}/${uiScale}: rest/hover/held and no-op click`;
        await expectUndo(false);
        await page.mouse.move(10, 10);
        await paint('rest');
        let p = await point();
        await page.mouse.move(p.x, p.y);
        await paint('hover');
        await page.mouse.down();
        assert.equal(await input.inputValue(), initial);
        assert.equal(await field.evaluate((node) => node.classList.contains('pressed')), true);
        await paint('held');
        await expectUndo(false);
        await page.mouse.up();
        await idle();
        await expectUndo(false);
        assert.equal(new Set(entry.states.map((item) => item.background)).size, 3);
        entry.checks.singleClickNoTextOrUndo = true;

        phase = `${appearance}/${uiScale}: drag and one native Undo`;
        p = await point();
        await page.mouse.move(p.x, p.y);
        await page.mouse.down();
        await page.mouse.move(p.x + 40, p.y, { steps: 8 });
        assert.equal(await field.evaluate((node) => node.classList.contains('scrubbing')), true);
        const dragged = Number(await input.inputValue());
        assert.ok(
          Math.abs(dragged - (Number(initial) + 0.4)) < 0.001,
          `actual drag value ${dragged}`,
        );
        await page
          .locator('.inspector')
          .screenshot({ path: path.join(out, `numeric-input-${appearance}-${uiScale}x-drag.png`) });
        await page.mouse.up();
        await idle();
        await undo();
        entry.checks.drag = { delta: dragged - Number(initial), oneUndo: true };

        phase = `${appearance}/${uiScale}: double click exact text and one native Undo`;
        const exact = '-12.3456789';
        await input.dblclick();
        assert.equal(await field.evaluate((node) => node.classList.contains('editing')), true);
        assert.equal(await input.evaluate((node) => document.activeElement === node), true);
        await input.fill(exact);
        await page
          .locator('.inspector')
          .screenshot({ path: path.join(out, `numeric-input-${appearance}-${uiScale}x-text.png`) });
        await input.press('Enter');
        assert.equal(await input.inputValue(), Number(exact).toFixed(3));
        await idle();
        await input.dblclick();
        assert.equal(await input.inputValue(), exact, 're-entering exposes authored precision');
        await input.press('Escape');
        await undo();
        entry.checks.doubleClickText = {
          authored: exact,
          display: Number(exact).toFixed(3),
          oneUndo: true,
        };

        phase = `${appearance}/${uiScale}: cancel drag and next gesture`;
        p = await point();
        await page.mouse.move(p.x, p.y);
        await page.mouse.down();
        await page.mouse.move(p.x - 24, p.y, { steps: 6 });
        assert.notEqual(await input.inputValue(), initial);
        await page.keyboard.press('Escape');
        assert.equal(await input.inputValue(), initial);
        await page.mouse.up();
        await idle();
        await expectUndo(false);
        await input.dblclick();
        await input.fill('25.123456');
        await input.press('Escape');
        assert.equal(await input.inputValue(), initial);
        await expectUndo(false);
        p = await point();
        await page.mouse.move(p.x, p.y);
        await page.mouse.down();
        await page.mouse.move(p.x + 20, p.y, { steps: 5 });
        await page.mouse.up();
        const next = Number(await input.inputValue());
        assert.ok(Math.abs(next - (Number(initial) + 0.2)) < 0.001);
        await idle();
        await undo();
        entry.checks.cancel = {
          dragRestored: true,
          textRestored: true,
          noUndo: true,
          nextGestureDelta: next - Number(initial),
          nextUndo: true,
        };
        assert.deepEqual(errors, []);
      } catch (error) {
        entry.failure = { phase, message: String(error), stack: error.stack };
        await page?.screenshot({
          path: path.join(out, `numeric-input-${appearance}-${uiScale}x-failure.png`),
        });
        throw error;
      } finally {
        if (app) {
          await app.evaluate(({ BrowserWindow }) =>
            BrowserWindow.getAllWindows().forEach((window) => window.destroy()),
          );
          await app.close();
        }
        await rm(profile, { recursive: true, force: true });
      }
    }
  console.log(JSON.stringify({ cases: evidence.length, errors, captures: out }, null, 2));
} finally {
  await writeFile(
    path.join(out, 'numeric-input-review.json'),
    JSON.stringify({ evidence, errors, phase }, null, 2) + '\n',
  );
  await server.close();
}
