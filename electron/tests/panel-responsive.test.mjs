import assert from 'node:assert/strict';
import { test, before, after } from 'node:test';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
let server, browser;
const errors = [];
before(async () => {
  server = await createServer({
    server: { host: '127.0.0.1', port: 0, strictPort: false },
    logLevel: 'error',
  });
  await server.listen();
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
async function resize(page, width, scale) {
  const splitter = page.getByRole('separator', { name: 'Resize left panel', exact: true }),
    box = await splitter.boundingBox();
  const current = Number(await splitter.getAttribute('aria-valuenow'));
  await page.mouse.move(box.x + box.width / 2, box.y + Math.min(200, box.height / 2));
  await page.mouse.down();
  await page.mouse.move(
    box.x + box.width / 2 + (width - current) * scale,
    box.y + Math.min(200, box.height / 2),
    { steps: 12 },
  );
  await page.mouse.up();
  await page.waitForTimeout(70);
}
async function metrics(page, panelClass) {
  return page.locator(panelClass).evaluate((panel) => {
    const host = panel.parentElement,
      panelRect = panel.getBoundingClientRect(),
      hostRect = host.getBoundingClientRect();
    return {
      panel: panelRect.width,
      host: hostRect.width,
      right: hostRect.right - panelRect.right,
      overflow: panel.scrollWidth - panel.clientWidth,
    };
  });
}
for (const scale of [1, 1.25])
  test(`real workspace joints and cameras fill dragged panes at ${scale}x with complete responsive XYZ rows`, async () => {
    const page = await browser.newPage({ viewport: { width: 1680, height: 1100 } });
    page.setDefaultTimeout(8000);
    page.on('pageerror', (e) => errors.push(e.message));
    await page.addInitScript((scale) => {
      localStorage.setItem(
        'mojive.electron.settings.v1',
        JSON.stringify({ language: 'en', appearance: 'dark', uiScale: scale }),
      );
      localStorage.removeItem('mojive.layout');
    }, scale);
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/`);
    await page.waitForLoadState('networkidle');
    const shell = await page.locator('.app-shell').evaluate((element) => {
      const rect = element.getBoundingClientRect();
      return {
        width: rect.width,
        height: rect.height,
        windowWidth: innerWidth,
        windowHeight: innerHeight,
      };
    });
    assert.ok(Math.abs(shell.width - shell.windowWidth) < 1, JSON.stringify(shell));
    assert.ok(Math.abs(shell.height - shell.windowHeight) < 1, JSON.stringify(shell));
    await page.getByTitle('Joints', { exact: true }).click();
    const joint = page.locator('.joint-entry').filter({ hasText: '04_free_6dof' });
    await joint.waitFor();
    await joint.getByTitle('Expand joint pose', { exact: true }).click();
    for (const [size, label] of [
      [220, 'narrow'],
      [410, 'wide'],
    ]) {
      await resize(page, size, scale);
      await joint.scrollIntoViewIfNeeded();
      const m = await metrics(page, '.joint-panel');
      assert.ok(Math.abs(m.panel - m.host) < 1.5, JSON.stringify(m));
      assert.ok(Math.abs(m.right) < 1.5, JSON.stringify(m));
      assert.ok(m.overflow <= 1);
      const geometry = await joint.evaluate((card) => {
        const fields = [...card.querySelector('.vector-fields').children].map((el) => {
          const r = el.getBoundingClientRect();
          return { x: r.x, y: r.y, width: r.width };
        });
        const r = card.getBoundingClientRect(),
          name = card.querySelector('.entity-card-heading').getBoundingClientRect(),
          style = getComputedStyle(card),
          textWidths = [...card.querySelectorAll('.vector-fields input')].map((input) => {
            const style = getComputedStyle(input),
              canvas = document.createElement('canvas'),
              context = canvas.getContext('2d');
            context.font = `${style.fontSize} ${style.fontFamily}`;
            return {
              available:
                input.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight),
              required: context.measureText(input.value).width,
            };
          });
        return {
          fields,
          radius: style.borderRadius,
          inset: style.padding,
          left: name.left - r.left,
          right: r.right - card.querySelector('.entity-card-heading').getBoundingClientRect().right,
          inner: getComputedStyle(card.querySelector('.axis-field')).borderRadius,
          textWidths,
        };
      });
      assert.equal(geometry.radius, '12px');
      assert.equal(geometry.inset, '8px');
      assert.equal(geometry.inner, '4px');
      assert.ok(Math.abs(geometry.left - 8 * scale) < 1);
      assert.ok(Math.abs(geometry.right - 8 * scale) < 1);
      if (label === 'wide') {
        assert.ok(geometry.fields.every((field) => Math.abs(field.y - geometry.fields[0].y) < 1));
        assert.ok(geometry.fields[1].x > geometry.fields[0].x);
      } else assert.ok(geometry.fields[1].y > geometry.fields[0].y);
      for (const metric of geometry.textWidths)
        assert.ok(metric.available + 1 >= metric.required, JSON.stringify(metric));
      await page.screenshot({ path: `../output/electron/verification/joints-dragged-${label}-${scale}x.png` });
      const heading = joint.locator('.entity-card-heading');
      await heading.hover();
      const hover = await heading.evaluate((element) => {
        const style = getComputedStyle(element),
          bounds = element.getBoundingClientRect(),
          icon = element.querySelector('.icon').getBoundingClientRect();
        return {
          radius: style.borderRadius,
          padding: style.paddingLeft,
          iconInset: icon.left - bounds.left,
          background: style.backgroundColor,
        };
      });
      assert.equal(hover.radius, '4px');
      assert.equal(hover.padding, '4px');
      assert.ok(Math.abs(hover.iconInset - 4 * scale) < 1);
      assert.notEqual(hover.background, 'rgba(0, 0, 0, 0)');
      await joint.screenshot({ path: `../output/electron/verification/joint-hover-${label}-${scale}x.png` });
    }
    const ball = page
      .locator('.joint-entry')
      .filter({ has: page.locator('.entity-card-heading').filter({ hasText: '03_ball_xyz' }) });
    for (const angle of [20, 30]) {
      if (!(await ball.getByLabel('Rotation X', { exact: true }).count()))
        await ball.getByRole('button', { name: /Expand/ }).click();
      await ball.getByLabel('Rotation X', { exact: true }).fill(String(angle));
      await ball.getByLabel('Rotation X', { exact: true }).press('Enter');
      assert.ok(
        Math.abs(
          Number(await ball.getByLabel('Rotation X', { exact: true }).inputValue()) - angle,
        ) < 0.01,
      );
    }
    await page.getByTitle('Cameras', { exact: true }).click();
    await page.locator('.cameras-panel').waitFor();
    for (const [size, label] of [
      [220, 'narrow'],
      [410, 'wide'],
    ]) {
      await resize(page, size, scale);
      const m = await metrics(page, '.cameras-panel');
      assert.ok(Math.abs(m.panel - m.host) < 1.5, JSON.stringify(m));
      assert.ok(Math.abs(m.right) < 1.5, JSON.stringify(m));
      assert.ok(m.overflow <= 1);
      const card = page.locator('.camera-entry').first();
      if (await card.count()) {
        const v = await card.evaluate((el) => ({
          width: el.getBoundingClientRect().width,
          parent: el.parentElement.getBoundingClientRect().width,
          radius: getComputedStyle(el).borderRadius,
          padding: getComputedStyle(el).padding,
        }));
        assert.ok(Math.abs(v.width - v.parent) < 1.5);
        assert.equal(v.radius, '12px');
        assert.equal(v.padding, '8px');
      }
      await page.screenshot({ path: `../output/electron/verification/cameras-dragged-${label}-${scale}x.png` });
      await card.locator('.camera-entry-name').hover();
      const grouping = await card.evaluate((element) => {
        const outer = element.getBoundingClientRect(),
          header = element.querySelector('.camera-entry-name'),
          h = header.getBoundingClientRect(),
          icon = header.querySelector('.icon').getBoundingClientRect();
        const actions = [...element.querySelectorAll('.camera-entry-actions > button')].map(
          (button) => ({
            y: button.getBoundingClientRect().y,
            right: button.getBoundingClientRect().right,
            radius: getComputedStyle(button).borderRadius,
          }),
        );
        return {
          left: h.left - outer.left,
          moreRight:
            outer.right - element.querySelector('.camera-entry-more').getBoundingClientRect().right,
          height: outer.height,
          inset: icon.left - h.left,
          radius: getComputedStyle(header).borderRadius,
          background: getComputedStyle(header).backgroundColor,
          actions,
          cardRight: outer.right,
        };
      });
      assert.equal(grouping.radius, '4px');
      assert.notEqual(grouping.background, 'rgba(0, 0, 0, 0)');
      assert.ok(
        Math.abs(grouping.left - 8 * scale) < 1 && Math.abs(grouping.moreRight - 8 * scale) < 1,
      );
      assert.ok(Math.abs(grouping.inset - 8 * scale) < 1);
      assert.ok(grouping.height <= 83 * scale);
      assert.equal(grouping.actions.length, 2);
      assert.ok(
        grouping.actions.every(
          (action) => Math.abs(action.y - grouping.actions[0].y) < 1 && action.radius === '4px',
        ),
      );
      assert.ok(Math.abs(grouping.cardRight - grouping.actions.at(-1).right - 8 * scale) < 1);
      await card.screenshot({ path: `../output/electron/verification/camera-card-hover-${label}-${scale}x.png` });
      await card.getByRole('button', { name: 'More camera actions', exact: true }).click();
      const more = page.getByRole('menu', { name: /More camera actions/ });
      assert.equal(await more.getByRole('menuitem').count(), 4);
      for (const name of ['Focus camera', 'Align to view', 'Capture image', 'Delete camera'])
        assert.equal(await more.getByRole('menuitem', { name, exact: true }).isVisible(), true);
      await page.keyboard.press('Escape');
      await card.getByTitle('View Camera', { exact: true }).click();
      assert.equal(
        await card.locator('.camera-entry-actions button').nth(0).getAttribute('aria-pressed'),
        'true',
      );
      await card.getByTitle('Pilot camera', { exact: true }).click();
      assert.equal(
        await card.locator('.camera-entry-actions button').nth(1).getAttribute('aria-pressed'),
        'true',
      );
      await page
        .locator('.cameras-panel')
        .getByRole('button', { name: 'Editor Camera', exact: true })
        .click();
    }
    await page.close();
    assert.deepEqual(errors, []);
  });
