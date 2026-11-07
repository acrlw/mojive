import { installPaintStyle } from './helpers/surface-style.mjs';
import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser, page, base;
const fixture = `<!doctype html><html><body><div id="root"></div><script type="module">
import React,{useState} from 'react';import{createRoot}from'react-dom/client';
import '/src/design-tokens.css';import '/src/styles.css';import '/src/workspace.css';
import{JointPanel}from'/src/JointPanel.tsx';import{createCatalogScene}from'/src/SceneCatalog.ts';
import{applySettings,normalizeSettings}from'/src/settings.tsx';
const doc=await createCatalogScene('joint_gizmo');
window.review={doc,selections:[],focus:[],begins:0,commits:0,leakedKeys:0};
window.mojive={writeClipboardText:async text=>{window.review.clipboard=text}};
window.reviewTheme=theme=>applySettings(normalizeSettings({language:'en',appearance:theme}));window.reviewTheme('dark');
window.addEventListener('keydown',e=>{if(e.key===' '||e.key==='Enter')window.review.leakedKeys++;});
function Fixture(){const[selected,setSelected]=useState(null),[,refresh]=useState(0);window.review.select=setSelected;
return React.createElement('div',{id:'pane',style:{width:356,height:'100vh',background:'var(--panel-color)'}},React.createElement(JointPanel,{doc,selectedId:selected,
onSelect:id=>{window.review.selections.push(id);setSelected(id)},onFocus:id=>window.review.focus.push(id),
onBegin:()=>window.review.begins++,onCommit:()=>window.review.commits++,onChange:()=>refresh(x=>x+1)}));}
createRoot(document.getElementById('root')).render(React.createElement(Fixture));
</script></body></html>`;
before(async () => {
  server = await createServer({
    configFile: false,
    root: process.cwd(),
    cacheDir: 'node_modules/.vite-joint-heading-test',
    plugins: [
      {
        name: 'joint-heading-test',
        configureServer(server) {
          server.middlewares.use('/__joint-heading', async (_, res, next) => {
            try {
              res.setHeader('Content-Type', 'text/html');
              res.end(await server.transformIndexHtml('/__joint-heading', fixture));
            } catch (error) {
              next(error);
            }
          });
        },
      },
    ],
    server: { host: '127.0.0.1', port: 0, hmr: false },
    optimizeDeps: { include: ['react', 'react-dom/client', 'react/jsx-runtime', 'three'] },
    logLevel: 'error',
  });
  await server.listen();
  base = `http://127.0.0.1:${server.httpServer.address().port}`;
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  page = await browser.newPage({ viewport: { width: 800, height: 1000 } });
  await page.addInitScript(installPaintStyle);
  await page.evaluate(installPaintStyle);
  page.setDefaultTimeout(5000);
  await mkdir('../output/electron/verification', { recursive: true });
});
after(async () => {
  await browser?.close();
  await server?.close();
});
beforeEach(async () => {
  await page.goto(base + '/__joint-heading');
  await page.locator('.joint-entry').first().waitFor();
  await page.evaluate(() => document.fonts.ready);
});
const card = (name) =>
  page
    .locator('.joint-entry')
    .filter({ has: page.locator('.entity-card-title', { hasText: name }) });
const heading = (name) => card(name).locator('.entity-card-heading');
const ball = '03_ball_xyz',
  free = '04_free_6dof';
async function assertOpen(name, open) {
  assert.equal(await heading(name).getAttribute('aria-expanded'), String(open));
  assert.equal(
    await card(name).locator('.joint-pose-fields:not(.joint-pose-summary)').count(),
    open ? 1 : 0,
  );
}

test('the complete ball/free header toggles exactly once and selection cannot reopen an explicit collapse', async () => {
  for (const name of [ball, free]) {
    assert.equal(await heading(name).evaluate((el) => el.tagName), 'BUTTON');
    assert.equal(
      await heading(name).locator('button,[role=button]').count(),
      0,
      'Heading does not contain nested controls',
    );
    await card(name).locator('.entity-card-title').click();
    await assertOpen(name, true);
    await card(name).locator('.entity-card-context').click();
    await assertOpen(name, false);
    await card(name).locator('.entity-card-disclosure').click();
    await assertOpen(name, true);
    const h = await heading(name).boundingBox();
    await page.mouse.click(h.x + h.width - 2, h.y + h.height / 2);
    await assertOpen(name, false);
    await heading(name).click();
    await assertOpen(name, true);
  }
  await heading(ball).click();
  await assertOpen(ball, false);
  assert.match(await card(ball).getAttribute('class'), /selected/);
  await page.evaluate(() =>
    window.review.select(window.review.doc.joints.find((j) => j.name === '04_free_6dof').id),
  );
  await page.locator('.joint-entry.selected').filter({ hasText: free }).waitFor();
  await page.evaluate(() =>
    window.review.select(window.review.doc.joints.find((j) => j.name === '03_ball_xyz').id),
  );
  await card(ball).locator('.entity-card-heading[aria-expanded=true]').waitFor();
  await assertOpen(ball, true);
  assert.equal(
    await page.evaluate(() => window.review.begins),
    0,
    'Disclosure does not create a document transaction',
  );
});

test('native button Enter/Space activation toggles once, owns its keys, and double-click retains focus', async () => {
  const button = heading(ball);
  await button.focus();
  await button.press('Enter');
  await assertOpen(ball, true);
  await button.press('Space');
  await assertOpen(ball, false);
  await button.press('Space');
  await assertOpen(ball, true);
  assert.equal(
    await page.evaluate(() => window.review.leakedKeys),
    0,
    'Header activation must not also trigger application playback',
  );
  await button.click();
  await assertOpen(ball, false);
  await button.dblclick();
  await assertOpen(ball, true);
  assert.equal(await page.evaluate(() => window.review.focus.length), 1);
  await button.click({ button: 'right' });
  await assertOpen(ball, true);
  assert.equal(await page.evaluate(() => window.review.clipboard), ball);
});

test('numeric input and reset actions remain outside disclosure ownership; scalar headers still select and focus', async () => {
  await heading(ball).click();
  const field = card(ball).getByRole('textbox', { name: 'Rotation X', exact: true });
  await field.fill('20');
  await field.press('Enter');
  await assertOpen(ball, true);
  assert.equal(Number(await field.inputValue()), 20);
  await field.hover();
  await card(ball).getByRole('button', { name: 'Reset Rotation X', exact: true }).click();
  await assertOpen(ball, true);
  assert.equal(Number(await field.inputValue()), 0);
  assert.equal(await page.evaluate(() => window.review.focus.length), 0);
  const scalar = page.locator('.joint-entry').first(),
    title = scalar.locator('.entity-card-heading');
  await title.dblclick();
  assert.equal(await title.getAttribute('aria-expanded'), null);
  assert.equal(await scalar.getByRole('slider').count(), 1);
  assert.match(await scalar.getAttribute('class'), /selected/);
  assert.equal(await page.evaluate(() => window.review.focus.length), 1);
});

test('whole-header hover and keyboard focus keep derived corners in dark/light and fractional panes', async () => {
  for (const theme of ['dark', 'light']) {
    await page.evaluate((theme) => {
      window.reviewTheme(theme);
      document.querySelector('#pane').style.zoom = '1.25';
    }, theme);
    await heading(ball).hover();
    const geometry = await heading(ball).evaluate((el) => {
      const s = window.paintStyle(el),
        h = el.closest('.entity-card-header').getBoundingClientRect(),
        card = el.closest('.joint-entry').getBoundingClientRect();
      return {
        radius: s.borderRadius,
        paint: s.backgroundColor,
        left: h.left - card.left,
        right: card.right - h.right,
        width: h.width,
        height: h.height,
      };
    });
    assert.equal(geometry.radius, '4px');
    assert.notEqual(geometry.paint, 'rgba(0, 0, 0, 0)');
    assert.equal(geometry.left, 10);
    assert.equal(geometry.right, 10);
    assert.equal(geometry.height, 35);
    await card(ball).screenshot({
      path: `../output/electron/verification/joint-heading-${theme}-hover-125.png`,
    });
    await heading(ball).focus();
    await page.keyboard.press('Space');
    await assertOpen(ball, theme === 'dark');
  }
});
