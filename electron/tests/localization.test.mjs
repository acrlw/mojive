import assert from 'node:assert/strict';
import { before, after, test } from 'node:test';
import { readFile, readdir, mkdir } from 'node:fs/promises';
import ts from 'typescript';
import { createServer } from 'vite';
import { chromium } from 'playwright';

let server, browser, base;
const out = '../output/electron/verification/input-audit';
before(async () => {
  await mkdir(out, { recursive: true });
  server = await createServer({
    server: { host: '127.0.0.1', port: 0, hmr: false },
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

test('source UI translation keys and menu resources have complete Chinese entries', async () => {
  const zh = {
    ...JSON.parse(await readFile('src/locales/zh.json', 'utf8')),
    ...JSON.parse(await readFile('src/locales/electron-zh.json', 'utf8')),
  };
  const missing = [];
  for (const file of (await readdir('src')).filter(
    (name) => /\.tsx?$/.test(name) && name !== 'icons.ts',
  )) {
    const source = ts.createSourceFile(
      file,
      await readFile(`src/${file}`, 'utf8'),
      ts.ScriptTarget.Latest,
      true,
      file.endsWith('tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
    );
    function visit(node) {
      if (
        ts.isCallExpression(node) &&
        ['t', 'message'].includes(node.expression.getText(source)) &&
        node.arguments[0] &&
        ts.isStringLiteral(node.arguments[0]) &&
        !zh[node.arguments[0].text]
      )
        missing.push(`${file}: ${node.arguments[0].text}`);
      ts.forEachChild(node, visit);
    }
    visit(source);
  }
  assert.deepEqual(missing, []);
  const enMenu = JSON.parse(await readFile('main/menu-en.json', 'utf8'));
  const zhMenu = JSON.parse(await readFile('main/menu-zh.json', 'utf8'));
  assert.deepEqual(Object.keys(enMenu).sort(), Object.keys(zhMenu).sort());
  for (const value of Object.values(zhMenu)) assert.match(value, /[\u3400-\u9fff]/);
});

test('structured messages switch languages without translating paths or interpolated user text', async () => {
  const page = await browser.newPage();
  try {
    await page.goto(base);
    const result = await page.evaluate(async () => {
      const { message, t, setLanguage } = await import('/src/i18n.ts');
      const value = message('Loaded {{name}} · {{objects}} objects · {{joints}} joints', {
        name: 'Saved {{name}} <robot>',
        objects: 45,
        joints: 9,
      });
      setLanguage('zh');
      const zh = t(value);
      setLanguage('en');
      return {
        zh,
        en: t(value),
        tag: document.documentElement.lang,
        file: t(message('Saved {{path}}', { path: '/data/Light & Camera.json' }), 'zh'),
        kind: t(message('Added {{kind}}', { kind: message('box') }), 'zh'),
      };
    });
    assert.equal(result.zh, '已加载 Saved {{name}} <robot> · 45 个对象 · 9 个关节');
    assert.equal(result.en, 'Loaded Saved {{name}} <robot> · 45 objects · 9 joints');
    assert.equal(result.tag, 'en');
    assert.equal(result.file, '已保存 /data/Light & Camera.json');
    assert.equal(result.kind, '已添加立方体');
  } finally {
    await page.close();
  }
});

test('Chinese Inspector and timeline localize enums and built-in titles while preserving scene names', async () => {
  const page = await browser.newPage({
    viewport: { width: 1540, height: 980 },
    deviceScaleFactor: 2,
  });
  page.setDefaultTimeout(7000);
  try {
    await page.goto(base);
    await page.locator('.tree-row[data-tooltip="01_revolute_y"]').click();
    assert.equal(await page.locator('.inspector .property-value').first().innerText(), '旋转关节');
    assert.equal(await page.locator('.timeline .lane-header').innerText(), '关节操作示例');
    assert.match(await page.locator('.document-title').innerText(), /关节操作示例/);
    assert.match(await page.locator('.inspector').innerText(), /01_revolute_anchor/);
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({ path: `${out}/localized-inspector-zh.png` });
    await page.locator('.tree-row[data-tooltip="joint_gizmos"]').click();
    assert.equal(await page.getByLabel('视点 X', { exact: true }).count(), 1);
    assert.equal(await page.getByLabel('目标 Z', { exact: true }).count(), 1);
    await page.locator('.tree-row[data-tooltip="01_revolute_y"]').click();

    await page.getByRole('button', { name: '输出', exact: true }).click();
    await page.locator('.output-message-text').first().waitFor();
    assert.match(
      await page.locator('.output-message-text').first().innerText(),
      /已加载 关节操作示例 · 45 个对象 · 9 个关节/,
    );
    await page.locator('.title-actions button:has([data-icon=settings])').click();
    await page.getByRole('combobox', { name: '语言', exact: true }).click();
    await page.getByRole('option', { name: 'English', exact: true }).click();
    await page.getByRole('button', { name: 'Close', exact: true }).click();
    assert.match(
      await page.locator('.output-message-text').first().innerText(),
      /Loaded Joint gizmos · 45 objects · 9 joints/,
    );
    assert.equal(
      await page.locator('.inspector .property-value').first().innerText(),
      'Hinge joint',
    );
    assert.equal(await page.evaluate(() => document.documentElement.lang), 'en');
    await page.reload();
    await page.locator('.tree-row').first().waitFor();
    assert.equal(await page.evaluate(() => document.documentElement.lang), 'en');
  } finally {
    await page.close();
  }
});
