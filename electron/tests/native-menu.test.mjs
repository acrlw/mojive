import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const menuFile = new URL('../main/menu.cjs', import.meta.url);
const localRequire = createRequire(menuFile);
function menuFixture() {
  const captures = [],
    actions = [],
    theme = {};
  const electron = {
    app: { isPackaged: false },
    nativeTheme: theme,
    nativeImage: { createEmpty: () => ({ addRepresentation() {}, setTemplateImage() {} }) },
    Menu: {
      buildFromTemplate: (template) => template,
      setApplicationMenu: (menu) => captures.push(menu),
    },
  };
  const module = { exports: {} };
  vm.runInNewContext(
    readFileSync(menuFile, 'utf8'),
    {
      module,
      exports: module.exports,
      __dirname: fileURLToPath(new URL('../main/', import.meta.url)),
      process: { platform: 'darwin' },
      require: (name) => (name === 'electron' ? electron : localRequire(name)),
    },
    { filename: fileURLToPath(menuFile) },
  );
  const controller = module.exports.createMenuController(() => ({
    isDestroyed: () => false,
    webContents: { send: (channel, action) => actions.push({ channel, action }) },
  }));
  return { controller, actions, captures, theme };
}
const chineseState = {
  locale: 'zh-CN',
  theme: 'light',
  canUndo: false,
  canRedo: false,
  hasSelection: false,
  showLeft: true,
  showInspector: true,
  showTimeline: true,
};

test('native Add groups primitive objects separately from ground, light and camera, matching frontend order', () => {
  const { captures } = menuFixture(),
    add = captures.at(-1).find((item) => item.label === 'Add').submenu;
  const items = Array.from(add, (item) => (item.type === 'separator' ? 'separator' : item.id));
  assert.deepEqual(items, [
    'add.box',
    'add.sphere',
    'add.cylinder',
    'add.capsule',
    'separator',
    'add.ground',
    'add.light',
    'add.camera',
  ]);
  const frontend = readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8');
  const definition = frontend.match(/const additions:[\s\S]*?= \[([\s\S]*?)\];/)[1];
  const frontendIds = [...definition.matchAll(/kind: '([^']+)'/g)].map(
    (match) => `add.${match[1]}`,
  );
  assert.deepEqual(
    items.filter((id) => id !== 'separator'),
    frontendIds,
  );
  assert.match(
    frontend,
    /a\.kind === 'ground' && <hr \/>/,
    'The frontend uses the same environment-group boundary',
  );
});

test('language updates preserve Add separators and use the same Camera term as the workspace', () => {
  const { controller, captures, actions } = menuFixture();
  controller.update(chineseState);
  const add = captures.at(-1).find((item) => item.label === '添加').submenu;
  const nativeCamera = add.find((item) => item.id === 'add.camera');
  const workspace = JSON.parse(
    readFileSync(new URL('../src/locales/zh.json', import.meta.url), 'utf8'),
  );
  assert.equal(nativeCamera.label, workspace.Camera);
  assert.equal(nativeCamera.label, '相机');
  assert.equal(add[4].type, 'separator');
  assert.equal(add[5].label, '地面');
  nativeCamera.click();
  assert.equal(actions.at(-1).channel, 'mojive:menu-action');
  assert.equal(actions.at(-1).action, 'add.camera');
  const catalog = readFileSync(new URL('../src/SceneCatalog.ts', import.meta.url), 'utf8');
  assert.ok(!catalog.includes('摄像机'), 'Scene catalog descriptions use the shared camera term');
});

test('Close Project Folder follows project state independently from scene selection and window close', () => {
  const { controller, captures, actions } = menuFixture();
  const fileMenu = () =>
    captures.at(-1).find((item) => item.submenu?.some((child) => child.id === 'file.openProject'))
      .submenu;
  assert.equal(fileMenu().find((item) => item.id === 'file.closeProject').enabled, false);
  controller.update({ ...chineseState, hasProject: true });
  const close = fileMenu().find((item) => item.id === 'file.closeProject');
  assert.equal(close.enabled, true);
  assert.equal(close.label, '关闭项目文件夹');
  assert.ok(close.icon);
  close.click();
  assert.equal(actions.at(-1).action, 'file.closeProject');
  assert.ok(
    fileMenu().some((item) => item.role === 'close'),
    'Window close remains a separate system action',
  );
  controller.update({ ...chineseState, hasProject: false, hasSelection: true });
  assert.equal(fileMenu().find((item) => item.id === 'file.closeProject').enabled, false);
  assert.throws(
    () => controller.update({ ...chineseState, hasProject: 'yes' }),
    /Invalid native project state/,
  );
});

test('native focus accelerators follow remapping, unbinding and text input ownership', () => {
  const { controller, captures } = menuFixture();
  const focus = () =>
    captures
      .at(-1)
      .flatMap((item) => item.submenu ?? [])
      .filter((item) => item.id?.startsWith('view.focus'));
  controller.update({ ...chineseState, focusKey: 'G' });
  assert.deepEqual(
    Array.from(focus(), (item) => item.accelerator),
    ['G', 'Shift+G'],
  );
  controller.update({ ...chineseState, focusKey: 'F', focusAllKey: null });
  assert.equal(focus()[1].accelerator, undefined);
  controller.update({ ...chineseState, focusKey: null });
  assert.ok(focus().every((item) => item.accelerator === undefined));
  controller.update({ ...chineseState, focusKey: 'R', textInput: true });
  assert.ok(focus().every((item) => item.accelerator === undefined));
  assert.throws(
    () => controller.update({ ...chineseState, focusKey: 'Cmd+Q' }),
    /Invalid native focus/,
  );
});

test('native system-role labels follow the app language independently of the host language', () => {
  const { controller, captures } = menuFixture();
  const roles = (items) =>
    items.flatMap((item) => [
      ...(item.role && item.role !== 'help' ? [item] : []),
      ...(Array.isArray(item.submenu) ? roles(item.submenu) : []),
    ]);
  controller.update(chineseState);
  const zh = roles(captures.at(-1));
  assert.ok(zh.length >= 14);
  for (const item of zh) assert.match(item.label, /[\u3400-\u9fff]/, item.role);
  assert.equal(zh.find((item) => item.role === 'copy').label, '复制');
  controller.update({ ...chineseState, locale: 'en' });
  assert.equal(roles(captures.at(-1)).find((item) => item.role === 'copy').label, 'Copy');
});
