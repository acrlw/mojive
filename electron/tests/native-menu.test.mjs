import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const menuFile = new URL('../main/menu.cjs', import.meta.url);
const localRequire = createRequire(menuFile);
function menuFixture(platform = 'darwin') {
  const captures = [], actions = [], edits = [], external = [], theme = {};
  const electron = {
    app: { isPackaged: false },
    nativeTheme: theme,
    nativeImage: { createEmpty: () => ({ addRepresentation() {}, setTemplateImage() {} }) },
    shell: { openExternal: async (url) => external.push(url) },
    Menu: {
      buildFromTemplate: (template) => template,
      setApplicationMenu: (menu) => captures.push(menu),
    },
  };
  const module = { exports: {} };
  vm.runInNewContext(readFileSync(menuFile, 'utf8'), {
    module, exports: module.exports,
    __dirname: fileURLToPath(new URL('../main/', import.meta.url)),
    process: { platform },
    require: (name) => name === 'electron' ? electron : name === 'node:fs'
      ? { readFileSync: () => Buffer.alloc(0) } : localRequire(name),
  }, { filename: fileURLToPath(menuFile) });
  const controller = module.exports.createMenuController(() => ({
    isDestroyed: () => false,
    webContents: {
      send: (channel, action) => actions.push({ channel, action }),
      undo: () => edits.push('undo'), redo: () => edits.push('redo'), delete: () => edits.push('delete'),
    },
  }));
  const flatten = (items) => items.flatMap((item) => [item, ...(item.submenu ? flatten(item.submenu) : [])]);
  const item = (id) => flatten(captures.at(-1)).find((entry) => entry.id === id);
  const group = (label) => captures.at(-1).find((entry) => entry.label === label).submenu;
  return { controller, actions, captures, theme, item, group, flatten, edits, external, api: module.exports };
}
const state = {
  locale: 'en', theme: 'dark', canUndo: false, canRedo: false, hasSelection: false,
  showLeft: true, showInspector: true, showTimeline: true,
};
const chineseState = { ...state, locale: 'zh-CN', theme: 'light' };

test('native groups and semantic boundaries match the shared native-parity catalog', () => {
  const { captures, group, item, flatten } = menuFixture();
  assert.deepEqual(Array.from(captures.at(-1), (entry) => entry.label),
    ['Mojive', 'File', 'Edit', 'Add', 'View', 'Simulate', 'Window', 'Help']);
  assert.deepEqual(Array.from(group('Add'), (entry) => entry.type === 'separator' ? 'separator' : entry.id),
    ['add.box', 'add.sphere', 'add.cylinder', 'add.capsule', 'add.cone', 'add.ellipsoid', 'add.ground', 'separator', 'add.light', 'add.camera']);
  assert.equal(group('View')[0].id, 'view.focusAll');
  assert.deepEqual(Array.from(item('menu:projection').submenu, (entry) => entry.id),
    ['view.perspective', 'view.orthographic']);
  assert.deepEqual(Array.from(item('menu:openModel').submenu, (entry) => entry.id),
    ['file.openSimulation', 'file.openUrdf']);
  assert.ok(flatten(group('Simulate')).some((entry) => entry.id === 'panel.actuators'),
    'The native Physics adapter command opens the physics control panel');
  assert.deepEqual(Array.from(item('menu:panels').submenu, (entry) => entry.type === 'separator' ? 'separator' : entry.id),
    ['panel.scene', 'panel.joints', 'panel.actuators', 'panel.project', 'panel.assets',
      'panel.cameras', 'panel.layers', 'panel.output', 'separator', 'view.toggleInspector', 'panel.timeline']);
});

test('Physics adapter opens the available physics controls and reflects panel visibility', () => {
  const { controller, item, actions } = menuFixture();
  controller.update({ ...state, panel: 'actuators' });
  assert.equal(item('panel.actuators').checked, true);
  item('panel.actuators').click();
  assert.equal(actions.at(-1).action, 'panel.actuators');
  controller.update({ ...state, panel: '', showLeft: false });
  assert.equal(item('panel.actuators').checked, false);
});

test('language updates preserve native Add separators and the shared Camera term', () => {
  const { controller, group, item, actions } = menuFixture();
  controller.update({ ...chineseState, canAdd: true });
  assert.equal(item('add.camera').label, '相机');
  assert.equal(item('add.ground').label, '平面');
  assert.equal(item('add.cone').label, '圆锥体');
  assert.equal(item('add.ellipsoid').label, '椭球体');
  assert.equal(group('添加')[7].type, 'separator');
  for (const kind of ['cone', 'ellipsoid']) {
    item(`add.${kind}`).click();
    assert.equal(actions.at(-1).action, `add.${kind}`);
  }
  item('add.camera').click();
  assert.deepEqual(actions.at(-1), { channel: 'mojive:menu-action', action: 'add.camera' });
});

test('project and document actions follow their own capabilities independently from window close', () => {
  const { controller, item, group, actions } = menuFixture();
  assert.equal(item('file.closeProject').enabled, false);
  assert.equal(item('file.save').enabled, false);
  controller.update({ ...chineseState, hasProject: true, hasDocument: true });
  assert.equal(item('file.closeProject').enabled, true);
  assert.equal(item('file.save').enabled, true);
  assert.equal(item('file.closeProject').label, '关闭项目文件夹');
  item('file.closeProject').click();
  assert.equal(actions.at(-1).action, 'file.closeProject');
  assert.ok(group('文件').some((entry) => entry.role === 'close'));
  controller.update({ ...chineseState, hasSelection: true });
  assert.equal(item('file.closeProject').enabled, false);
  assert.throws(() => controller.update({ ...chineseState, hasProject: 'yes' }), /Invalid native menu state: hasProject/);
});

test('checked state follows tools, projection, panels and live simulation capabilities', () => {
  const { controller, item } = menuFixture();
  controller.update({ ...state, hasSelection: true, selectionVisible: true, projection: 'orthographic',
    tool: 'rotation', frame: 'body', panel: 'cameras', hasSimulation: true, simulationPlaying: true,
    simulationRecording: true, simulationSpeed: 0.5, canToggleSimulation: true,
    canSimulationPrevious: true, canSimulationStep: true });
  for (const id of ['edit.visibility', 'view.orthographic', 'tool.rotation', 'frame.body',
    'panel.cameras', 'simulation.record', 'simulation.speed.0.5']) assert.equal(item(id).checked, true, id);
  assert.equal(item('view.perspective').checked, false);
  assert.equal(item('simulation.toggle').label, 'Pause simulation');
  assert.equal(item('simulation.record').label, 'Stop recording');
  assert.equal(item('simulation.previous').enabled, true);
  assert.equal(item('simulation.step').enabled, true);
  controller.update(state);
  assert.equal(item('simulation.previous').enabled, false);
  assert.equal(item('simulation.step').enabled, false);
});

test('modal blocking disables domain actions and suppresses stale native callbacks', () => {
  const { controller, item, actions } = menuFixture();
  const stale = item('add.box').click;
  controller.update({ ...state, blocked: true, canUndo: true, hasSelection: true, hasProject: true });
  for (const id of ['file.openScene', 'file.closeProject', 'add.box', 'edit.undo', 'view.focusSelection'])
    assert.equal(item(id).enabled, false, id);
  stale();
  item('edit.undo').click();
  assert.deepEqual(actions, []);
});

test('scene capabilities gate Add and Delete independently of selection', () => {
  const { controller, item } = menuFixture();
  controller.update({ ...state, hasSelection: true, canAdd: false, canDelete: false });
  assert.equal(item('add.box').enabled, false);
  assert.equal(item('edit.delete').enabled, false);
  controller.update({ ...state, hasSelection: true, canAdd: true, canDelete: true });
  assert.equal(item('add.box').enabled, true);
  assert.equal(item('edit.delete').enabled, true);
});

test('hiding the sidebar clears the checked panel without rejecting the state', () => {
  const { controller, item } = menuFixture();
  controller.update({ ...state, panel: 'cameras' });
  assert.equal(item('panel.cameras').checked, true);
  controller.update({ ...state, panel: '', showLeft: false });
  assert.equal(item('view.toggleLeft').checked, false);
  for (const panel of ['scene', 'project', 'assets', 'joints', 'actuators', 'cameras', 'layers'])
    assert.equal(item(`panel.${panel}`).checked, false, panel);
});

test('focus accelerators support remapping and unbinding while native text edits own undo/redo/delete', () => {
  const { controller, item, edits, actions } = menuFixture();
  controller.update({ ...state, focusKey: 'G' });
  assert.equal(item('view.focusSelection').accelerator, 'G');
  assert.equal(item('view.focusAll').accelerator, 'Shift+G');
  controller.update({ ...state, focusKey: 'F', focusAllKey: null });
  assert.equal(item('view.focusAll').accelerator, undefined);
  controller.update({ ...state, focusKey: null });
  assert.equal(item('view.focusSelection').accelerator, undefined);
  assert.equal(item('view.focusAll').accelerator, undefined);
  controller.update({ ...state, focusKey: 'R', textInput: true });
  assert.equal(item('view.focusSelection').accelerator, undefined);
  assert.equal(item('view.focusAll').accelerator, undefined);
  for (const id of ['edit.undo', 'edit.redo', 'edit.delete']) {
    assert.equal(item(id).enabled, true);
    item(id).click();
  }
  assert.equal(item('edit.delete').accelerator, undefined);
  assert.deepEqual(edits, ['undo', 'redo', 'delete']);
  assert.deepEqual(actions, []);
  assert.throws(() => controller.update({ ...state, focusKey: 'Cmd+Q' }), /Invalid native focus/);
});

test('configured directional focus keys translate to Electron accelerator names', () => {
  const { controller, item } = menuFixture();
  for (const [input, expected] of [['ArrowLeft', 'Left'], ['ArrowRight', 'Right'],
    ['Meta+ArrowLeft', 'Command+Left'], ['Control+Alt+ArrowRight', 'Control+Alt+Right'], ['Space', 'Space']]) {
    controller.update({ ...state, focusKey: input });
    assert.equal(item('view.focusSelection').accelerator, expected);
    assert.equal(item('view.focusAll').accelerator, `Shift+${expected}`);
  }
});

test('modal text fields retain native edits while scene commands remain blocked', () => {
  const { controller, item, edits, actions } = menuFixture();
  controller.update({ ...state, blocked: true, textInput: true, canUndo: false, canRedo: false });
  for (const id of ['edit.undo', 'edit.redo', 'edit.delete']) {
    assert.equal(item(id).enabled, true, id);
    item(id).click();
  }
  assert.deepEqual(edits, ['undo', 'redo', 'delete']);
  assert.deepEqual(actions, []);
  assert.equal(item('file.openScene').enabled, false);
  assert.equal(item('add.box').enabled, false);
});

test('timeline focus preserves local focus, delete and clipboard keyboard ownership', () => {
  const { controller, item, flatten, captures } = menuFixture();
  controller.update({ ...state, scopedKeyboard: true, hasSelection: true, canDelete: true });
  for (const id of ['view.focusSelection', 'view.focusAll', 'edit.delete']) {
    assert.equal(item(id).accelerator, undefined, id);
    assert.equal(item(id).enabled, true, `${id} is still available as an explicit scene menu action`);
  }
  for (const name of ['cut', 'copy', 'paste', 'selectAll']) {
    assert.equal(item(`role.${name}`).enabled, false);
    assert.equal(item(`role.${name}`).role, undefined);
    assert.equal(item(`role.${name}`).accelerator, undefined);
  }
  controller.update({ ...state, scopedKeyboard: true, textInput: true });
  for (const name of ['cut', 'copy', 'paste', 'selectAll'])
    assert.ok(flatten(captures.at(-1)).some((entry) => entry.role === name), name);
  controller.update(state);
  assert.equal(item('view.focusSelection').accelerator, 'F');
  assert.equal(item('edit.delete').accelerator, 'Backspace');
});

test('native system-role labels follow the app language independently of the host language', () => {
  const { controller, captures, flatten } = menuFixture();
  controller.update(chineseState);
  const roles = flatten(captures.at(-1)).filter((entry) => entry.role && entry.role !== 'help');
  assert.ok(roles.length >= 14);
  for (const item of roles) assert.match(item.label, /[\u3400-\u9fff]/, item.role);
  assert.equal(roles.find((item) => item.role === 'copy').label, '复制');
  controller.update(state);
  assert.equal(flatten(captures.at(-1)).find((entry) => entry.role === 'copy').label, 'Copy');
});

test('documentation opens only the fixed project URL and non-macOS leaves the menu to the renderer', async () => {
  const { api, external } = menuFixture();
  assert.equal(await api.openDocumentation('https://example.com'), false);
  assert.deepEqual(external, []);
  assert.equal(await api.openDocumentation(), true);
  assert.deepEqual(external, ['https://github.com/acrlw/mojive#readme']);
  assert.equal(menuFixture('linux').captures.at(-1), null);
});
