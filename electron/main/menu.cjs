const { app, Menu, nativeTheme, nativeImage, shell } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
const catalog = require('./menu-catalog.json');
const labels = { en: require('./menu-en.json'), 'zh-CN': require('./menu-zh.json') };
const DOCUMENTATION_URL = 'https://github.com/acrlw/mojive#readme';
function openDocumentation(url = DOCUMENTATION_URL) {
  if (url !== DOCUMENTATION_URL) return Promise.resolve(false);
  return shell.openExternal(DOCUMENTATION_URL).then(() => true);
}

// One catalog owns the native and in-window menu hierarchy. Platform roles are
// added here; scene commands and capability checks remain in the renderer.
const MENU_ICONS = { 'role.copy': 'copy', 'role.paste': 'paste' };
function collectIcons(entries) {
  for (const entry of entries) {
    const id = entry.action ?? `menu:${entry.label}`;
    if (entry.icon) MENU_ICONS[id] = entry.icon;
    if (entry.activeIcon) MENU_ICONS[`${id}:active`] = entry.activeIcon;
    if (entry.children) collectIcons(entry.children);
  }
}
for (const group of catalog) collectIcons(group.items);
const menuImages = new Map();
function menuIcon(action) {
  const name = MENU_ICONS[action];
  if (!name) return undefined;
  if (!menuImages.has(name)) {
    const directory = path.resolve(__dirname, '..', app.isPackaged ? 'dist' : 'public', 'menu-icons');
    const image = nativeImage.createEmpty();
    for (const scaleFactor of [1, 2]) {
      const filename = path.join(directory, `${name}${scaleFactor === 2 ? '@2x' : ''}.png`);
      image.addRepresentation({ scaleFactor, buffer: fs.readFileSync(filename) });
    }
    image.setTemplateImage(true);
    menuImages.set(name, image);
  }
  return menuImages.get(name);
}
const defaults = {
  locale: 'en', theme: 'dark', blocked: false, canUndo: false, canRedo: false,
  hasSelection: false, canAdd: false, canDelete: false, hasProject: false, hasDocument: false, selectionVisible: false,
  showLeft: true, showInspector: true, showTimeline: true, showOutput: false,
  showGizmos: true, projection: 'perspective', tool: 'select', frame: 'world', panel: 'scene',
  hasSimulation: false, simulationPlaying: false, simulationRecording: false,
  canToggleSimulation: false, canSimulationPrevious: false, canSimulationStep: false,
  simulationSpeed: 1, sceneId: '', textInput: false, scopedKeyboard: false, focusKey: 'F', focusAllKey: 'Shift+F',
};
const requiredBooleans = ['canUndo', 'canRedo', 'hasSelection', 'showLeft', 'showInspector', 'showTimeline'];
const enums = {
  projection: ['perspective', 'orthographic'],
  tool: ['select', 'position', 'rotation', 'scale', 'joint'],
  frame: ['world', 'body'],
  panel: ['', 'scene', 'project', 'assets', 'joints', 'actuators', 'cameras', 'layers'],
};
function normalizeState(next) {
  if (!next || typeof next !== 'object' || Array.isArray(next)) throw new Error('Invalid native menu state.');
  if (!['en', 'zh-CN'].includes(next.locale) || !['dark', 'light'].includes(next.theme))
    throw new Error('Invalid native menu locale or theme.');
  for (const key of requiredBooleans)
    if (typeof next[key] !== 'boolean') throw new Error(`Invalid native menu state: ${key}.`);
  const result = { ...defaults };
  for (const [key, fallback] of Object.entries(defaults)) {
    if (next[key] === undefined || key === 'focusKey' || key === 'focusAllKey') continue;
    if (typeof next[key] !== typeof fallback) throw new Error(`Invalid native menu state: ${key}.`);
    if (enums[key] && !enums[key].includes(next[key])) throw new Error(`Invalid native menu state: ${key}.`);
    result[key] = next[key];
  }
  if (!Number.isFinite(result.simulationSpeed) || result.simulationSpeed <= 0)
    throw new Error('Invalid native simulation speed.');
  for (const key of ['focusKey', 'focusAllKey']) {
    if (next[key] !== undefined && next[key] !== null &&
        (typeof next[key] !== 'string' || !/^(?:(?:Control|Alt|Shift|Meta)\+)*(?:[A-Z0-9]|F(?:[1-9]|1[0-2])|Space|ArrowLeft|ArrowRight)$/.test(next[key])))
      throw new Error('Invalid native focus shortcut.');
  }
  result.focusKey = next.focusKey === undefined ? 'F' : next.focusKey;
  result.focusAllKey = next.focusAllKey === undefined
    ? result.focusKey && !result.focusKey.split('+').includes('Shift') ? `Shift+${result.focusKey}` : null
    : next.focusAllKey;
  return result;
}
function createMenuController(getWindow, development = false) {
  let state = { ...defaults };
  const t = (key) => labels[state.locale][key] ?? labels.en[key] ?? key;
  const send = (action) => {
    if (state.blocked) return;
    if (action === 'help.documentation') {
      void openDocumentation().catch((error) => console.error('Could not open documentation:', error));
      return;
    }
    const window = getWindow();
    if (window && !window.isDestroyed()) window.webContents.send('mojive:menu-action', action);
  };
  const active = (entry) => entry.checked && state[entry.checked.key] === entry.checked.value;
  const resolve = (entry) => {
    if (entry.type === 'separator') return { type: 'separator' };
    const id = entry.action ?? `menu:${entry.label}`;
    const checked = active(entry);
    const result = {
      id,
      label: t(checked && entry.activeLabel ? entry.activeLabel : entry.label),
      icon: menuIcon(checked && entry.activeIcon ? `${id}:active` : id),
      enabled: !state.blocked && (entry.enabled ? Boolean(state[entry.enabled]) : true),
    };
    if (entry.children) result.submenu = entry.children.map(resolve);
    else result.click = () => send(entry.action);
    if (entry.type) { result.type = entry.type; result.checked = Boolean(checked); }
    if (entry.accelerator) result.accelerator = entry.accelerator;
    if (entry.shortcut) result.accelerator = state.textInput || state.scopedKeyboard ? undefined : state[entry.shortcut]?.replaceAll('Meta', 'Command').replaceAll('ArrowLeft', 'Left').replaceAll('ArrowRight', 'Right') || undefined;
    const editingMethod = { 'edit.undo': 'undo', 'edit.redo': 'redo', 'edit.delete': 'delete' }[entry.action];
    if (editingMethod) {
      result.enabled = state.textInput || result.enabled;
      if ((state.textInput || state.scopedKeyboard) && editingMethod === 'delete') result.accelerator = undefined;
      result.click = () => {
        if (state.textInput) getWindow()?.webContents[editingMethod]();
        else send(entry.action);
      };
    }
    return result;
  };
  const role = (name) => ({ role: name, label: t(`role_${name}`), icon: menuIcon(`role.${name}`) });
  const rebuild = () => {
    nativeTheme.themeSource = state.theme;
    if (process.platform !== 'darwin') { Menu.setApplicationMenu(null); return; }
    const groups = catalog.map((group) => {
      const submenu = group.items.map(resolve);
      if (group.name === 'File') submenu.push({ type: 'separator' }, role('close'));
      if (group.name === 'Edit') submenu.splice(2, 0, { type: 'separator' }, ...['cut', 'copy', 'paste', 'selectAll'].map((name) => state.scopedKeyboard && !state.textInput
        ? { id: `role.${name}`, label: t(`role_${name}`), icon: menuIcon(`role.${name}`), enabled: false }
        : role(name)));
      if (group.name === 'Window') submenu.push(
        { type: 'separator' }, ...['minimize', 'zoom', 'front', 'togglefullscreen'].map(role),
        ...(development ? [role('toggleDevTools')] : []),
      );
      // AppKit owns the native Help search field, including its sizing.
      return { label: t(group.label), ...(group.name === 'Help' ? { role: 'help' } : {}), submenu };
    });
    Menu.setApplicationMenu(Menu.buildFromTemplate([
      { label: 'Mojive', submenu: [
        resolve({ action: 'app.about', label: 'about', icon: 'info' }),
        { type: 'separator' }, { ...resolve({ action: 'app.settings', label: 'settings' }), accelerator: 'Command+,' },
        { type: 'separator' }, role('services'), { type: 'separator' },
        ...['hide', 'hideOthers', 'unhide'].map(role), { type: 'separator' }, role('quit'),
      ] },
      ...groups,
    ]));
  };
  rebuild();
  return { t, update(next) {
    const updated = normalizeState(next);
    if (JSON.stringify(updated) !== JSON.stringify(state)) { state = updated; rebuild(); }
  } };
}
module.exports = { createMenuController, MENU_ICONS, DOCUMENTATION_URL, openDocumentation };
