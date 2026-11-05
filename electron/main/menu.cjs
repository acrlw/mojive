const { app, Menu, nativeTheme, nativeImage } = require('electron');
const path = require('node:path');
const fs = require('node:fs');
// Semantic masters are rasterized as complete 24-unit canvases at 1× and 2×.
// Template images let AppKit own selected, disabled, light, and dark menu ink.
const MENU_ICONS = {
  'app.settings': 'settings',
  'file.openUrdf': 'file',
  'file.openProject': 'folder',
  'file.closeProject': 'close',
  'file.openScene': 'scene',
  'file.save': 'save',
  'file.examples': 'scene',
  'edit.undo': 'undo',
  'edit.redo': 'redo',
  'edit.delete': 'delete',
  'role.copy': 'copy',
  'role.paste': 'paste',
  'add.box': 'shape-box',
  'add.sphere': 'shape-sphere',
  'add.cylinder': 'shape-cylinder',
  'add.capsule': 'shape-capsule',
  'add.ground': 'shape-plane',
  'add.light': 'light',
  'add.camera': 'camera',
  'view.focusSelection': 'target',
  'view.focusAll': 'fit',
  'view.front': 'view-front',
  'view.back': 'view-back',
  'view.left': 'view-left',
  'view.right': 'view-right',
  'view.top': 'view-top',
  'view.bottom': 'view-bottom',
  'view.perspective': 'perspective',
  'view.orthographic': 'orthographic',
  'view.toggleLeft': 'sidebar-left',
  'view.toggleInspector': 'inspect',
  'view.toggleTimeline': 'waveform',
  'help.interaction': 'help',
};
const menuImages = new Map();
function menuIcon(action) {
  const name = MENU_ICONS[action];
  if (!name) return undefined;
  if (!menuImages.has(name)) {
    const directory = path.resolve(
      __dirname,
      '..',
      app.isPackaged ? 'dist' : 'public',
      'menu-icons',
    );
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
const labels = {
  en: require('./menu-en.json'),
  'zh-CN': require('./menu-zh.json'),
};
function createMenuController(getWindow, development = false) {
  let state = {
    locale: 'en',
    theme: 'dark',
    canUndo: false,
    canRedo: false,
    hasSelection: false,
    hasProject: false,
    showLeft: true,
    showInspector: true,
    showTimeline: true,
    sceneId: 'joint_gizmo',
    textInput: false,
    focusKey: 'F',
    focusAllKey: 'Shift+F',
  };
  const t = (key) => labels[state.locale][key] ?? labels.en[key] ?? key;
  const send = (action) => {
    const window = getWindow();
    if (window && !window.isDestroyed()) window.webContents.send('mojive:menu-action', action);
  };
  const item = (action, key, options = {}) => ({
    id: action,
    label: t(key),
    icon: menuIcon(action),
    click: () => send(action),
    ...options,
  });
  const editing = (action, key, method, enabled, accelerator) =>
    item(action, key, {
      enabled: state.textInput || enabled,
      accelerator: state.textInput && method === 'delete' ? undefined : accelerator,
      click: () => {
        if (state.textInput) getWindow()?.webContents[method]();
        else send(action);
      },
    });
  const rebuild = () => {
    nativeTheme.themeSource = state.theme;
    if (process.platform !== 'darwin') {
      Menu.setApplicationMenu(null);
      return;
    }
    Menu.setApplicationMenu(
      Menu.buildFromTemplate([
        {
          label: 'Mojive',
          submenu: [
            item('app.about', 'about'),
            { type: 'separator' },
            item('app.settings', 'settings', { accelerator: 'Command+,' }),
            { type: 'separator' },
            { role: 'services', label: t('role_services') },
            { type: 'separator' },
            { role: 'hide', label: t('role_hide') },
            { role: 'hideOthers', label: t('role_hideOthers') },
            { role: 'unhide', label: t('role_unhide') },
            { type: 'separator' },
            { role: 'quit', label: t('role_quit') },
          ],
        },
        {
          label: t('file'),
          submenu: [
            item('file.openUrdf', 'openUrdf', { accelerator: 'Command+O' }),
            item('file.openProject', 'openProject', { accelerator: 'Command+Shift+O' }),
            item('file.closeProject', 'closeProject', { enabled: state.hasProject }),
            item('file.openScene', 'openScene'),
            { type: 'separator' },
            item('file.save', 'save', { accelerator: 'Command+S' }),
            { type: 'separator' },
            {
              label: t('examples'),
              icon: menuIcon('file.examples'),
              submenu: ['joint_gizmo', 'atelier_arm', 'geometry_lab', 'primitive_gallery'].map(
                (id) => item(`scene.${id}`, id, { type: 'radio', checked: state.sceneId === id }),
              ),
            },
            { type: 'separator' },
            { role: 'close', label: t('role_close') },
          ],
        },
        {
          label: t('edit'),
          submenu: [
            editing('edit.undo', 'undo', 'undo', state.canUndo, 'Command+Z'),
            editing('edit.redo', 'redo', 'redo', state.canRedo, 'Command+Shift+Z'),
            { type: 'separator' },
            { role: 'cut', label: t('role_cut') },
            { role: 'copy', label: t('role_copy'), icon: menuIcon('role.copy') },
            { role: 'paste', label: t('role_paste'), icon: menuIcon('role.paste') },
            { role: 'selectAll', label: t('role_selectAll') },
            { type: 'separator' },
            editing('edit.delete', 'delete', 'delete', state.hasSelection, 'Backspace'),
          ],
        },
        {
          label: t('add'),
          submenu: [
            ...['box', 'sphere', 'cylinder', 'capsule'].map((id) => item(`add.${id}`, id)),
            { type: 'separator' },
            ...['ground', 'light', 'camera'].map((id) => item(`add.${id}`, id)),
          ],
        },
        {
          label: t('view'),
          submenu: [
            item('view.focusSelection', 'focusSelection', {
              enabled: state.hasSelection,
              accelerator: state.textInput
                ? undefined
                : state.focusKey?.replaceAll('Meta', 'Command') || undefined,
            }),
            item('view.focusAll', 'focusAll', {
              accelerator: state.textInput
                ? undefined
                : state.focusAllKey?.replaceAll('Meta', 'Command') || undefined,
            }),
            { type: 'separator' },
            ...[
              'front',
              'back',
              'left',
              'right',
              'top',
              'bottom',
              'perspective',
              'orthographic',
            ].map((id) => item(`view.${id}`, id)),
            item('view.toggleProjection', 'toggleProjection'),
            { type: 'separator' },
            ...['Left', 'Inspector', 'Timeline'].map((suffix) =>
              item(`view.toggle${suffix}`, `toggle${suffix}`, {
                type: 'checkbox',
                checked: state[`show${suffix}`],
              }),
            ),
            { type: 'separator' },
            {
              label: t('appearance'),
              submenu: [
                item('theme.dark', 'dark', { type: 'radio', checked: state.theme === 'dark' }),
                item('theme.light', 'lightTheme', {
                  type: 'radio',
                  checked: state.theme === 'light',
                }),
              ],
            },
            {
              label: t('language'),
              submenu: [
                {
                  id: 'locale.en',
                  label: 'English',
                  type: 'radio',
                  checked: state.locale === 'en',
                  click: () => send('locale.en'),
                },
                {
                  id: 'locale.zh-CN',
                  label: '简体中文',
                  type: 'radio',
                  checked: state.locale === 'zh-CN',
                  click: () => send('locale.zh-CN'),
                },
              ],
            },
            { type: 'separator' },
            { role: 'togglefullscreen', label: t('role_togglefullscreen') },
            ...(development ? [{ role: 'toggleDevTools', label: t('role_toggleDevTools') }] : []),
          ],
        },
        {
          label: t('window'),
          submenu: [
            { role: 'minimize', label: t('role_minimize') },
            { role: 'zoom', label: t('role_zoom') },
            { type: 'separator' },
            { role: 'front', label: t('role_front') },
          ],
        },
        { label: t('help'), role: 'help', submenu: [item('help.interaction', 'interaction')] },
      ]),
    );
  };
  rebuild();
  return {
    t,
    update(next) {
      if (!next || typeof next !== 'object' || Array.isArray(next))
        throw new Error('Invalid native menu state.');
      if (!['en', 'zh-CN'].includes(next.locale) || !['dark', 'light'].includes(next.theme))
        throw new Error('Invalid native menu locale or theme.');
      for (const key of [
        'canUndo',
        'canRedo',
        'hasSelection',
        'showLeft',
        'showInspector',
        'showTimeline',
      ])
        if (typeof next[key] !== 'boolean') throw new Error(`Invalid native menu state: ${key}.`);
      if (next.textInput !== undefined && typeof next.textInput !== 'boolean')
        throw new Error('Invalid native text-input state.');
      if (next.hasProject !== undefined && typeof next.hasProject !== 'boolean')
        throw new Error('Invalid native project state.');
      if (next.sceneId !== undefined && typeof next.sceneId !== 'string')
        throw new Error('Invalid native scene identity.');
      for (const key of ['focusKey', 'focusAllKey'])
        if (
          next[key] !== undefined &&
          next[key] !== null &&
          !/^(?:(?:Control|Alt|Shift|Meta)\+)*(?:[A-Z0-9]|F(?:[1-9]|1[0-2])|Space)$/.test(next[key])
        )
          throw new Error('Invalid native focus shortcut.');
      const focusKey = next.focusKey === undefined ? 'F' : next.focusKey;
      const updated = {
        focusKey,
        focusAllKey:
          next.focusAllKey === undefined
            ? focusKey && !focusKey.split('+').includes('Shift')
              ? `Shift+${focusKey}`
              : null
            : next.focusAllKey,
        locale: next.locale,
        theme: next.theme,
        canUndo: next.canUndo,
        canRedo: next.canRedo,
        hasSelection: next.hasSelection,
        hasProject: next.hasProject ?? false,
        showLeft: next.showLeft,
        showInspector: next.showInspector,
        showTimeline: next.showTimeline,
        sceneId: next.sceneId ?? '',
        textInput: next.textInput ?? false,
      };
      if (JSON.stringify(updated) !== JSON.stringify(state)) {
        state = updated;
        rebuild();
      }
    },
  };
}
module.exports = { createMenuController, MENU_ICONS };
