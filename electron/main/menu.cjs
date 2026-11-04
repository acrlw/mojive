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
  en: {
    file: 'File',
    edit: 'Edit',
    add: 'Add',
    view: 'View',
    window: 'Window',
    help: 'Help',
    settings: 'Settings…',
    about: 'About Mojive',
    openUrdf: 'Open URDF…',
    openProject: 'Open Project Folder…',
    closeProject: 'Close Project Folder',
    openScene: 'Open Saved Scene…',
    save: 'Save Scene…',
    examples: 'Example Scenes',
    joint_gizmo: 'Joint Gizmos',
    atelier_arm: 'Atelier Arm',
    geometry_lab: 'Geometry Comparison',
    primitive_gallery: 'Scene Objects',
    undo: 'Undo',
    redo: 'Redo',
    delete: 'Delete Selection',
    box: 'Box',
    sphere: 'Sphere',
    cylinder: 'Cylinder',
    capsule: 'Capsule',
    ground: 'Ground',
    light: 'Light',
    camera: 'Camera',
    focusSelection: 'Frame Selection',
    focusAll: 'Frame All',
    front: 'Front',
    back: 'Back',
    left: 'Left',
    right: 'Right',
    top: 'Top',
    bottom: 'Bottom',
    perspective: 'Perspective',
    orthographic: 'Orthographic',
    toggleProjection: 'Toggle Projection',
    toggleLeft: 'Show Left Panel',
    toggleInspector: 'Show Inspector',
    toggleTimeline: 'Show Timeline',
    appearance: 'Appearance',
    dark: 'Dark',
    lightTheme: 'Light',
    language: 'Language',
    interaction: 'Interaction Guide',
    keepEditing: 'Keep Editing',
    discard: 'Discard Changes',
    unsaved: 'Unsaved Scene',
    discardQuestion: 'Discard unsaved changes and close Mojive?',
    discardDetail: 'Save the scene before closing to keep your edits.',
  },
  'zh-CN': {
    file: '文件',
    edit: '编辑',
    add: '添加',
    view: '视图',
    window: '窗口',
    help: '帮助',
    settings: '设置…',
    about: '关于 Mojive',
    openUrdf: '打开 URDF…',
    openProject: '打开项目文件夹…',
    closeProject: '关闭项目文件夹',
    openScene: '打开已保存场景…',
    save: '保存场景…',
    examples: '示例场景',
    joint_gizmo: '关节操作示例',
    atelier_arm: 'Atelier 机械臂',
    geometry_lab: '几何显示对比',
    primitive_gallery: '场景对象',
    undo: '撤销',
    redo: '重做',
    delete: '删除所选对象',
    box: '立方体',
    sphere: '球体',
    cylinder: '圆柱体',
    capsule: '胶囊体',
    ground: '地面',
    light: '灯光',
    camera: '相机',
    focusSelection: '聚焦所选对象',
    focusAll: '聚焦全部',
    front: '前视图',
    back: '后视图',
    left: '左视图',
    right: '右视图',
    top: '顶视图',
    bottom: '底视图',
    perspective: '透视投影',
    orthographic: '正交投影',
    toggleProjection: '切换投影',
    toggleLeft: '显示左侧面板',
    toggleInspector: '显示检查器',
    toggleTimeline: '显示时间轴',
    appearance: '外观',
    dark: '深色',
    lightTheme: '浅色',
    language: '语言',
    interaction: '交互指南',
    keepEditing: '继续编辑',
    discard: '放弃更改',
    unsaved: '场景尚未保存',
    discardQuestion: '放弃尚未保存的更改并关闭 Mojive？',
    discardDetail: '关闭前保存场景即可保留编辑内容。',
  },
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
            { role: 'services' },
            { type: 'separator' },
            { role: 'hide' },
            { role: 'hideOthers' },
            { role: 'unhide' },
            { type: 'separator' },
            { role: 'quit' },
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
            { role: 'close' },
          ],
        },
        {
          label: t('edit'),
          submenu: [
            editing('edit.undo', 'undo', 'undo', state.canUndo, 'Command+Z'),
            editing('edit.redo', 'redo', 'redo', state.canRedo, 'Command+Shift+Z'),
            { type: 'separator' },
            { role: 'cut' },
            { role: 'copy', icon: menuIcon('role.copy') },
            { role: 'paste', icon: menuIcon('role.paste') },
            { role: 'selectAll' },
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
              accelerator: state.textInput ? undefined : 'F',
            }),
            item('view.focusAll', 'focusAll', {
              accelerator: state.textInput ? undefined : 'Shift+F',
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
            { role: 'togglefullscreen' },
            ...(development ? [{ role: 'toggleDevTools' }] : []),
          ],
        },
        {
          label: t('window'),
          submenu: [
            { role: 'minimize' },
            { role: 'zoom' },
            { type: 'separator' },
            { role: 'front' },
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
      const updated = {
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
