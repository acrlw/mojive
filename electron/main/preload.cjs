const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('mojive', {
  platform: process.platform,
  browseFiles: (path, includeHidden) =>
    ipcRenderer.invoke('mojive:browse-files', path, includeHidden),
  openModelPath: (path) => ipcRenderer.invoke('mojive:open-model-path', path),
  openUrdfPath: (path) => ipcRenderer.invoke('mojive:open-urdf-path', path),
  openProjectPath: (path) => ipcRenderer.invoke('mojive:open-project-path', path),
  readProjectDirectory: (path) => ipcRenderer.invoke('mojive:read-project-directory', path),
  searchProject: (path, query) => ipcRenderer.invoke('mojive:search-project', path, query),
  searchModelAssets: (path, query) => ipcRenderer.invoke('mojive:search-model-assets', path, query),
  openScenePath: (path) => ipcRenderer.invoke('mojive:open-scene-path', path),
  saveScenePath: (path, content, overwrite) =>
    ipcRenderer.invoke('mojive:save-scene-path', path, content, overwrite),
  saveBinaryFile: (path, content, overwrite) =>
    ipcRenderer.invoke('mojive:save-binary-file', path, content, overwrite),
  confirmClose: () => ipcRenderer.invoke('mojive:confirm-close'),
  readClipboardText: () => ipcRenderer.invoke('mojive:read-clipboard-text'),
  writeClipboardText: (text) => ipcRenderer.invoke('mojive:write-clipboard-text', text),
  hasNativeMenu: process.platform === 'darwin',
  updateMenuState: (state) => ipcRenderer.invoke('mojive:update-menu', state),
  onMenuAction: (callback) => {
    if (typeof callback !== 'function')
      throw new TypeError('Menu action callback must be a function.');
    const listener = (_event, action) => callback(action);
    ipcRenderer.on('mojive:menu-action', listener);
    return () => ipcRenderer.removeListener('mojive:menu-action', listener);
  },
  readUrdf: (path) => ipcRenderer.invoke('mojive:read-urdf', path),
});
