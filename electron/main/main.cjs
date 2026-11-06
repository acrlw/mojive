const { app, BrowserWindow, ipcMain, protocol, net, Menu } = require('electron');
const fs = require('node:fs/promises');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { Projects, within } = require('./resource-paths.cjs');
const { FileAccess, filePath } = require('./file-access.cjs');
const { createMenuController, openDocumentation } = require('./menu.cjs');
const projects = new Projects();
const developmentUrl = process.env.MOJIVE_DEV_URL;
if (developmentUrl && !/^http:\/\/127\.0\.0\.1:\d+\/?$/.test(developmentUrl))
  throw new Error('MOJIVE_DEV_URL must use the loopback interface.');
protocol.registerSchemesAsPrivileged(
  ['mojive-app', 'mojive-resource'].map((scheme) => ({
    scheme,
    privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true },
  })),
);
let mainWindow;
let menuController;
async function selectedUrdf(filename, register = false, model = false) {
  if (typeof filename !== 'string' || !(model ? /\.(xml|urdf)$/i.test(filename) : filename.toLowerCase().endsWith('.urdf')))
    throw new Error('Choose a .urdf resource.');
  if (register) {
    const directory = path.dirname(filename);
    const root = ['urdf', 'robots'].includes(path.basename(directory).toLowerCase())
      ? path.dirname(directory)
      : directory;
    await projects.register(root);
  }
  const { token, root, filename: resolved } = await projects.authorized(filename);
  const stat = await fs.stat(resolved);
  if (!stat.isFile() || stat.size > 100 * 1024 * 1024)
    throw new Error('URDF must be a file no larger than 100 MiB.');
  const relative = path
    .relative(root, path.dirname(resolved))
    .split(path.sep)
    .filter(Boolean)
    .map(encodeURIComponent)
    .join('/');
  return {
    name: path.basename(filename),
    path: resolved,
    content: await fs.readFile(resolved, 'utf8'),
    resourceBase: `mojive-resource://project/${token}/${relative ? `${relative}/` : ''}`,
    files: await projects.tree(root),
  };
}
function handle(channel, fn) {
  ipcMain.handle(channel, async (event, ...args) => {
    if (
      event.sender !== mainWindow?.webContents ||
      event.senderFrame !== mainWindow.webContents.mainFrame
    )
      throw new Error('Invalid IPC caller.');
    return fn(...args);
  });
}
app.whenReady().then(async () => {
  protocol.handle('mojive-app', async (request) => {
    try {
      const url = new URL(request.url);
      if (url.hostname !== 'app' || request.method !== 'GET')
        return new Response('Not found', { status: 404 });
      const root = await fs.realpath(path.join(__dirname, '../dist'));
      const filename = await fs.realpath(path.join(root, decodeURIComponent(url.pathname)));
      if (!within(root, filename)) return new Response('Forbidden', { status: 403 });
      return net.fetch(pathToFileURL(filename).href);
    } catch (error) {
      return new Response(error.message, { status: 404 });
    }
  });
  protocol.handle('mojive-resource', async (request) => {
    try {
      if (request.method !== 'GET') return new Response('Method not allowed', { status: 405 });
      const response = await net.fetch(pathToFileURL(await projects.resolve(request.url)).href);
      const headers = new Headers(response.headers);
      headers.set('Access-Control-Allow-Origin', '*');
      return new Response(response.body, { status: response.status, headers });
    } catch (error) {
      return new Response(error.message, {
        status: 404,
        headers: { 'Access-Control-Allow-Origin': '*' },
      });
    }
  });
  const fileAccess = new FileAccess([
    { label: 'Home', path: app.getPath('home') },
    { label: 'Desktop', path: app.getPath('desktop') },
    { label: 'Documents', path: app.getPath('documents') },
    { label: 'Downloads', path: app.getPath('downloads') },
  ]);
  handle('mojive:browse-files', (directory, includeHidden) =>
    fileAccess.browse(directory, includeHidden),
  );
  handle('mojive:open-urdf-path', async (filename) =>
    selectedUrdf(await fs.realpath(filePath(filename)), true),
  );
  handle('mojive:open-project-path', async (filename) => {
    const root = await fs.realpath(filePath(filename));
    if (!(await fs.stat(root)).isDirectory()) throw new Error('Choose a project folder.');
    await projects.register(root);
    return { path: root, name: path.basename(root), files: await projects.list(root) };
  });
  handle('mojive:read-project-directory', async (directory) => {
    const { filename } = await projects.authorized(filePath(directory));
    return projects.list(filename);
  });
  handle('mojive:search-project', async (directory, query) => {
    const { filename } = await projects.authorized(filePath(directory));
    return projects.search(filename, query);
  });
  handle('mojive:search-model-assets', async (directory, query) => {
    const { filename } = await projects.authorized(filePath(directory));
    return projects.search(filename, query, ['.xml', '.urdf']);
  });
  handle('mojive:open-scene-path', async (filename) => {
    const result = await fileAccess.readScene(filename);
    await projects.register(path.dirname(result.path));
    return result;
  });
  handle('mojive:save-scene-path', (filename, content, overwrite) =>
    fileAccess.saveScene(filename, content, overwrite),
  );
  handle('mojive:save-binary-file', (filename, content, overwrite) =>
    fileAccess.saveBinary(filename, content, overwrite),
  );
  handle('mojive:confirm-close', () => {
    mainWindow.destroy();
  });
  handle('mojive:open-model-path', async (filename) => selectedUrdf(await fs.realpath(filePath(filename)), true, true));
  handle('mojive:read-urdf', (filename) => selectedUrdf(filename));
  mainWindow = new BrowserWindow({
    width: 1540,
    height: 980,
    minWidth: 1040,
    minHeight: 700,
    show: false,
    title: 'Mojive',
    backgroundColor: '#1E2125',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    trafficLightPosition: { x: 14, y: 13 },
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: true,
    },
  });
  menuController = createMenuController(() => mainWindow, Boolean(developmentUrl));
  handle('mojive:read-clipboard-text', () => require('electron').clipboard.readText());
  handle('mojive:write-clipboard-text', (text) => {
    if (typeof text !== 'string' || text.length > 4 * 1024 * 1024)
      throw new Error('Clipboard text must be a string up to 4 MiB.');
    require('electron').clipboard.writeText(text);
  });
  handle('mojive:update-menu', (state) => menuController.update(state));
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    // The Help command is the only renderer request allowed to open a browser.
    void openDocumentation(url).catch((error) => console.error('Could not open documentation:', error));
    return { action: 'deny' };
  });
  mainWindow.webContents.on('will-navigate', (event) => event.preventDefault());
  // The renderer's beforeunload guard owns unsaved state. Its custom confirmation
  // may save, discard, or keep editing; destroy is requested only after resolution.
  mainWindow.webContents.on('will-prevent-unload', () => {
    mainWindow.webContents.send('mojive:menu-action', 'app.requestClose');
  });
  mainWindow.webContents.session.setPermissionRequestHandler((_contents, _permission, callback) =>
    callback(false),
  );
  mainWindow.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    const dev = developmentUrl ? ` ${developmentUrl} ws://127.0.0.1:*` : '';
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          `default-src 'self'; script-src 'self' 'wasm-unsafe-eval'${developmentUrl ? " 'unsafe-eval'" : ''}; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: mojive-resource:; font-src 'self'; connect-src 'self' mojive-resource:${dev}; object-src 'none'; base-uri 'self'; frame-src 'none'`,
        ],
      },
    });
  });
  mainWindow.once('ready-to-show', () => {
    if (!process.argv.includes('--hidden')) mainWindow.show();
  });
  if (developmentUrl) await mainWindow.loadURL(developmentUrl);
  else await mainWindow.loadURL('mojive-app://app/index.html');
});
app.on('window-all-closed', () => app.quit());
