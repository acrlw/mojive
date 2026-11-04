/** Build native template images from the same adopted SVG canvases as the renderer. */
const { app, BrowserWindow } = require('electron');
const fs = require('node:fs/promises');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { MENU_ICONS } = require('../main/menu.cjs');
const root = path.resolve(__dirname, '..');
app.dock?.hide();
app
  .whenReady()
  .then(async () => {
    const directory = path.join(root, 'public/menu-icons');
    const sources = await Promise.all(
      [...new Set(Object.values(MENU_ICONS))].sort().map(async (name) => ({
        name,
        svg: (await fs.readFile(path.join(root, 'public/icons', `${name}.svg`), 'utf8')).replace(
          /<svg\b[^>]*>/,
          (tag) =>
            tag
              .replace(/\s(?:width|height|xmlns)="[^"]*"/g, '')
              .replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"'),
        ),
      })),
    );
    const digest = createHash('sha256')
      .update('native-template-16pt-v1')
      .update(JSON.stringify(sources))
      .digest('hex');
    const manifestPath = path.join(directory, 'manifest.json');
    try {
      const previous = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
      if (previous.sha256 === digest) {
        await Promise.all(
          sources.flatMap(({ name }) =>
            ['', '@2x'].map((suffix) => fs.access(path.join(directory, `${name}${suffix}.png`))),
          ),
        );
        console.log(`Native menu icons current (${sources.length} semantic masters, 1×/2×).`);
        app.quit();
        return;
      }
    } catch {
      /* Missing generated files are rebuilt from their masters. */
    }
    const window = new BrowserWindow({
      show: false,
      webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false },
    });
    await window.loadURL('data:text/html,<!doctype html><title>Compile menu icons</title>');
    const images = await window.webContents.executeJavaScript(`(async () => {
    const output = [];
    for (const {name, svg} of ${JSON.stringify(sources)}) {
      for (const scale of [1, 2]) {
        const size = 16 * scale;
        const source = svg.replace('<svg ', '<svg width="' + size + '" height="' + size + '" ').replaceAll('currentColor', '#000000');
        const image = new Image(); image.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(source))); try { await image.decode(); } catch (error) { throw new Error(name + ': ' + error.message); }
        const canvas = document.createElement('canvas'); canvas.width = canvas.height = size;
        canvas.getContext('2d').drawImage(image, 0, 0, size, size);
        output.push({name, scale, png:canvas.toDataURL('image/png').split(',')[1]});
      }
    }
    return output;
  })()`);
    await fs.mkdir(directory, { recursive: true });
    for (const { name, scale, png } of images)
      await fs.writeFile(
        path.join(directory, `${name}${scale === 2 ? '@2x' : ''}.png`),
        Buffer.from(png, 'base64'),
      );
    await fs.writeFile(
      manifestPath,
      JSON.stringify(
        {
          sha256: digest,
          size: 16,
          scales: [1, 2],
          masters: sources.map(({ name }) => name),
          source: '../icons/sources.json',
        },
        null,
        2,
      ) + '\n',
    );
    window.destroy();
    console.log(`Compiled ${sources.length} native template icons at 1×/2×.`);
    app.quit();
  })
  .catch((error) => {
    console.error(error);
    app.exit(1);
  });
