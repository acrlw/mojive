/** Import Mojive's adopted design; this output does not become a second specification. */
import { readFile, writeFile, mkdir, copyFile, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const app = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repo = path.resolve(app, '..');
const spec = JSON.parse(
  await readFile(path.join(repo, 'python/design/resources/design-spec.json'), 'utf8'),
);
const svgRoot = path.join(repo, 'examples/ui_design/native/svg');
const manifest = JSON.parse(await readFile(path.join(svgRoot, 'sources.json'), 'utf8'));
const localRoot = path.join(app, 'design/icons');
const localManifest = JSON.parse(await readFile(path.join(localRoot, 'sources.json'), 'utf8'));
await mkdir(path.join(app, 'public/icons'), { recursive: true });
await mkdir(path.join(app, 'public/fonts'), { recursive: true });
const icons = {};
const iconPaths = {};
for (const [name, file] of Object.entries(manifest.icons)) {
  let svg = await readFile(path.join(svgRoot, file), 'utf8');
  // The native importer applies the family default to upstream 2-unit strokes.
  if (!file.startsWith('mojive/')) svg = svg.replace(/stroke-width="2"/g, 'stroke-width="1.5"');
  icons[name] = svg;
  iconPaths[name] = `./icons/${name}.svg`;
  await writeFile(path.join(app, `public/icons/${name}.svg`), svg);
}
for (const [name, file] of Object.entries(localManifest.icons)) {
  if (icons[name]) throw new Error(`Local icon must not replace an adopted native master: ${name}`);
  const svg = await readFile(path.join(localRoot, file), 'utf8');
  icons[name] = svg;
  iconPaths[name] = `./icons/${name}.svg`;
  await writeFile(path.join(app, `public/icons/${name}.svg`), svg);
}
for (const [alias, original] of Object.entries({
  'panel-right': 'inspect',
  trash: 'delete',
  palette: 'visual',
  lightbulb: 'light',
  eyeoff: 'eyeoff',
  'eye-off': 'eyeoff',
  'mouse-pointer-2': 'select',
  'chevron-right': 'right',
  'chevron-down': 'down',
  'sliders-horizontal': 'control',
  'rotate-cw': 'reset',
  'square-terminal': 'terminal',
})) {
  icons[alias] = icons[original];
  iconPaths[alias] = iconPaths[original];
}
await writeFile(
  path.join(app, 'src/icons.ts'),
  '// Generated from native SVG sources.json and design/icons/sources.json; run node scripts/sync-design.mjs.\nexport const icons: Record<string,string> = ' +
    JSON.stringify(icons) +
    ';\nexport const iconPaths: Record<string,string> = ' +
    JSON.stringify(iconPaths) +
    ';\n',
);
await writeFile(
  path.join(app, 'public/icons/sources.json'),
  JSON.stringify(
    {
      ...manifest,
      electron: { source: 'design/icons/sources.json', ...localManifest },
    },
    null,
    2,
  ) + '\n',
);
for (const family of ['lucide', 'tabler'])
  await copyFile(
    path.join(svgRoot, family, 'LICENSE'),
    path.join(app, `public/icons/${family}-LICENSE`),
  );
for (const file of ['MiSans-Regular.ttf', 'MiSans-Semibold.ttf', 'MiSans-License.pdf', 'README.md'])
  await copyFile(
    path.join(repo, 'python/text/resources/fonts', file),
    path.join(app, 'public/fonts', file),
  );
const jetbrains = 'https://raw.githubusercontent.com/JetBrains/JetBrainsMono/v2.304/';
for (const [file, remote] of [
  ['JetBrainsMono-Regular.woff2', 'fonts/webfonts/JetBrainsMono-Regular.woff2'],
  ['JetBrainsMono-OFL.txt', 'OFL.txt'],
]) {
  const dest = path.join(app, 'public/fonts', file);
  try {
    await access(dest);
  } catch {
    const response = await fetch(jetbrains + remote);
    if (!response.ok) throw new Error(`Font download failed: ${response.status}`);
    await writeFile(dest, Buffer.from(await response.arrayBuffer()));
  }
}
const vars = Object.entries({ ...spec.tokens, ...spec.layout })
  .filter(([, v]) => typeof v === 'number' || String(v).startsWith('#'))
  .map(([k, v]) => `  --${k.replaceAll('_', '-')}: ${v}${typeof v === 'number' ? 'px' : ''};`)
  .join('\n');
await writeFile(
  path.join(app, 'src/design-tokens.css'),
  `/* Generated from python/design/resources/design-spec.json. */\n@font-face{font-family:MiSans;src:url('/fonts/MiSans-Regular.ttf') format('truetype');font-weight:400;font-display:swap}\n@font-face{font-family:MiSans;src:url('/fonts/MiSans-Semibold.ttf') format('truetype');font-weight:600;font-display:swap}\n@font-face{font-family:'JetBrains Mono';src:url('/fonts/JetBrainsMono-Regular.woff2') format('woff2');font-weight:400;font-display:swap}\n:root{\n${vars}\n  --font-ui:'MiSans',sans-serif;\n  --font-mono:'JetBrains Mono',monospace;\n}\n`,
);
await writeFile(
  path.join(app, 'public/fonts/PROVENANCE.md'),
  `MiSans Regular and Semibold are copied without changes from Mojive's bundled official Xiaomi distribution. Preserve README.md and MiSans-License.pdf.\n\nJetBrains Mono Regular is the unmodified v2.304 webfont from ${jetbrains}fonts/webfonts/JetBrainsMono-Regular.woff2, licensed under SIL OFL 1.1 (JetBrainsMono-OFL.txt).\n`,
);
console.log(
  `Imported ${Object.keys(manifest.icons).length} native and ${Object.keys(localManifest.icons).length} Electron semantic icons, adopted tokens, MiSans and JetBrains Mono.`,
);
