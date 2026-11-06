import {
  access,
  cp,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rename,
  rm,
  writeFile,
} from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { packager } from '@electron/packager';

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const release = path.join(root, '../output/electron/release');
const metadata = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'));
const runtimeVersion = '39.8.10';
const preflight = process.argv.includes('--preflight');
const runtimeFiles = [
  'main.cjs',
  'preload.cjs',
  'resource-paths.cjs',
  'menu.cjs',
  'menu-catalog.json',
  'menu-en.json',
  'menu-zh.json',
  'file-access.cjs',
];
const menuCatalog = JSON.parse(await readFile(path.join(root, 'main/menu-catalog.json'), 'utf8'));
const menuIconNames = new Set(['copy', 'paste']);
function collectMenuIcons(entries) {
  for (const entry of entries) {
    if (entry.icon) menuIconNames.add(entry.icon);
    if (entry.activeIcon) menuIconNames.add(entry.activeIcon);
    if (entry.children) collectMenuIcons(entry.children);
  }
}
for (const group of menuCatalog) collectMenuIcons(group.items);
const requiredFiles = [
  'dist/index.html',
  'dist/menu-icons/manifest.json',
  ...[...menuIconNames].flatMap((name) => ['', '@2x'].map((scale) => `dist/menu-icons/${name}${scale}.png`)),
  'dist/demo/atelier_arm.urdf',
  'dist/demo/joint_gizmo.xml',
  'dist/demo/physics_pendulum.xml',
  'dist/licenses/MuJoCo-LICENSE.txt',
  'dist/licenses/MuJoCo-THIRD_PARTY_NOTICES.txt',
  'dist/fonts/MiSans-License.pdf',
  'dist/fonts/JetBrainsMono-OFL.txt',
  ...runtimeFiles.map((name) => `main/${name}`),
];
if (process.platform !== 'darwin')
  throw new Error('This package target produces a macOS application; run it on macOS.');
if (!['arm64', 'x64'].includes(process.arch))
  throw new Error(`Unsupported macOS architecture: ${process.arch}`);
if (require('electron/package.json').version !== runtimeVersion)
  throw new Error(`Packaging requires the pinned Electron ${runtimeVersion}. Run npm ci first.`);
for (const filename of requiredFiles) {
  try {
    await access(path.join(root, filename));
  } catch {
    throw new Error(`Missing ${filename}. Run npm run build before packaging.`);
  }
}

async function fileDigests(directory, prefix = '') {
  const result = {};
  for (const item of (await readdir(directory, { withFileTypes: true })).sort((a, b) =>
    a.name.localeCompare(b.name),
  )) {
    const relative = prefix ? `${prefix}/${item.name}` : item.name;
    if (item.isDirectory())
      Object.assign(result, await fileDigests(path.join(directory, item.name), relative));
    else if (item.isFile())
      result[relative] = createHash('sha256')
        .update(await readFile(path.join(directory, item.name)))
        .digest('hex');
    else throw new Error(`Unexpected non-file build artifact: ${relative}`);
  }
  return result;
}
const sourceDigests = await fileDigests(path.join(root, 'dist'), 'dist');
if (!Object.keys(sourceDigests).some((filename) => /^dist\/assets\/mujoco-.*\.wasm$/.test(filename)))
  throw new Error('The production build is missing the MuJoCo WebAssembly runtime.');
if (process.argv.includes('--check')) {
  console.log(
    `Package input verified: ${Object.keys(sourceDigests).length} built files; Electron ${runtimeVersion}; ${process.arch}.`,
  );
  process.exit(0);
}

const temporary = await mkdtemp(path.join(root, '.package-'));
try {
  const source = path.join(temporary, 'source');
  await mkdir(source);
  await cp(path.join(root, 'dist'), path.join(source, 'dist'), { recursive: true });
  await mkdir(path.join(source, 'main'));
  for (const filename of runtimeFiles)
    await cp(path.join(root, 'main', filename), path.join(source, 'main', filename));
  await writeFile(
    path.join(source, 'package.json'),
    JSON.stringify(
      {
        name: metadata.name,
        preflight,
        productName: 'Mojive',
        version: metadata.version,
        description: 'Mojive interactive 3D and simulation viewer',
        type: 'module',
        main: 'main/main.cjs',
        license: 'MIT',
      },
      null,
      2,
    ) + '\n',
  );
  await cp(path.resolve(root, '../LICENSE'), path.join(source, 'LICENSE'));
  const licenses = path.join(source, 'licenses');
  await mkdir(licenses);
  for (const name of [
    'react',
    'react-dom',
    'three',
    'i18next',
    'react-i18next',
    'use-sync-external-store',
    'html-parse-stringify',
    '@babel/runtime',
  ]) {
    const location = path.join(root, 'node_modules', name);
    const files = await readdir(location);
    const filename = files.find((value) => /^LICENSE(?:\.txt|\.md)?$/i.test(value));
    if (!filename) throw new Error(`Missing dependency license: ${name}`);
    await cp(
      path.join(location, filename),
      path.join(licenses, `${name.replaceAll('/', '-')}-LICENSE.txt`),
    );
  }
  const electronRoot = path.dirname(require.resolve('electron/package.json'));
  for (const filename of ['LICENSE', 'LICENSES.chromium.html'])
    await cp(
      path.join(electronRoot, 'dist', filename),
      path.join(licenses, `Electron-${filename}`),
    );
  await writeFile(
    path.join(licenses, 'README.txt'),
    'Runtime notices for Mojive. MiSans and JetBrains Mono licenses and font provenance are included in dist/fonts/. MuJoCo and its third-party notices are included in dist/licenses/. React, React DOM, Three.js, i18next, react-i18next and their runtime helpers are bundled into dist/assets/. Electron and Chromium notices are preserved here and with the runtime.\n',
  );

  const { downloadArtifact } = require('@electron/get');
  const archive = await downloadArtifact({
    version: runtimeVersion,
    artifactName: 'electron',
    platform: 'darwin',
    arch: process.arch,
  });
  const filename = `electron-v${runtimeVersion}-darwin-${process.arch}.zip`;
  const expected = require(path.join(electronRoot, 'checksums.json'))[filename];
  const actual = createHash('sha256')
    .update(await readFile(archive))
    .digest('hex');
  if (!expected || actual !== expected)
    throw new Error('Official Electron archive checksum failed.');
  const packages = await packager({
    dir: source,
    name: 'Mojive',
    executableName: 'Mojive',
    appBundleId: 'org.mojive.editor',
    appCategoryType: 'public.app-category.developer-tools',
    appVersion: metadata.version,
    buildVersion: metadata.version,
    platform: 'darwin',
    arch: process.arch,
    electronVersion: runtimeVersion,
    electronZipDir: path.dirname(archive),
    out: path.join(temporary, 'built'),
    asar: false,
    prune: false,
    overwrite: true,
    darwinDarkModeSupport: true,
    osxSign: false,
    extendInfo: { CFBundleDisplayName: 'Mojive' },
  });
  if (packages.length !== 1)
    throw new Error(`Expected one application bundle, received ${packages.length}.`);
  const app = path.join(packages[0], 'Mojive.app');
  await access(path.join(app, 'Contents/MacOS/Mojive'));
  const packagedSource = path.join(app, 'Contents/Resources/app');
  const entries = (await readdir(packagedSource)).sort();
  const expectedEntries = ['LICENSE', 'dist', 'main', 'licenses', 'package.json'].sort();
  if (JSON.stringify(entries) !== JSON.stringify(expectedEntries))
    throw new Error(`Unexpected packaged files: ${entries.join(', ')}`);
  const packagedDigests = await fileDigests(path.join(packagedSource, 'dist'), 'dist');
  if (JSON.stringify(sourceDigests) !== JSON.stringify(packagedDigests))
    throw new Error('Packaged renderer differs from the reviewed build.');
  await mkdir(release, { recursive: true });
  await rm(path.join(release, 'Mojive.app'), { recursive: true, force: true });
  await rename(app, path.join(release, 'Mojive.app'));
  await writeFile(
    path.join(release, 'package-manifest.json'),
    JSON.stringify(
      {
        preflight,
        productName: 'Mojive',
        version: metadata.version,
        electronVersion: runtimeVersion,
        platform: 'darwin',
        arch: process.arch,
        signed: false,
        notarized: false,
        electronArchiveSha256: actual,
        rendererSha256: sourceDigests,
      },
      null,
      2,
    ) + '\n',
  );
  if (preflight)
    await writeFile(
      path.join(release, 'PREFLIGHT.txt'),
      'Preflight bundle; rebuild and package the reviewed final source before delivery.\n',
    );
  else await rm(path.join(release, 'PREFLIGHT.txt'), { force: true });
  console.log(
    `Packaged ${path.join(release, 'Mojive.app')} (${process.arch}, Electron ${runtimeVersion}).`,
  );
} finally {
  await rm(temporary, { recursive: true, force: true });
}
