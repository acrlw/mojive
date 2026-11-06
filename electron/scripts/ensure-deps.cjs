const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { spawnSync } = require('node:child_process');

function digest(...values) {
  const hash = createHash('sha256');
  for (const value of values) hash.update(value).update('\0');
  return hash.digest('hex');
}

function installedState(root, manifest) {
  const modules = path.join(root, 'node_modules');
  try {
    for (const name of Object.keys({ ...manifest.dependencies, ...manifest.devDependencies })) {
      if (!fs.existsSync(path.join(modules, name, 'package.json'))) return null;
    }
    const executable = fs.readFileSync(path.join(modules, 'electron/path.txt'), 'utf8').trim();
    if (!executable) return null;
    fs.accessSync(path.join(modules, 'electron/dist', executable), fs.constants.X_OK);
    fs.accessSync(path.join(modules, '.bin/tsc'), fs.constants.X_OK);
    fs.accessSync(path.join(modules, '.bin/vite'), fs.constants.X_OK);
    return digest(fs.readFileSync(path.join(modules, '.package-lock.json')));
  } catch (error) {
    if (['ENOENT', 'EACCES'].includes(error.code)) return null;
    throw error;
  }
}

function ensureDependencies(root, force = false) {
  const packageBytes = fs.readFileSync(path.join(root, 'package.json'));
  const manifest = JSON.parse(packageBytes);
  const expected = digest(
    packageBytes,
    fs.readFileSync(path.join(root, 'package-lock.json')),
    process.platform,
    process.arch,
    process.versions.modules,
  );
  const stamp = path.join(root, 'node_modules/.mojive-deps.json');
  let previous;
  try {
    previous = JSON.parse(fs.readFileSync(stamp, 'utf8'));
  } catch (error) {
    if (error.code !== 'ENOENT' && !(error instanceof SyntaxError)) throw error;
  }
  const installed = installedState(root, manifest);
  if (!force && installed && previous?.expected === expected && previous.installed === installed) {
    return;
  }

  console.log('Synchronizing Electron dependencies with package-lock.json...');
  // Record success only after npm and the Electron postinstall have both completed.
  fs.rmSync(stamp, { force: true });
  const result = spawnSync('npm', ['ci', '--no-audit', '--no-fund'], {
    cwd: root,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`Dependency installation failed (${result.signal || result.status}).`);
  }
  const ready = installedState(root, manifest);
  if (!ready)
    throw new Error('Dependency installation is incomplete; check npm lifecycle scripts.');
  fs.writeFileSync(stamp, JSON.stringify({ expected, installed: ready }) + '\n');
}

if (require.main === module) {
  try {
    ensureDependencies(path.resolve(__dirname, '..'), process.argv.includes('--force'));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

module.exports = { ensureDependencies };
