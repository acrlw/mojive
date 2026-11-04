// Node 26 can exit before extract-zip finishes its stream. Verify the official
// artifact and use the platform extractor when Electron's installer is incomplete.
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const electronRoot = path.dirname(require.resolve('electron/package.json'));
const platformPath =
  process.platform === 'darwin'
    ? 'Electron.app/Contents/MacOS/Electron'
    : process.platform === 'win32'
      ? 'electron.exe'
      : 'electron';
const executable = path.join(electronRoot, 'dist', platformPath);
async function install() {
  if (fs.existsSync(executable)) {
    fs.writeFileSync(path.join(electronRoot, 'path.txt'), platformPath);
    return;
  }
  execFileSync(process.execPath, [path.join(electronRoot, 'install.js')], { stdio: 'inherit' });
  if (fs.existsSync(executable)) return;
  const { downloadArtifact } = require('@electron/get');
  const { version } = require(path.join(electronRoot, 'package.json'));
  const filename = `electron-v${version}-${process.platform}-${process.arch}.zip`;
  const archive = await downloadArtifact({
    version,
    artifactName: 'electron',
    platform: process.platform,
    arch: process.arch,
  });
  const expected = require(path.join(electronRoot, 'checksums.json'))[filename];
  const actual = createHash('sha256').update(fs.readFileSync(archive)).digest('hex');
  if (!expected || actual !== expected) throw Error('Electron archive checksum failed');
  if (process.platform === 'win32')
    throw Error('Electron installation did not finish. Install with Node 22 or Node 24.');
  execFileSync('unzip', ['-oq', archive, '-d', path.join(electronRoot, 'dist')], {
    stdio: 'inherit',
  });
  if (!fs.existsSync(executable)) throw Error('Electron executable missing after extraction');
  fs.writeFileSync(path.join(electronRoot, 'path.txt'), platformPath);
}
install().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
