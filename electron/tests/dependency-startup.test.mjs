import assert from 'node:assert/strict';
import { test } from 'node:test';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const project = fileURLToPath(new URL('../', import.meta.url));

function fixture(t) {
  const output = path.resolve(project, '../output/electron/verification');
  mkdirSync(output, { recursive: true });
  const root = mkdtempSync(path.join(output, 'dependency-startup-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(path.join(root, 'scripts'));
  mkdirSync(path.join(root, 'bin'));
  copyFileSync(path.join(project, 'Makefile'), path.join(root, 'Makefile'));
  copyFileSync(
    path.join(project, 'scripts/ensure-deps.cjs'),
    path.join(root, 'scripts/ensure-deps.cjs'),
  );
  writeFileSync(
    path.join(root, 'package.json'),
    JSON.stringify({ dependencies: { electron: '1.0.0', typescript: '1.0.0', vite: '1.0.0' } }),
  );
  writeFileSync(path.join(root, 'package-lock.json'), '{"lockfileVersion":3}');
  writeFileSync(
    path.join(root, 'bin/npm.cjs'),
    `#!/usr/bin/env node
const fs = require('node:fs');
const path = require('node:path');
fs.appendFileSync('installs.jsonl', JSON.stringify(process.argv.slice(2)) + '\\n');
if (process.env.FAIL_INSTALL === '1') {
  console.error('Simulated npm failure');
  process.exit(17);
}
fs.rmSync('node_modules', { recursive: true, force: true });
const manifest = JSON.parse(fs.readFileSync('package.json'));
for (const name of Object.keys({ ...manifest.dependencies, ...manifest.devDependencies })) {
  fs.mkdirSync(path.join('node_modules', name), { recursive: true });
  fs.writeFileSync(path.join('node_modules', name, 'package.json'), '{}');
}
fs.mkdirSync('node_modules/electron/dist', { recursive: true });
fs.writeFileSync('node_modules/electron/path.txt', 'electron');
fs.writeFileSync('node_modules/electron/dist/electron', '', { mode: 0o755 });
fs.mkdirSync('node_modules/.bin', { recursive: true });
for (const tool of ['tsc', 'vite']) fs.writeFileSync('node_modules/.bin/' + tool, '', { mode: 0o755 });
fs.copyFileSync('package-lock.json', 'node_modules/.package-lock.json');
`,
    { mode: 0o755 },
  );
  if (process.platform === 'win32') {
    writeFileSync(path.join(root, 'bin/npm.cmd'), `@"${process.execPath}" "%~dp0npm.cjs" %*\r\n`);
  } else {
    copyFileSync(path.join(root, 'bin/npm.cjs'), path.join(root, 'bin/npm'));
  }
  const run = (target = 'ensure-deps', env = {}) =>
    spawnSync('make', [target], {
      cwd: root,
      encoding: 'utf8',
      env: {
        ...process.env,
        PATH: path.join(root, 'bin') + path.delimiter + process.env.PATH,
        ...env,
      },
    });
  const calls = () => readFileSync(path.join(root, 'installs.jsonl'), 'utf8').trim().split('\n');
  const ok = (result) => assert.equal(result.status, 0, result.stdout + result.stderr);
  return { root, run, calls, ok };
}

test('Make installs missing dependencies once and reuses a verified installation', (t) => {
  const { root, run, calls, ok } = fixture(t);
  ok(run());
  assert.deepEqual(JSON.parse(calls()[0]), ['ci', '--no-audit', '--no-fund']);
  ok(run());
  assert.equal(calls().length, 1);
  const manifest = JSON.parse(readFileSync(path.join(root, 'package.json'), 'utf8'));
  manifest.dependencies.i18next = '26.7.1';
  writeFileSync(path.join(root, 'package.json'), JSON.stringify(manifest));
  ok(run());
  assert.equal(calls().length, 2);
  assert.equal(readFileSync(path.join(root, 'node_modules/i18next/package.json'), 'utf8'), '{}');
});

test('Make refreshes a changed lockfile even when compiler and runtime already exist', (t) => {
  const { root, run, calls, ok } = fixture(t);
  ok(run());
  writeFileSync(path.join(root, 'package-lock.json'), '{"lockfileVersion":3,"version":"2.0.0"}');
  ok(run());
  assert.equal(calls().length, 2);
  ok(run());
  assert.equal(calls().length, 2);
});

test('Make repairs removed packages, missing runtime and changed installed dependency trees', (t) => {
  const { root, run, calls, ok } = fixture(t);
  ok(run());
  for (const name of ['typescript/package.json', 'electron/dist/electron', '.bin/vite']) {
    rmSync(path.join(root, 'node_modules', name));
    ok(run());
  }
  writeFileSync(path.join(root, 'node_modules/.package-lock.json'), '{}');
  ok(run());
  assert.equal(calls().length, 5);
});

test('Failed install stops Make and cannot mark stale dependencies as ready', (t) => {
  const { root, run, calls, ok } = fixture(t);
  ok(run());
  writeFileSync(path.join(root, 'package-lock.json'), '{"lockfileVersion":3,"version":"2.0.0"}');
  const failed = run('ensure-deps', { FAIL_INSTALL: '1' });
  assert.notEqual(failed.status, 0);
  assert.match(failed.stderr, /Simulated npm failure/);
  assert.throws(() => readFileSync(path.join(root, 'node_modules/.mojive-deps.json')), {
    code: 'ENOENT',
  });
  ok(run());
  assert.equal(calls().length, 3);
});

test('Explicit install forces a clean refresh and corrupt state is recoverable', (t) => {
  const { root, run, calls, ok } = fixture(t);
  ok(run());
  ok(run('install'));
  writeFileSync(path.join(root, 'node_modules/.mojive-deps.json'), '{');
  ok(run());
  assert.equal(calls().length, 3);
});
