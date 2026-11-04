import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';
const source = await readFile(new URL('../src/viewport/viewcube.ts', import.meta.url), 'utf8');
const js = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText;
const {
  g3LineOutline,
  arcRibbonMesh,
  openStrokeOutline,
  lollipopPoints,
  backAlpha,
  viewCubeLayout,
  hitTest,
  yawPitchForAxis,
} = await import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'));
test('arc strips preserve native indices and near-full-turn holes before folded-offset repair', async () => {
  const reference = JSON.parse(
    await readFile(new URL('./fixtures/arc-ribbon-native.json', import.meta.url), 'utf8'),
  );
  for (const sample of reference.cases) {
    const actual = arcRibbonMesh(sample.points, sample.width);
    assert.equal(
      actual.vertices.length,
      sample.vertices.length,
      `height=${sample.height}, span=${sample.span}`,
    );
    for (let i = 0; i < actual.vertices.length; i++)
      for (let j = 0; j < 2; j++)
        assert.ok(Math.abs(actual.vertices[i][j] - sample.vertices[i][j]) < 1e-8);
    if (actual.indices !== null) assert.deepEqual(actual.indices, sample.indices);
  }
});
const reference = JSON.parse(
  await readFile(new URL('./viewcube-native-reference.json', import.meta.url), 'utf8'),
);
test('lollipop geometry matches independently exported Python production contours including disappearing neck', () => {
  for (const sample of reference.curves) {
    const actual = lollipopPoints(sample.distance, sample.radius, sample.width);
    assert.equal(
      actual.length,
      sample.points.length,
      `vertex count at distance ${sample.distance}`,
    );
    for (let i = 0; i < actual.length; i++)
      for (let j = 0; j < 2; j++)
        assert.ok(
          Math.abs(actual[i][j] - sample.points[i][j]) < 1e-9,
          `point ${i}/${j} at distance ${sample.distance}: ${actual[i][j]} != ${sample.points[i][j]}`,
        );
  }
});
test('back faces use native smooth fade and aligned axis owns the origin', () => {
  for (const [depth, alpha] of reference.fade)
    assert.ok(Math.abs(backAlpha(depth) - alpha) < 1e-12);
  const balls = viewCubeLayout([0, 1, 0], [0, 0, 1], [-1, 0, 0]);
  assert.equal(balls[0].depth, 1);
  assert.equal(balls.at(-1).depth, -1);
  const nearest = hitTest(balls, [45, 45]);
  assert.equal(nearest.axis, 0);
  assert.equal(nearest.sign, 1);
  assert.equal(nearest.alpha, 1);
  const farthest = balls.find((b) => b.axis === 0 && b.sign === -1);
  assert.equal(farthest.alpha, 0);
});
test('orientation remains orthographic, symmetric and maps native top pitch limit', () => {
  const balls = viewCubeLayout([0.8, 0.6, 0], [-0.36, 0.48, 0.8], [0.48, -0.64, 0.6]);
  for (let axis = 0; axis < 3; axis++) {
    const a = balls.find((b) => b.axis === axis && b.sign === 1),
      b = balls.find((b) => b.axis === axis && b.sign === -1);
    assert.ok(Math.abs(a.x + b.x - 90) < 1e-12);
    assert.ok(Math.abs(a.y + b.y - 90) < 1e-12);
    assert.equal(a.radius, b.radius);
  }
  assert.deepEqual(yawPitchForAxis(2, 1), [-90, 89.9]);
  assert.deepEqual(yawPitchForAxis(2, -1), [-90, -89.9]);
});

test('open ring and tick strokes match native circular caps and folded extreme-angle exterior', () => {
  for (const sample of reference.strokes) {
    const actual = openStrokeOutline(sample.points, sample.width);
    assert.equal(
      actual.length,
      sample.outline.length,
      `stroke vertex count at ${JSON.stringify(sample.points[1])}`,
    );
    for (let i = 0; i < actual.length; i++)
      for (let j = 0; j < 2; j++)
        assert.ok(
          Math.abs(actual[i][j] - sample.outline[i][j]) < 1e-8,
          `stroke point ${i}/${j}: ${actual[i][j]} != ${sample.outline[i][j]}`,
        );
  }
});

test('joint ticks and value rails retain native fitted G3 caps', () => {
  for (const sample of reference.g3lines) {
    const actual = g3LineOutline(
      sample.start,
      sample.end,
      sample.width,
      sample.roundStart,
      sample.roundEnd,
    );
    assert.equal(actual.length, sample.outline.length);
    for (let i = 0; i < actual.length; i++)
      for (let j = 0; j < 2; j++)
        assert.ok(
          Math.abs(actual[i][j] - sample.outline[i][j]) < 1e-8,
          `G3 line ${i}/${j}: ${actual[i][j]} != ${sample.outline[i][j]}`,
        );
  }
});

test('precision rail segments preserve independent native circular end caps', () => {
  for (const sample of reference.railSegments) {
    const actual = openStrokeOutline(
      sample.points,
      sample.width,
      sample.roundStart,
      sample.roundEnd,
    );
    assert.equal(actual.length, sample.outline.length);
    for (let i = 0; i < actual.length; i++)
      for (let j = 0; j < 2; j++) assert.ok(Math.abs(actual[i][j] - sample.outline[i][j]) < 1e-8);
  }
});
