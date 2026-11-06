import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import loadMujoco from '@mujoco/mujoco';
import { createModelBundle } from '../src/resources/modelBundle.ts';
import { MujocoBackend } from '../src/physics/backends/mujoco/MujocoBackend.ts';
import { SceneBinding } from '../src/physics/SceneBinding.ts';
import { createTexture, applySurface } from '../src/physics/appearance.ts';

const mj = await loadMujoco();

test('compiled MuJoCo texture roles and color spaces reach the renderer', async (t) => {
  const xml = `<mujoco><asset>
    <texture name="color" type="2d" builtin="flat" width="4" height="4" rgb1=".2 .4 .6" colorspace="sRGB"/>
    <texture name="normal" type="2d" builtin="flat" width="4" height="4" rgb1=".5 .5 1" colorspace="linear"/>
    <texture name="orm" type="2d" builtin="flat" width="4" height="4" rgb1=".25 .5 .75" colorspace="linear"/>
    <texture name="opacity" type="2d" builtin="flat" width="4" height="4" rgb1=".5 .5 .5" colorspace="linear"/>
    <material name="layered" texrepeat="2 3" metallic=".8" roughness=".7">
      <layer role="rgb" texture="color"/><layer role="normal" texture="normal"/>
      <layer role="orm" texture="orm"/><layer role="opacity" texture="opacity"/>
    </material>
    <material name="data_only" texrepeat="3 4"><layer role="orm" texture="orm"/></material>
  </asset><worldbody><geom name="layered" type="plane" size="1 1 .1" material="layered"/>
    <geom name="data_only" type="box" pos="0 0 1" size=".1 .1 .1" material="data_only"/>
  </worldbody></mujoco>`;
  const bundle = await createModelBundle('Texture roles', 'model.xml', [{ path: 'model.xml', bytes: new TextEncoder().encode(xml) }]);
  const engine = await MujocoBackend.load(mj, bundle);
  t.after(() => engine.dispose());
  const binding = new SceneBinding(engine.source, await engine.frame());
  const mesh = binding.document.nodes.find((node) => node.name === 'layered').object;
  assert.equal(mesh.material.map.colorSpace, THREE.SRGBColorSpace);
  assert.equal(engine.source.textures.find((texture) => texture.name === 'normal').colorSpace, 'linear');
  for (const key of ['normalMap', 'aoMap', 'roughnessMap', 'metalnessMap', 'alphaMap']) {
    assert.ok(mesh.material[key]?.isDataTexture, key);
    assert.equal(mesh.material[key].colorSpace, THREE.NoColorSpace, key);
    assert.deepEqual(mesh.material[key].repeat.toArray(), [1, 1], key);
  }
  assert.deepEqual(Array.from(mesh.material.aoMap.image.data.slice(0, 3)), [63, 127, 191]);
  assert.equal(mesh.material.transparent, true);
  assert.ok(Math.abs(mesh.material.metalness - .8) < 1e-6);
  assert.ok(Math.abs(mesh.material.roughness - .7) < 1e-6);
  const uv = mesh.geometry.getAttribute('uv');
  assert.ok(Math.max(...Array.from(uv.array)) > 1, 'generated primitive UVs retain texrepeat');
  const dataOnly = binding.document.nodes.find((node) => node.name === 'data_only').object;
  assert.equal(dataOnly.material.map, null);
  assert.ok(dataOnly.material.roughnessMap && dataOnly.geometry.hasAttribute('uv'));
  assert.equal(dataOnly.material.roughness, 1, 'unset factors preserve the texture values');
  assert.equal(dataOnly.material.metalness, 1);
});

test('rigid, flex and skin surfaces normalize MuJoCo PBR sentinels without losing explicit zero', async () => {
  const cases = [
    { name: 'legacy', attributes: 'shininess=".8"', roughness: .2, metallic: 0 },
    { name: 'default', attributes: '', roughness: .5, metallic: 0 },
    { name: 'explicit', attributes: 'roughness=".3" metallic=".7"', roughness: .3, metallic: .7 },
    { name: 'zero', attributes: 'roughness="0" metallic="0"', layer: '<layer role="orm" texture="data"/>', roughness: 0, metallic: 0 },
    { name: 'texture', attributes: '', layer: '<layer role="orm" texture="data"/>', roughness: 1, metallic: 1 },
    { name: 'separate_textures', attributes: '', layer: '<layer role="roughness" texture="data"/><layer role="metallic" texture="data"/>', roughness: 1, metallic: 1 },
    { name: 'bounded', attributes: 'roughness="2" metallic="2"', roughness: 1, metallic: 1 },
  ];
  for (const entry of cases) {
    const xml = `<mujoco><asset>
      <texture name="data" type="2d" builtin="flat" width="2" height="2" rgb1=".5 .5 .5" colorspace="linear"/>
      <material name="surface" ${entry.attributes}>${entry.layer ?? ''}</material>
    </asset><worldbody>
      <body name="bone"><geom name="rigid" size=".1" material="surface"/></body>
      <geom name="unassigned" pos="2 0 0" size=".1"/>
      <flexcomp name="flex" type="grid" count="2 2 1" spacing=".2 .2 .2" dim="2" mass="1" material="surface"><edge damping="1"/></flexcomp>
    </worldbody><deformable>
      <skin name="skin" material="surface" vertex="0 0 0 1 0 0 0 1 0" face="0 1 2">
        <bone body="bone" bindpos="0 0 0" bindquat="1 0 0 0" vertid="0 1 2" vertweight="1 1 1"/>
      </skin>
    </deformable></mujoco>`;
    const bundle = await createModelBundle(entry.name, 'model.xml', [{ path: 'model.xml', bytes: new TextEncoder().encode(xml) }]);
    const engine = await MujocoBackend.load(mj, bundle);
    try {
      const binding = new SceneBinding(engine.source, await engine.frame());
      for (const name of ['rigid', 'flex', 'skin']) {
        const geometry = engine.source.geometries.find((geometry) => geometry.name === name);
        const material = binding.document.nodes.find((node) => node.name === name).object.material;
        for (const [actual, expected] of [[geometry.roughness, entry.roughness], [geometry.metallic, entry.metallic],
          [material.roughness, entry.roughness], [material.metalness, entry.metallic]])
          assert.ok(Math.abs(actual - expected) < 1e-6, `${entry.name}/${name}: ${actual} != ${expected}`);
      }
      const unassigned = engine.source.geometries.find((geometry) => geometry.name === 'unassigned');
      assert.equal(unassigned.roughness, .5);
      assert.equal(unassigned.metallic, 0);
    } finally { engine.dispose(); }
  }
});

test('non-finite compiled material factors fail loading before reaching shaders', async () => {
  for (const field of ['roughness', 'metallic', 'shininess']) for (const value of ['nan', 'inf']) {
    const xml = `<mujoco><asset><material name="invalid" ${field}="${value}"/></asset>
      <worldbody><geom size=".1" material="invalid"/></worldbody></mujoco>`;
    const bundle = await createModelBundle('Invalid surface', 'model.xml', [{ path: 'model.xml', bytes: new TextEncoder().encode(xml) }]);
    await assert.rejects(MujocoBackend.load(mj, bundle), new RegExp(`Material "invalid" has non-finite ${field}`));
  }
});

test('RGBA alpha enables blending without changing the shared texture or linear color values', () => {
  const texture = createTexture({ name: 'linear RGBA', type: '2d', width: 1, height: 1,
    channels: 4, colorSpace: 'linear', pixels: new Uint8Array([64, 128, 192, 96]) });
  const surface = new THREE.MeshStandardMaterial();
  applySurface(surface, new THREE.PlaneGeometry(2, 2), {
    shape: { type: 'plane', size: [1, 1, .1] },
    material: { name: 'RGBA', texture: 0, textureAlpha: true, repeat: [1, 1], uniform: false, emission: 0 },
  }, [texture]);
  assert.equal(surface.transparent, true);
  assert.equal(surface.map.colorSpace, THREE.LinearSRGBColorSpace);
  assert.deepEqual(Array.from(surface.map.image.data), [64, 128, 192, 96]);
  assert.notEqual(surface.map, texture);
  assert.deepEqual(texture.repeat.toArray(), [1, 1]);
});
