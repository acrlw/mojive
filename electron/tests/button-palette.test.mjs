import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buttonPalette } from '../src/button-palette.ts';
const themes = JSON.parse(readFileSync(new URL('../src/locales/themes.json', import.meta.url)));
// Goldens evaluated from the native primary_action_ink / secondary_action_ink functions.
const native = {
  dark: ['#9CBF8D', '#67875A', '#B8D2AC', '#1A1D20', '#2B2F34', '#363B41', '#40474E'],
  light: ['#4C7040', '#45683A', '#3E6135', '#FFFFFF', '#CAD0C7', '#C0C8BA', '#B5C0AF'],
};
for (const theme of ['dark', 'light'])
  test(`${theme} semantic action palette matches native rest, hover and pressed colors`, () => {
    assert.deepEqual(Object.values(buttonPalette(themes[theme])), native[theme]);
  });
