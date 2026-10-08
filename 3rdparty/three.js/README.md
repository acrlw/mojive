# Three.js browser reference dependency

The UI design study uses unmodified files from
[Three.js r180](https://github.com/mrdoob/three.js/tree/r180). Each copied file was
compared byte-for-byte with its upstream URL. `../dependencies.json` records the
version, source URLs and SHA-256 digests. The upstream MIT license is retained in
`LICENSE` and the JavaScript license headers are unchanged.

`three.core.js` and `three.module.js` come from upstream `build/`;
`OrbitControls.js` and `TransformControls.js` come from
`examples/jsm/controls/`. The web reference's import map resolves their imports
without rewriting upstream source. These files are used only by the browser
example; they are not Python package or native renderer dependencies.
