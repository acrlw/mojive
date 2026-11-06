# MuJoCo runtime notices

The viewer bundles the unmodified single-threaded `@mujoco/mujoco` 3.17.0 WebAssembly
runtime. The npm package omits license files, so these notices are retained from the
matching official 3.17.0 release:

- Source: https://github.com/google-deepmind/mujoco/releases/tag/3.17.0
- Archive: `mujoco-3.17.0-linux-x86_64.tar.gz`, verified against its published SHA-256.
- `MuJoCo-LICENSE.txt`: upstream `LICENSE` (Apache-2.0).
- `MuJoCo-THIRD_PARTY_NOTICES.txt`: upstream `THIRD_PARTY_NOTICES`, retained in full.

Update these files with the pinned runtime version. Vite includes this directory in
browser builds, and desktop packaging preserves it in `dist/licenses/`.
