# Python and C++ development

Mojive remains a Python package. `mojive.Renderer`, `SceneRenderer`, Viewer entry points and
existing snake_case methods remain the user-facing API. C++ supplies selected implementation
workloads; a future private `mojive._native` extension will sit behind the Python facade.
The current C++ tools are acceptance fixtures, not a replacement standalone product or an
already-integrated Python renderer.

## Source layout

```text
python/
  src/mojive/        Python package and compatibility facade
  tests/            Python tests, GPU tests and reviewed golden fixtures
  bindingTests/     Optional native binding comparison tests
cpp/
  include/mojive/   Backend-neutral C++ contracts
  src/             Common code and private renderer adapters
  bindings/        C++ binding translation units
  tools/           Native acceptance fixtures and benchmarks
  tests/           C++ contract tests
  cmake/           Build integration
thirdParty/        Vendored ImGui and pinned upstream submodules
examples/          Python usage examples
tools/             Repository maintenance and verification scripts
output/            Generated builds, captures and reports
```

The root `pyproject.toml` remains the Python build entry point. Imports never contain `python`,
`cpp`, `thirdParty`, or a backend library name. C++ headers describe Mojive's contracts rather
than exposing bgfx handles, ImGui types, or MuJoCo-owned mutable state.

## Naming

Mojive-owned C++ uses a Qt-like style without adopting Qt or its `Q` type prefix:

| Item | Style | Example |
|---|---|---|
| Types, enums and enum values | PascalCase | `SceneFrame`, `ReadbackState::Pending` |
| Functions and methods | camelCase | `setScene()`, `createTarget()`, `waitForReadback()` |
| Parameters, locals and public fields | camelCase | `objectId`, `cameraRevision` |
| Private class members | `m` + PascalCase | `mDevice`, `mReadbacks` |
| Namespaces | camelCase | `mojive`, `mojive::bindingProbe` |
| Compound source filenames | camelCase | `sceneStream.hpp`, `sceneCapture.cpp` |

Use `.hpp` / `.cpp`, four spaces and the committed `cpp/.clang-format`. The optional
`cpp/.clang-tidy` names the same identifier rules. Upstream APIs, standard-library names,
platform/compiler macros, existing build target names and Python ABI names keep their required
spelling; they are not mechanically renamed to match Mojive's C++ style.

Python keeps its existing snake_case function/module names and PascalCase classes. Do not add
a second public spelling merely because C++ uses camelCase. The binding layer translates
between conventions, preserving argument names, return shapes, dtypes, context-manager behavior
and error semantics. For example, Python `update_scene()` stays unchanged even if the native
implementation uses `setScene()`.

## Local preparation

```bash
uv sync --extra dev --extra docs --extra mujoco
make cpp-deps
make check
make cpp-test
make cpp-probe
```

Python installation remains a pure-Python Hatch build for this preparation step. Its wheel
contains the existing Python package, shaders and bundled assets; it does not yet contain the
native renderer. Native wheel integration is a subsequent feature with its own compatibility
and distribution checks.

`cpp-*` targets are convenient entry points; the earlier `native-*` acceptance commands remain
available. C++ builds use fresh `output/cpp-build`, `output/cpp-core-build` and
`output/cpp-bindings-build` directories so previous evaluation artifacts stay readable.
The renderer is selected at the composition root; SDL comparison remains opt-in.

The user deferred workflow CI. No evaluation workflow is installed by this branch. Linux
validation will be performed on the user's Linux system; macOS verification does not establish
Windows/Linux runtime support. Local commands and coverage are in the
[native verification guide](native-probe.md) and [verification matrix](testing.md).

## Dependency changes

Read the repository's `thirdParty/README.md` before editing an upstream source tree.
Dear ImGui is tracked directly to make custom drawing changes reviewable alongside Mojive.
Its first import has no local behavior patches. The native library and Python `imgui-bundle`
are separate until integration; editing the former does not change the latter's installed wheel.

The [native dependency plan (Chinese)](../plans/cpp-dependencies.zh.md) recommends GLM for
graphics math and spdlog for runtime-owned native output and bounded log history. Python can
publish records, configure output and subscribe through optional Loguru/logging bridges; native
output must not depend on a Python consumer. C++ provides rendering and runtime infrastructure;
business algorithms and extension policy remain in Python. Eigen and native business solvers are
outside the current roadmap. EnTT requires a demonstrated infrastructure need. These recommended
libraries are not yet dependencies in the current build.
