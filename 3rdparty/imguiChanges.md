# Mojive changes to Dear ImGui

Upstream baseline: `v1.92.9b-docking`, commit `b48d1afbe8ee8b238e2961dc363a949dd7304e23`.
Provenance and archive checksum: `dependencies.json`. File checksums: `imguiBaseline.json`.

## Floating-window resize decoration

`imgui.cpp` omits the filled corner resize grips in `RenderWindowDecorations`. The triangular
fill crossed the content inset and made nested rounded panels appear clipped. Native hit
targets, resize cursors, drag/autofit behavior and outer-border feedback remain unchanged.

The same checked patch is applied to the Python Viewer's Bundle sources by
`python/tools/build_imgui.py`. Application-level transparent grip colors would also hide dock
splitter feedback, because Dear ImGui reuses those colors for splitters during `NewFrame`.
Rebuild the installed Bundle with `make setup-imgui`; validate actual floated panels with
`make ui-native-capture ARGS='--review --asset joint_gizmo'` and the resize regressions in
`tests/gpu/test_native_resize.py`.

The core retains upstream circular corner geometry; this patch does not introduce G3 curves.
