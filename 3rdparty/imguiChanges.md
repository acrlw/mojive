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

## Floating-window border contour

`RenderWindowOuterBorders` derives its stroke centerline and radius from the filled window's
outer contour. `AddRect` already shifts its input by half a point; retaining the fill radius
after that shift leaves a thin light/dark background halo around corners. The correction
accounts for the complete border width, including fractional UI scales. Native input,
content bounds and circular corners remain unchanged. The checked Bundle recipe applies
the same patch; rebuild with `make setup-imgui` and inspect both themes using
`make ui-focused-review`.

## Native rounded-rectangle tessellation

`ImDrawList::PathRect` uses `PathArcTo` for its four circular corners. Unlike the fixed fast-arc
entry point, this honors the error budget when a radius exceeds the lookup table's precision.
The Python window owner resolves `CircleTessellationMaxError` to 0.125 physical pixels after
refreshing framebuffer density, independently of application zoom. Small corners still use
the native lookup path; large corners receive the necessary additional segments. Silhouettes,
radii and native hit geometry remain circular and unchanged.

The Bundle recipe carries the same patch. `tests/test_draw2d.py` measures actual native
segment chord error across radii and display densities, and `make ui-popup-review` captures
native popup edges and production switches in both themes and at fractional scale.
