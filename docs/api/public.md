# Public package exports

This page is the supported import surface declared by `forge_viewer.__all__`. Each exported class,
protocol, function, enum, and value type links to its owning implementation where available.

::: forge_viewer

## Lazy MuJoCo audit exports

`audit_model` and `visual_coverage` are top-level exports loaded on first access so importing
`forge_viewer` remains physics-package neutral. Their owning module also defines the structured
finding and coverage value types.

::: forge_viewer.mujoco_audit
