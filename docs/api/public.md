# Public package exports

This page is the supported import surface declared by `mojive.__all__`. Each exported class,
protocol, function, enum, and value type links to its owning implementation where available.

::: mojive

## Lazy MuJoCo audit exports

`audit_model` and `visual_coverage` are top-level exports loaded on first access so importing
`mojive` remains physics-package neutral. Their owning module also defines the structured
finding and coverage value types.

::: mojive.mujoco_audit
