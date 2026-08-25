# Debug drawing

Debug primitives are retained by stable string identifiers. Reusing an identifier updates its
storage in place, which keeps per-frame diagnostics compact. Widths, point radii, text offsets,
and arrow heads use screen pixels; anchors and segment endpoints use Z-up world coordinates.

Occlusion modes:

- `DEPTH`: normal scene depth testing;
- `ALWAYS`: overlay visible above scene geometry;
- `GHOST`: depth-aware diagnostic overlay.

## Interactive example

```bash
.venv/bin/python examples/debug_draw.py
```

```python
--8<-- "examples/debug_draw.py"
```

Use finite `duration` values for transient contacts or events. The default retains a primitive
until `erase()`, `Layer.clear()`, or `DebugDraw.clear()` removes it.
