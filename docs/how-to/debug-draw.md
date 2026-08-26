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

Use the singular methods (`line`, `arrow`, `point`, and `frame`) for a small number of primitives
that need independent retained IDs, expiration, or erasure. Use their plural batch forms
(`lines`, `arrows`, `points`, and `frames`) for homogeneous high-cardinality diagnostics. A batch
has one retained ID and replaces all of its records together, avoiding thousands of Python calls,
transport dictionaries, and retained-index entries.

```python
starts = robot_positions
ends = starts + robot_velocities * 0.1
depth.arrows("robot-velocities", starts, ends, (0.2, 0.7, 1.0, 1.0), 2.0)
```

Do not split a batch merely to assign numeric IDs to records. Split only where independent
lifetime or erasure semantics are required.
