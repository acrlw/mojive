# Scene and workspace files

Mojive uses explicit `format` and `version` fields for JSON documents. The current authored-scene
and workspace schemas are version 1. Writers always emit Mojive format identifiers; readers retain
the previous `forge-viewer.*` identifiers only to migrate files created before the project rename.

## Mojive scene JSON

::: mojive.scene_io

`mojive.scene` stores one programmatic scene: materials, meshes, textures, objects, environment,
lights, cameras, and ID allocators. It does not contain model references or simulation state.

## Composed workspace documents

::: mojive.workspace_io

`mojive.workspace` embeds one authored scene and adds MJCF/URDF model references, model root
transforms, resource search directories, and normalized edited MJCF where needed. Missing model
references are reported before loading; the repair functions can replace one path or search a
directory for unambiguous filename matches.

File-less MuJoCo edits are stored as inline `root_mjcf`. A workspace is the editable project
format; MJCF export is the portable runtime format.

## Other persisted formats

- `.fvs` snapshot recordings use `mojive.snapshot-recording` version 2 and contain pickled remote
  structure/frame packets. Load them only from trusted sources.
- complete scene-state snapshots use `mojive.scene-snapshot` version 2;
- camera bookmark JSON uses version 1; and
- video output is streamed through FFmpeg and is not a Mojive document format.

Current readers reject future or truncated formats with an explicit error instead of guessing.
