# Windows

These commands assume a prepared source checkout with a Windows `.venv` and
native extension. See the [development guide](../guides/development.md) and
[native backend setup](native-viewer.md) for building the package.


Run from the repository in PowerShell:

```powershell
$env:MOJIVE_RENDERER = "bgfx"
.\.venv\Scripts\python.exe -m mojive.cli doctor joint_types --frames 90
.\.venv\Scripts\python.exe -m mojive.cli view joint_types
```

The native Windows bgfx build uses Direct3D 12. Set `MOJIVE_RENDERER` to `opengl`
to select OpenGL. Inspect `doctor`'s GPU device and frame checks after changing drivers.

Set the entire UI scale before starting the application:

```powershell
$env:MOJIVE_UI_SCALE = "1.5"
.\.venv\Scripts\python.exe -m mojive.cli editor
```

Use `1`, `1.25`, `1.5`, or `2` as appropriate. Close and reopen an existing window
for an environment-variable change to take effect. To return to automatic display scaling:

```powershell
Remove-Item Env:MOJIVE_UI_SCALE -ErrorAction SilentlyContinue
```

This scales fonts, controls, and overlays. The viewport's separate overlay/capsule
size controls do not change the entire application's scale.
