PY := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
.DEFAULT_GOAL := help

.PHONY: rollout-video

.PHONY: help setup check lint fmt docs docs-check docs-serve examples-check test test-fast test-integration test-physics test-all gpu gpu-wgpu egl p0 p1 renderer-api renderer-api-wgpu renderer-benchmark renderer-benchmark-full golden golden-accept parity calibrate gallery ui-feasibility ui-runtime readme-media ui-frame-profile ui-gallery tool-icons mouse-icons gizmo-gallery hidpi-gallery model-loading model-composition mjcf-roundtrip editor-performance stability rpc-soak format-validation scene-io editor-files entity-edit undo-redo remote-authoring additive bench showcase probe reverse viewer egl-viewer hidpi empty editor settings workspace-edit canvas canvas-2d lighting image-light many-lights material-parity material-parity-accept texture-minification local-shadow-precision shadow-quality shadow-scheduling scene-icons scene-entities text-overlay capture record serve attach live-view snapshot-record snapshot-replay camera-state scene-snapshot cli rpc toy-physics adapter-conformance inspector gizmo joint-gizmo primitive-authoring material-authoring contact-authoring body-authoring resource-authoring asset-browser joint-site-authoring model-component-authoring keyframe-authoring batch-editing perturb reflect outline robot mujoco-physics mujoco-audit mujoco-model-suite mujoco-visuals mujoco-debug mujoco-actuators mujoco-slider-crank mujoco-solver-diagnostics mujoco-islands mujoco-bvh mujoco-convex-hull mujoco-rangefinder mujoco-constraints mujoco-editing mujoco-overlays cameras camera-intrinsics geom-groups deformables assets backends doctor clean

help:
	@printf '%s\n' \
		'Interactive:' \
		'  make viewer             default MuJoCo scene' \
		'  make egl-viewer         Linux viewer with a GLFW EGL context' \
		'  make hidpi              viewer with an explicit 200% UI scale' \
		'  make empty              empty viewer; load MJCF or URDF from File menu' \
		'  make editor             empty Mojive workspace; combine MJCF/URDF and entities' \
		'  make settings           editor with the dockable Settings panel open' \
		'  make workspace-edit     workspace, MjSpec topology, camera and light acceptance' \
		'  make model-loading      empty, MJCF, and URDF loading reference images' \
		'  make model-composition  add and remove MJCF/URDF models at runtime' \
		'  make mjcf-roundtrip     export edited workspaces as re-loadable MJCF' \
		'  make editor-performance composition editing performance baseline' \
		'  make stability          long-frame, large-model, and multi-camera gates' \
		'  make rpc-soak           persistent RPC, concurrency, timeout, and recovery' \
		'  make format-validation  current scene snapshot and recording formats' \
		'  make robot             Unitree Go2; downloads on first run' \
		'  make outline           selection and antialiased outline' \
		'  make inspector         compact Inspector transform reference image' \
		'  make ui-feasibility    interactive M1-M18 UI feasibility probe' \
		'  make ui-gallery        deterministic UI feasibility acceptance pages' \
		'  make readme-media      refresh unmodified production screenshots for README' \
		'  make tool-icons        transparent 1024px Tool Column icon sources' \
		'  make mouse-icons       black/transparent 1024px mouse hint sources' \
		'  make gizmo             2D/3D position/rotation gizmo' \
		'  make joint-gizmo       numbered joint-gizmo acceptance scene' \
		'  make primitive-authoring  fixed transforms and primitive dimensions' \
		'  make material-authoring  material creation, binding, and 2D texture import' \
		'  make contact-authoring  geometry contact, mass, group, and fluid properties' \
		'  make body-authoring     body mass, inertia, gravcomp, mocap, and sleep policy' \
		'  make resource-authoring geometry shape, mesh/hfield, cube, and skybox import' \
		'  make asset-browser     model-local files, materials, and height-field dimensions' \
		'  make joint-site-authoring advanced joint and site shape/endpoint properties' \
		'  make model-component-authoring contacts, actuators, sensors, tendons, and equality' \
		'  make keyframe-authoring record/replay take and snapshot Dope Sheet' \
		'  make batch-editing      Ctrl/Cmd multi-select and one-rebuild topology deletion' \
		'  make gizmo-gallery     enlarged 2D/3D gizmo reference images' \
		'  make hidpi-gallery     gizmo references at explicit 200% UI scale' \
		'  make perturb           MuJoCo translation/rotation perturbation' \
		'  make text-overlay      GPU world-space text' \
		'  make mujoco-visuals    hfield/site/tendon/contact' \
		'  make mujoco-debug      joint/COM/inertia debug visuals' \
		'  make mujoco-actuators  joint/site/body actuator visuals' \
		'  make mujoco-slider-crank  slider-crank linkage reference image' \
		'  make mujoco-solver-diagnostics  contact split and autoconnect images' \
		'  make mujoco-islands     constraint-island color reference images' \
		'  make mujoco-bvh         body, mesh, and flex BVH reference images' \
		'  make mujoco-convex-hull original and collision-hull reference images' \
		'  make mujoco-rangefinder site/camera rays, hits, and normals' \
		'  make mujoco-constraints  equality constraint endpoint markers' \
		'  make mujoco-editing     mocap pose and equality controls' \
		'  make mujoco-overlays    flex edges/vertices, labels, and frames' \
		'  make deformables       flex/skin dynamic meshes' \
		'' \
		'Rendering and output:' \
		'  make lighting          editable lights and environment' \
		'  make image-light       MuJoCo cube-map environment light' \
		'  make many-lights       16-light and 24-light reference images' \
		'  make material-parity   texture, transparency, tendon, deformable, and dense scenes' \
		'  make texture-minification  near/far textured-plane filtering acceptance' \
		'  make local-shadow-precision  anymal-c spotlight receiver precision acceptance' \
		'  make shadow-quality    performance/balanced/high shadow comparison' \
		'  make shadow-scheduling deterministic light and shadow-slot report' \
		'  make scene-icons       camera and light scene icons' \
		'  make reflect           multiple planar reflections' \
		'  make additive          standard and additive transparency images' \
		'  make cameras           free, named, and orthographic cameras' \
		'  make capture           write PNG' \
		'  make record            stream MP4' \
		'  make rollout-video     offscreen MP4 with simulation-time subtitles' \
		'  make showcase          render feature overview' \
		'' \
		'Backends and remote viewing:' \
		'  make canvas            standalone scene and material editor' \
		'  make canvas-2d         retained 2D physics/geometry debug canvas' \
		'  make scene-io          save, load, and capture a Mojive scene' \
		'  make editor-files      scene document workflow acceptance' \
		'  make entity-edit       Entity lifecycle acceptance' \
		'  make undo-redo        editor history and continuous edit acceptance' \
		'  make toy-physics       minimal independent physics backend' \
		'  make live-view         one publisher and two remote viewers' \
		'  make remote-authoring  runtime entity creation over Live View' \
		'  make snapshot-record   record remote scene snapshots' \
		'  make snapshot-replay   replay recorded snapshots' \
		'' \
		'Verification:' \
		'  make p0                complete Renderer compatibility gate' \
		'  make p1                complete P0 and P1 acceptance gate' \
		'  make test-fast         pure CPU behavior used during iteration' \
		'  make test-integration  file, protocol, and multi-module CPU tests' \
		'  make test-physics      tests that compile or step physics worlds' \
		'  make check             lint plus fast and integration CPU tests' \
		'  make test-all          CPU, physics, and both real GPU backends' \
		'  make gpu               real OpenGL tests' \
		'  make gpu-wgpu          real WebGPU tests' \
		'  make egl               Linux EGL Renderer and wireframe contract' \
		'  make scene-renderer    authored RGB, depth, and object-ID images' \
		'  make renderer-api      public Renderer CPU and GPU contract' \
		'  make renderer-api-wgpu public Renderer contract over wgpu' \
		'  make renderer-benchmark MuJoCo/OpenGL/wgpu public API timing comparison' \
		'  make renderer-benchmark-full complete resolution and output-mode matrix' \
		'  make camera-state      camera bookmark serialization and restore' \
		'  make scene-snapshot    complete scene-state serialization and restore' \
		'  make cli               typed local control commands' \
		'  make rpc               local RPC protocol and capture artifacts' \
		'  make agent-control     discover, edit, and verify an authored scene through RPC' \
		'  make agent-viewer      verify RPC edits and presented viewport/window capture' \
		'  make material-parity   material and dense-scene image baselines' \
		'  make shadow-scheduling deterministic light and shadow selection' \
		'  make doctor            window-path smoke test' \
		'  make mujoco-audit      MuJoCo visualization and core schema coverage' \
		'  make mujoco-model-suite compile, adapt, and render MuJoCo model collections' \
		'  make adapter-conformance  adapter contract report' \
		'  make docs              build the API and user guide under output/site' \
		'  make docs-check        validate public docs, examples, and the strict site build' \
		'  make examples-check    validate the runnable examples without opening a window' \
		'  make docs-serve        serve the documentation locally' \
		'' \
		'Display and backend options:' \
		'  make editor BACKEND=wgpu' \
		'  make editor LANGUAGE=zh_CN                 simplified Chinese UI' \
		'  MOJIVE_UI_SCALE=2 make editor' \
		'  MOJIVE_CJK_FONT=/path/font.otf make editor' \
		'  make hidpi BACKEND=wgpu UI_SCALE=2' \
		'  MOJIVE_UI_SCALE=1.5 make viewer BACKEND=wgpu SCENE=gizmo ARGS="--paused"' \
		'  make egl-viewer                   Linux GLFW EGL context' \
		'' \
		'BACKEND accepts opengl (OpenGL) or wgpu. Leave UI scale unset for automatic scaling.'

setup:
	uv sync --python 3.11 --extra dev --extra mujoco --extra wgpu

## Lint, formatting, and CPU tests.
check: lint test

lint:
	$(RUFF) check src tests tools examples
	$(RUFF) format --check src tests tools examples

fmt:
	$(RUFF) check --fix src tests tools examples
	$(RUFF) format src tests tools examples

docs:
	uv run --extra docs mkdocs build --strict --site-dir output/site

docs-check: examples-check
	$(PY) tools/check_docs.py
	$(MAKE) docs

examples-check:
	$(PY) -m compileall -q examples

docs-serve:
	uv run --extra docs mkdocs serve --strict --open

## Fast tests contain pure CPU behavior and module contracts.
test-fast:
	$(PYTEST) -q -m "not integration and not gpu and not physics and not slow and not golden"

## Integration tests cover files, protocols, processes, and composed modules.
test-integration:
	$(PYTEST) -q -m "integration and not gpu and not physics and not slow and not golden"

## The default CPU suite excludes real GPU and physics worlds.
test: test-fast test-integration

test-physics:
	$(PYTEST) -q -m physics

test-all: test test-physics gpu gpu-wgpu

mjcf-roundtrip:
	$(PYTEST) -q -m physics tests/test_workspace.py -k 'mjcf_export or exports_formatted or export_current_pose'

## Isolate files because OpenGL and physics libraries own process-global registries.
gpu:
	@for f in $$(ls tests/gpu/test_*.py); do echo "--- $$f"; $(PYTEST) -q -m "gpu or physics" $$f || exit 1; done

GPU_WGPU_FILES := tests/gpu/test_input_ownership.py tests/gpu/test_scene_renderer.py tests/gpu/test_renderer_api.py tests/gpu/test_control_rpc_capture.py tests/gpu/test_hidpi.py tests/gpu/test_horizon_haze.py tests/gpu/test_shading.py tests/gpu/test_shadows.py tests/gpu/test_reflection.py tests/gpu/test_outline.py tests/gpu/test_tendon.py tests/gpu/test_debugdraw.py tests/gpu/test_gizmo.py tests/gpu/test_pipeline.py tests/gpu/test_viewer_wgpu.py tests/gpu/test_static_viewer.py tests/gpu/test_model_loading.py tests/gpu/test_ui_interaction.py tests/gpu/test_wgpu_shader_reload.py
## Per-file GPU tests against the wgpu backend; extend GPU_WGPU_FILES as coverage grows.
## test_viewer_wgpu.py opens real (hidden-then-shown) windows and needs a display server, like the GL window tests.
gpu-wgpu:
	@export MOJIVE_RENDERER=wgpu; for f in $(GPU_WGPU_FILES); do echo "--- $$f"; $(PYTEST) -q -m "gpu or physics" $$f || exit 1; done

egl:
	@test "$$(uname -s)" = Linux || { echo 'make egl requires Linux'; exit 2; }
	MOJIVE_GL=egl $(PYTEST) -q -m gpu tests/gpu/test_renderer_api.py

.PHONY: scene-renderer
scene-renderer:
	$(PY) examples/offscreen_scene.py

renderer-api:
	$(PYTEST) -q tests/test_renderer_api.py
	$(PYTEST) -q -m gpu tests/gpu/test_renderer_api.py
	$(PY) -m mojive.tools.renderer_api

## Same Renderer API checks against the wgpu backend.
renderer-api-wgpu:
	MOJIVE_RENDERER=wgpu $(PYTEST) -q tests/test_renderer_api.py
	MOJIVE_RENDERER=wgpu $(PYTEST) -q -m gpu tests/gpu/test_renderer_api.py
	MOJIVE_RENDERER=wgpu $(PY) -m mojive.tools.renderer_api

renderer-benchmark:
	$(PY) -m mojive.tools.renderer_benchmark $(ARGS)

renderer-benchmark-full:
	$(PY) -m mojive.tools.renderer_benchmark --preset full \
		-o output/renderer-benchmark/full-report.json $(ARGS)

p0: renderer-api

p1: check p0 renderer-api-wgpu mujoco-physics camera-state scene-snapshot rpc material-parity shadow-scheduling mujoco-audit golden parity reverse gpu gpu-wgpu
	$(MAKE) adapter-conformance ADAPTER=mujoco CONFORMANCE_ASSET=deformables

## Compare golden images. Use golden-accept after visual review.
golden:
	$(PY) -m mojive.tools.golden

golden-accept:
	$(PY) -m mojive.tools.golden --accept

## Render OpenGL and the MuJoCo reference from one camera. The reference uses a subprocess.
parity:
	$(PY) -m mojive.tools.parity $(ARGS)

## Calibrate diffuse, headlight, ambient, and texture lighting against MuJoCo.
calibrate:
	$(PY) -m mojive.tools.calibrate

## Render the scene gallery for visual review.
gallery:
	$(PY) -m mojive.tools.gallery

ui-feasibility:
	$(PY) design/tools/render_ui_feasibility.py --interactive $(ARGS)

ui-runtime:
	$(PY) -m mojive.tools.ui_runtime $(ARGS)

## Refresh README images with unmodified production UI and renderer captures.
readme-media:
	MOJIVE_RENDERER=opengl $(PY) -m mojive.tools.ui_runtime \
		-o output/readme-media/runtime
	MOJIVE_RENDERER=opengl $(PY) -m mojive.tools.showcase \
		-o output/readme-media/showcase.png --width 1920 --height 1080
	$(PY) tools/build_readme_media.py \
		--runtime output/readme-media/runtime \
		--showcase output/readme-media/showcase.png \
		--output docs/images/readme

## Export production Tool Column geometry on transparent 1024px canvases.
tool-icons:
	$(PY) -m mojive.tools.tool_icons $(ARGS)

## Export production mouse hint geometry with black and transparent shells.
mouse-icons:
	$(PY) -m mojive.tools.mouse_hint_icons $(ARGS)

## Profile production viewport chrome and enforce its incremental frame budget.
ui-frame-profile:
	$(PY) -m mojive.tools.ui_frame_profile $(ARGS)

ui-gallery:
	$(PY) design/tools/render_ui_feasibility.py --page workspace -o output/ui-workspace.png
	$(PY) design/tools/render_ui_feasibility.py --page geometry --geometry-tab playback -o output/ui-geometry-playback.png
	$(PY) design/tools/render_ui_feasibility.py --page geometry --geometry-tab tools -o output/ui-geometry-tools.png
	$(PY) design/tools/render_ui_feasibility.py --page geometry --geometry-tab hints -o output/ui-geometry-hints.png
	$(PY) design/tools/render_ui_feasibility.py --page geometry --geometry-tab gizmos -o output/ui-geometry-transform-gizmos.png
	$(PY) design/tools/render_ui_feasibility.py --page geometry --geometry-tab helpers -o output/ui-geometry-joint-helpers.png
	$(PY) design/tools/render_ui_feasibility.py --page geometry --geometry-tab status -o output/ui-geometry-status.png
	$(PY) design/tools/render_ui_feasibility.py --page geometry --geometry-tab shell -o output/ui-geometry-shell.png
	$(PY) design/tools/render_ui_feasibility.py --page geometry --geometry-tab panels -o output/ui-geometry-panels.png
	$(PY) design/tools/render_ui_feasibility.py --page geometry --geometry-tab workspaces -o output/ui-geometry-workspaces.png
	$(PY) design/tools/render_ui_feasibility.py --ui-scale 4 --page workspace -o output/ui-workspace-hidpi.png
	$(PY) design/tools/render_ui_feasibility.py --ui-scale 4 --page panels -o output/ui-panels-hidpi.png
	$(PY) design/tools/render_ui_feasibility.py --ui-scale 4 --page geometry --geometry-tab helpers -o output/ui-geometry-joint-helpers-hidpi.png
	$(PY) -m mojive.tools.ui_runtime -o output/ui-runtime

gizmo-gallery:
	$(PY) -m mojive.tools.gizmo_gallery $(ARGS)

hidpi-gallery:
	MOJIVE_UI_SCALE=$(UI_SCALE) $(PY) -m mojive.tools.gizmo_gallery \
		-o output/gizmo-gallery-hidpi $(ARGS)

model-loading:
	$(PY) -m mojive.tools.model_loading $(ARGS)

model-composition:
	$(PYTEST) -q -m physics tests/test_adapter.py -k 'mjspec_model_composition'
	$(PY) -m mojive.tools.model_composition $(ARGS)

editor-performance:
	$(PYTEST) -q -m physics tests/test_editor_performance.py
	$(PY) -m mojive.tools.editor_performance $(ARGS)

stability: rpc-soak format-validation
	$(PYTEST) -q -m physics tests/test_stability.py
	MOJIVE_RENDERER=$(BACKEND) $(PYTEST) -q -m gpu tests/gpu/test_renderer_api.py -k 'multi_camera_concurrency'
	$(PY) -m mojive.tools.stability $(ARGS)

rpc-soak:
	$(PYTEST) -q -m physics tests/test_control_rpc.py -k 'reuses_one_connection or recovers_after_invalid or idle_connection or reconnects_on_the_call'

format-validation:
	$(PYTEST) -q tests/test_recording.py
	$(PYTEST) -q -m physics tests/test_scene_state.py -k 'version or current or future'

scene-io:
	$(PY) -m mojive.tools.scene_io $(ARGS)

editor-files:
	$(PYTEST) -q tests/test_static_scene.py -k 'document_commands'
	$(PY) -m mojive.tools.scene_io $(ARGS)

entity-edit:
	$(PYTEST) -q tests/test_static_scene.py -k 'entity_lifecycle'
	$(PYTEST) -q -m gpu tests/gpu/test_static_viewer.py -k 'editor_actions'

undo-redo:
	$(PYTEST) -q tests/test_static_scene.py -k 'undo_redo or history or edit_transaction'
	$(PYTEST) -q -m gpu tests/gpu/test_static_viewer.py -k 'undo_redo'

remote-authoring:
	$(PY) -m mojive.tools.remote_authoring $(ARGS)

additive:
	$(PY) -m mojive.tools.additive $(ARGS)

bench:
	$(PY) -m mojive.tools.bench

showcase:
	$(PY) -m mojive.tools.showcase

## Refresh the measurements recorded in docs/PLATFORM.md.
probe:
	$(PY) tools/probe_gl.py

## Apply registered mutations and confirm their regression checks fail.
reverse:
	$(PY) tools/reverse_verify.py

## Open an asset. Pass viewer flags through ARGS.
SCENE ?= test_scene
ARGS  ?=
BACKEND ?= opengl
LANGUAGE ?= $(MOJIVE_LANGUAGE)
viewer:
	MOJIVE_RENDERER=$(BACKEND) MOJIVE_LANGUAGE=$(LANGUAGE) $(PY) -m mojive.cli view $(SCENE) $(ARGS)

egl-viewer:
	@test "$$(uname -s)" = Linux || { echo 'make egl-viewer requires Linux'; exit 2; }
	MOJIVE_GL=egl PYOPENGL_PLATFORM=egl $(PY) -m mojive.cli view $(SCENE) $(ARGS)

UI_SCALE ?= 2
hidpi:
	MOJIVE_RENDERER=$(BACKEND) MOJIVE_LANGUAGE=$(LANGUAGE) MOJIVE_UI_SCALE=$(UI_SCALE) $(PY) -m mojive.cli view gizmo --paused $(ARGS)

## Open an empty MuJoCo scene and load MJCF or URDF from File > Open Model.
empty:
	MOJIVE_RENDERER=$(BACKEND) MOJIVE_LANGUAGE=$(LANGUAGE) $(PY) -m mojive.cli view empty --paused $(ARGS)

## Empty authored scene with New/Open/Save and Entity creation workflows.
editor:
	MOJIVE_RENDERER=$(BACKEND) MOJIVE_LANGUAGE=$(LANGUAGE) $(PY) -m mojive.cli editor $(ARGS)

workspace-edit:
	$(PYTEST) -q tests/test_workspace.py tests/test_scene_entities.py
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli editor $(ARGS)

## Programmatic scene, OpenGL rendering, and the standard UI.
canvas:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli canvas $(ARGS)

canvas-2d:
	MOJIVE_RENDERER=$(BACKEND) $(PY) examples/canvas2d.py $(ARGS)

## Independent physics adapter with gravity, collision, controls, and pose editing.
toy-physics:
	$(PY) -m mojive.cli toy $(ARGS)

ADAPTER ?= toy
CONFORMANCE_ASSET ?=
## Headless contract report for third-party adapters. MuJoCo example:
## make adapter-conformance ADAPTER=mujoco CONFORMANCE_ASSET=test_scene
adapter-conformance:
	$(PY) -m mojive.cli conformance $(ADAPTER) $(if $(CONFORMANCE_ASSET),--asset $(CONFORMANCE_ASSET),) $(ARGS)

## Editable lights and Environment controls for ambient light, fog, haze, and headlight.
lighting:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli canvas --demo lighting $(ARGS)

image-light:
	$(PY) -m mojive.cli view assets/image_light.xml --paused $(ARGS)

many-lights:
	$(PY) -m mojive.tools.mujoco_many_lights $(ARGS)

material-parity:
	$(PYTEST) -q tests/test_builder.py tests/test_scene.py
	$(PY) -m mojive.tools.material_parity $(ARGS)

material-parity-accept:
	$(PY) -m mojive.tools.material_parity --accept $(ARGS)

shadow-scheduling:
	$(PYTEST) -q tests/test_light_schedule.py
	$(PYTEST) -q -m gpu tests/gpu/test_shadows.py -k 'eight_local or local_light_indices'
	$(PY) -m mojive.tools.shadow_scheduling $(ARGS)

scene-icons:
	$(PY) -m mojive.cli canvas --demo lighting \
		--enable-render camera --enable-render light $(ARGS)

scene-entities:
	$(PYTEST) -q tests/test_scene_entities.py
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.tools.scene_entities $(ARGS)

## World anchors, screen offsets, alignment, and depth modes with the UI font.
text-overlay:
	$(PY) -m mojive.cli canvas --demo text $(ARGS)

OUTPUT ?= output/recording.mp4
SCREENSHOT ?= output/capture.png
## Capture at any resolution. Example: make capture SCENE=test_scene ARGS="--width 1920 --height 1080".
capture:
	$(PY) -m mojive.cli capture $(SCENE) -o $(SCREENSHOT) $(ARGS)

## Stream video encoding. Example: make record SCENE=test_scene ARGS="--frames 120".
record:
	$(PY) -m mojive.cli record $(SCENE) -o $(OUTPUT) $(ARGS)

## End-to-end Renderer + VideoRecorder + Pillow acceptance, without editor UI.
rollout-video:
	$(PY) examples/mujoco_video.py assets/test_scene.xml --label "Mojive rollout" \
		--output output/examples/rollout.mp4 $(ARGS)

LIVE_HOST ?= 127.0.0.1
LIVE_PORT ?= 47650
LIVE_SCENE ?= gizmo
## Run physics headlessly and publish the latest snapshot.
serve:
	$(PY) -m mojive.cli serve $(LIVE_SCENE) --host $(LIVE_HOST) --port $(LIVE_PORT) $(ARGS)

## Open an independent remote viewer.
attach:
	$(PY) -m mojive.cli attach --host $(LIVE_HOST) --port $(LIVE_PORT) $(ARGS)

## Start one physics publisher, one effect viewer, and one normal-debug viewer.
live-view:
	@$(PY) -m mojive.cli serve $(LIVE_SCENE) --host $(LIVE_HOST) --port $(LIVE_PORT) $(ARGS) & server=$$!; \
	$(PY) -m mojive.cli attach --host $(LIVE_HOST) --port $(LIVE_PORT) --title "Mojive effect" & effect=$$!; \
	$(PY) -m mojive.cli attach --host $(LIVE_HOST) --port $(LIVE_PORT) --title "Mojive debug" --debug-view normal & debug=$$!; \
	trap 'kill $$effect $$debug $$server 2>/dev/null || true' EXIT INT TERM; \
	wait $$effect; wait $$debug

SNAPSHOT ?= output/session.fvs
## Record structure revisions, physics frames, and debug commands.
snapshot-record:
	$(PY) -m mojive.cli serve $(LIVE_SCENE) --host $(LIVE_HOST) --port $(LIVE_PORT) --record-snapshot $(SNAPSHOT) $(ARGS)

## Replay a snapshot loop through the remote protocol.
snapshot-replay:
	@$(PY) -m mojive.cli replay $(SNAPSHOT) --host $(LIVE_HOST) --port $(LIVE_PORT) --loop & server=$$!; \
	trap 'kill $$server 2>/dev/null || true' EXIT INT TERM; \
	$(PY) -m mojive.cli attach --host $(LIVE_HOST) --port $(LIVE_PORT) --title "Mojive replay" $(ARGS)

camera-state:
	$(PY) -m pytest -q -m physics tests/test_scene_state.py -k camera
	$(PY) -m mojive.tools.scene_state $(ARGS)

scene-snapshot:
	$(PY) -m pytest -q -m physics tests/test_scene_state.py
	$(PY) -m mojive.tools.scene_state $(ARGS)

cli:
	$(PYTEST) -q -m physics tests/test_control_rpc.py

rpc: cli
	$(PYTEST) -q -m "gpu or physics" tests/gpu/test_control_rpc_capture.py
	$(PY) -m mojive.tools.control_rpc

.PHONY: agent-control agent-viewer
agent-control:
	$(PY) examples/agent_inspection.py $(ARGS)

agent-viewer:
	$(PY) examples/agent_inspection.py --viewer $(ARGS)

## Compact Inspector transform acceptance image.
inspector:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.tools.inspector $(ARGS)

## Native gizmo acceptance: G position, R rotation, T frame, F9 settings.
gizmo:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli view gizmo --paused $(ARGS)

## Numbered revolute, prismatic, ball, free, multi-joint, and compact-range acceptance.
joint-gizmo:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli editor joint_gizmo $(ARGS)

## Fixed-body transform and sphere/box/cylinder/capsule dimension authoring acceptance.
primitive-authoring:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli editor test_scene $(ARGS)

## Material creation/copy/binding and 2D image import acceptance.
material-authoring:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli editor test_scene $(ARGS)

## Geometry contact, solver, surface, mass, group, inertia, and fluid acceptance.
contact-authoring:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli editor test_scene $(ARGS)

## Body auto/explicit inertia, gravcomp, mocap, and sleep-policy acceptance.
body-authoring:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli editor test_scene $(ARGS)

## Geometry shape switching plus mesh, height-field, cube, and skybox import acceptance.
resource-authoring:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli editor test_scene $(ARGS)

## Model-local files, materials, and height-field physical-dimension acceptance.
asset-browser:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli editor test_scene $(ARGS)

## Joint dynamics/solver/force limits and site shape/group/endpoints acceptance.
joint-site-authoring:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli editor joint_gizmo $(ARGS)

## Contact, actuator, sensor, tendon, and equality component acceptance.
model-component-authoring:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli editor test_scene $(ARGS)

## Simulation-take transport plus model-local snapshot Dope Sheet acceptance.
keyframe-authoring:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli editor test_scene $(ARGS)

## Ctrl/Cmd multi-selection and one-rebuild model topology deletion acceptance.
batch-editing:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli editor test_scene $(ARGS)

## Dockable non-modal Settings acceptance.
settings:
	MOJIVE_OPEN_SETTINGS=1 MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli editor $(ARGS)

## Perturbation acceptance: Ctrl+left translates and Ctrl+right rotates a selected free body.
perturb:
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.cli view gizmo $(ARGS)

## Selection outline acceptance across multiple geoms and occlusion.
outline:
	$(PY) -m mojive.cli view outline --paused

## Multiple planar reflection acceptance for height, clipping, and winding.
reflect:
	$(PY) -m mojive.cli view reflection_multiple --paused

## Sparse checkout of one Google DeepMind MuJoCo Menagerie model.
ROBOT ?= unitree_go2
MENAGERIE_DIR ?= output/mujoco_menagerie
robot:
	@if [ ! -d "$(MENAGERIE_DIR)/.git" ]; then \
		git clone --depth 1 --filter=blob:none --sparse \
			https://github.com/google-deepmind/mujoco_menagerie.git "$(MENAGERIE_DIR)"; \
	fi
	@git -C "$(MENAGERIE_DIR)" sparse-checkout add assets "$(ROBOT)"
	$(PY) -m mojive.cli view "$(MENAGERIE_DIR)/$(ROBOT)/scene.xml" $(ARGS)

TEXTURE_MINIFICATION_SCENE ?= $(MENAGERIE_DIR)/anybotics_anymal_c/scene.xml
texture-minification:
	@test -f "$(TEXTURE_MINIFICATION_SCENE)" || { \
		echo "missing scene: $(TEXTURE_MINIFICATION_SCENE)"; \
		echo "set TEXTURE_MINIFICATION_SCENE=/path/to/anybotics_anymal_c/scene.xml"; \
		exit 2; \
	}
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.tools.texture_minification \
		"$(TEXTURE_MINIFICATION_SCENE)"

LOCAL_SHADOW_SCENE ?= $(MENAGERIE_DIR)/anybotics_anymal_c/scene.xml
local-shadow-precision:
	@test -f "$(LOCAL_SHADOW_SCENE)" || { \
		echo "missing scene: $(LOCAL_SHADOW_SCENE)"; \
		echo "set LOCAL_SHADOW_SCENE=/path/to/anybotics_anymal_c/scene.xml"; \
		exit 2; \
	}
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.tools.local_shadow_precision \
		"$(LOCAL_SHADOW_SCENE)"

shadow-quality:
	$(PYTEST) -q tests/test_cascades.py tests/test_panels.py -k shadow_quality
	MOJIVE_RENDERER=$(BACKEND) $(PY) -m mojive.tools.shadow_quality $(ARGS)

AUDIT_SCENE ?= mujoco_visuals
## Full MuJoCo adapter and simulation regression suite.
mujoco-physics:
	$(PYTEST) -q -m physics

## Headless MuJoCo visualization and exact linked-schema coverage report.
mujoco-audit:
	$(PY) -m mojive.cli audit $(AUDIT_SCENE) --strict

MUJOCO_MODEL_ROOTS ?= $(HOME)/Downloads/mujoco/model $(HOME)/Projects/PhysicsEngines/mujoco/model
MUJOCO_MODEL_REPORT ?= output/mujoco-model-suite.json
MUJOCO_MODEL_JOBS ?= 4
## Compile, adapt, and render every XML below the configured MuJoCo model roots.
mujoco-model-suite:
	$(PY) -m mojive.tools.mujoco_model_suite $(MUJOCO_MODEL_ROOTS) \
		--jobs $(MUJOCO_MODEL_JOBS) --report $(MUJOCO_MODEL_REPORT) $(ARGS)

## Interactive heightfield, site, tendon, and contact scene.
mujoco-visuals:
	$(PY) -m mojive.cli view mujoco_visuals \
		--enable-render tendon --enable-render contactpoint --enable-render contactforce $(ARGS)

## MuJoCo joint markers, root subtree COM and body inertia boxes.
mujoco-debug:
	$(PY) -m mojive.cli view joint_types --paused \
		--enable-render joint --enable-render com --enable-render inertia $(ARGS)

mujoco-actuators:
	$(PY) -m mojive.cli view actuator_visuals --paused \
		--camera overview --enable-render actuator --enable-render activation $(ARGS)

mujoco-slider-crank:
	$(PY) -m mojive.tools.mujoco_slider_crank

mujoco-solver-diagnostics:
	$(PY) -m mojive.tools.mujoco_solver_diagnostics

mujoco-islands:
	$(PY) -m mojive.tools.mujoco_islands

mujoco-bvh:
	$(PY) -m mojive.tools.mujoco_bvh $(ARGS)

mujoco-convex-hull:
	$(PY) -m mojive.tools.mujoco_convex_hull $(ARGS)

mujoco-rangefinder:
	$(PY) -m mojive.cli view rangefinder --paused \
		--enable-render rangefinder $(ARGS)

mujoco-constraints:
	$(PY) -m mojive.cli view constraints --paused \
		--camera overview --enable-render constraint $(ARGS)

mujoco-editing:
	$(PY) -m mojive.cli view mocap_equality --paused $(ARGS)

## Open Camera with F6; calibrated_shift demonstrates an off-center principal point.
cameras:
	$(PY) -m mojive.cli view mujoco_visuals $(ARGS)

camera-intrinsics:
	$(PY) -m mojive.cli capture mujoco_visuals --camera calibrated_shift \
		--width 1600 --height 1000 -o output/cameras/calibrated-shift.png $(ARGS)

## Inspect MuJoCo visual groups and synchronized picking filters in Settings.
geom-groups:
	$(PY) -m mojive.cli view mujoco_visuals --paused $(ARGS)

## Inspect 1D, 2D, and 3D flex geometry plus skinned meshes.
deformables:
	$(PY) -m mojive.cli view deformables --paused $(ARGS)

## Capture flex topology, scene labels, and coordinate-frame overlays.
mujoco-overlays:
	$(PY) -m mojive.tools.mujoco_overlays $(ARGS)

## List assets, free-body metadata, and optional dependency status.
assets:
	$(PY) -m mojive.cli assets

backends:
	$(PY) -m mojive.cli backends

doctor:
	$(PY) -m mojive.cli doctor $(SCENE) $(ARGS)

clean:
	rm -rf out .pytest_cache **/__pycache__
