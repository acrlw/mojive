PY := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
.DEFAULT_GOAL := help

.PHONY: help setup check lint fmt test gpu gpu-wgpu egl p0 p1 renderer-api renderer-api-wgpu golden golden-accept parity calibrate gallery gizmo-gallery hidpi-gallery model-loading model-composition scene-io editor-files entity-edit undo-redo remote-authoring additive bench showcase probe reverse viewer egl-viewer hidpi empty editor canvas lighting image-light many-lights material-parity material-parity-accept shadow-scheduling scene-icons text-overlay capture record serve attach live-view snapshot-record snapshot-replay camera-state scene-snapshot cli rpc toy-physics adapter-conformance gizmo perturb reflect outline robot mujoco-physics mujoco-audit mujoco-visuals mujoco-debug mujoco-actuators mujoco-slider-crank mujoco-solver-diagnostics mujoco-islands mujoco-bvh mujoco-convex-hull mujoco-rangefinder mujoco-constraints mujoco-editing mujoco-overlays mujoco-ik cameras camera-intrinsics geom-groups deformables assets backends doctor clean

help:
	@printf '%s\n' \
		'Interactive:' \
		'  make viewer             default MuJoCo scene' \
		'  make egl-viewer         Linux viewer with a GLFW EGL context' \
		'  make hidpi              viewer with an explicit 200% UI scale' \
		'  make empty              empty viewer; load MJCF or URDF from File menu' \
		'  make editor             empty Forge scene with file and Entity authoring menus' \
		'  make model-loading      empty, MJCF, and URDF loading reference images' \
		'  make model-composition  add and remove MJCF/URDF models at runtime' \
		'  make robot             Unitree Go2; downloads on first run' \
		'  make outline           selection and antialiased outline' \
		'  make gizmo             2D/3D position/rotation gizmo' \
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
		'  make shadow-scheduling deterministic light and shadow-slot report' \
		'  make scene-icons       camera and light scene icons' \
		'  make reflect           multiple planar reflections' \
		'  make additive          standard and additive transparency images' \
		'  make cameras           free, named, and orthographic cameras' \
		'  make capture           write PNG' \
		'  make record            stream MP4' \
		'  make showcase          render feature overview' \
		'' \
		'Backends and remote viewing:' \
		'  make canvas            standalone scene and material editor' \
		'  make scene-io          save, load, and capture a Forge scene' \
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
		'  make check             lint and CPU tests' \
		'  make gpu               real OpenGL tests' \
		'  make gpu-wgpu          real WebGPU tests' \
		'  make egl               Linux EGL Renderer and wireframe contract' \
		'  make renderer-api      public Renderer CPU and GPU contract' \
		'  make renderer-api-wgpu public Renderer contract over wgpu' \
		'  make mujoco-ik         body/site inverse-kinematics acceptance' \
		'  make camera-state      camera bookmark serialization and restore' \
		'  make scene-snapshot    complete scene-state serialization and restore' \
		'  make cli               typed local control commands' \
		'  make rpc               local RPC protocol and capture artifacts' \
		'  make material-parity   material and dense-scene image baselines' \
		'  make shadow-scheduling deterministic light and shadow selection' \
		'  make doctor            window-path smoke test' \
		'  make mujoco-audit      MuJoCo visualization coverage' \
		'  make adapter-conformance  adapter contract report' \
		'' \
		'Example: make viewer SCENE=humanoid ARGS="--paused"'

setup:
	uv sync --python 3.11 --extra dev --extra mujoco --extra wgpu

## Lint, formatting, and CPU tests.
check: lint test

lint:
	$(RUFF) check src tests tools
	$(RUFF) format --check src tests tools

fmt:
	$(RUFF) check --fix src tests tools
	$(RUFF) format src tests tools

## The default suite excludes real GPU and physics worlds.
test:
	$(PYTEST) -q

## Isolate files because OpenGL and physics libraries own process-global registries.
gpu:
	@for f in $$(ls tests/gpu/test_*.py); do echo "--- $$f"; $(PYTEST) -q -m "gpu or physics" $$f || exit 1; done

GPU_WGPU_FILES := tests/gpu/test_renderer_api.py tests/gpu/test_control_rpc_capture.py tests/gpu/test_hidpi.py tests/gpu/test_horizon_haze.py tests/gpu/test_shading.py tests/gpu/test_shadows.py tests/gpu/test_reflection.py tests/gpu/test_outline.py tests/gpu/test_tendon.py tests/gpu/test_debugdraw.py tests/gpu/test_gizmo.py tests/gpu/test_pipeline.py tests/gpu/test_viewer_wgpu.py tests/gpu/test_static_viewer.py tests/gpu/test_model_loading.py tests/gpu/test_ui_interaction.py tests/gpu/test_wgpu_shader_reload.py
## Per-file GPU tests against the wgpu backend; extend GPU_WGPU_FILES as coverage grows.
## test_viewer_wgpu.py opens real (hidden-then-shown) windows and needs a display server, like the GL window tests.
gpu-wgpu:
	@export FORGE_VIEWER_BACKEND=wgpu; for f in $(GPU_WGPU_FILES); do echo "--- $$f"; $(PYTEST) -q -m "gpu or physics" $$f || exit 1; done

egl:
	@test "$$(uname -s)" = Linux || { echo 'make egl requires Linux'; exit 2; }
	FORGE_VIEWER_GL=egl $(PYTEST) -q -m gpu tests/gpu/test_renderer_api.py

renderer-api:
	$(PYTEST) -q tests/test_renderer_api.py
	$(PYTEST) -q -m gpu tests/gpu/test_renderer_api.py
	$(PY) -m forge_viewer.tools.renderer_api

## Same Renderer API checks against the wgpu backend.
renderer-api-wgpu:
	FORGE_VIEWER_BACKEND=wgpu $(PYTEST) -q tests/test_renderer_api.py
	FORGE_VIEWER_BACKEND=wgpu $(PYTEST) -q -m gpu tests/gpu/test_renderer_api.py
	FORGE_VIEWER_BACKEND=wgpu $(PY) -m forge_viewer.tools.renderer_api

p0: renderer-api

p1: check p0 renderer-api-wgpu mujoco-physics mujoco-ik camera-state scene-snapshot rpc material-parity shadow-scheduling mujoco-audit golden parity reverse gpu gpu-wgpu
	$(MAKE) adapter-conformance ADAPTER=mujoco CONFORMANCE_ASSET=deformables

## Compare golden images. Use golden-accept after visual review.
golden:
	$(PY) -m forge_viewer.tools.golden

golden-accept:
	$(PY) -m forge_viewer.tools.golden --accept

## Render Forge and the MuJoCo reference from one camera. The reference uses a subprocess.
parity:
	$(PY) -m forge_viewer.tools.parity

## Calibrate diffuse, headlight, ambient, and texture lighting against MuJoCo.
calibrate:
	$(PY) -m forge_viewer.tools.calibrate

## Render the scene gallery for visual review.
gallery:
	$(PY) -m forge_viewer.tools.gallery

gizmo-gallery:
	$(PY) -m forge_viewer.tools.gizmo_gallery $(ARGS)

hidpi-gallery:
	FORGE_VIEWER_UI_SCALE=$(UI_SCALE) $(PY) -m forge_viewer.tools.gizmo_gallery \
		-o output/gizmo-gallery-hidpi $(ARGS)

model-loading:
	$(PY) -m forge_viewer.tools.model_loading $(ARGS)

model-composition:
	$(PYTEST) -q -m physics tests/test_adapter.py -k 'mjspec_model_composition'
	$(PY) -m forge_viewer.tools.model_composition $(ARGS)

scene-io:
	$(PY) -m forge_viewer.tools.scene_io $(ARGS)

editor-files:
	$(PYTEST) -q tests/test_static_scene.py -k 'document_commands'
	$(PY) -m forge_viewer.tools.scene_io $(ARGS)

entity-edit:
	$(PYTEST) -q tests/test_static_scene.py -k 'entity_lifecycle'
	$(PYTEST) -q -m gpu tests/gpu/test_static_viewer.py -k 'editor_actions'

undo-redo:
	$(PYTEST) -q tests/test_static_scene.py -k 'undo_redo or history or edit_transaction'
	$(PYTEST) -q -m gpu tests/gpu/test_static_viewer.py -k 'undo_redo'

remote-authoring:
	$(PY) -m forge_viewer.tools.remote_authoring $(ARGS)

additive:
	$(PY) -m forge_viewer.tools.additive $(ARGS)

bench:
	$(PY) -m forge_viewer.tools.bench

showcase:
	$(PY) -m forge_viewer.tools.showcase

## Refresh the measurements recorded in docs/PLATFORM.md.
probe:
	$(PY) tools/probe_gl.py

## Apply registered mutations and confirm their regression checks fail.
reverse:
	$(PY) tools/reverse_verify.py

## Open an asset. Pass viewer flags through ARGS.
SCENE ?= test_scene
ARGS  ?=
viewer:
	$(PY) -m forge_viewer.cli view $(SCENE) $(ARGS)

egl-viewer:
	@test "$$(uname -s)" = Linux || { echo 'make egl-viewer requires Linux'; exit 2; }
	FORGE_VIEWER_GL=egl $(PY) -m forge_viewer.cli view $(SCENE) $(ARGS)

UI_SCALE ?= 2
hidpi:
	FORGE_VIEWER_UI_SCALE=$(UI_SCALE) $(PY) -m forge_viewer.cli view gizmo --paused $(ARGS)

## Open an empty MuJoCo scene and load MJCF or URDF from File > Open Model.
empty:
	$(PY) -m forge_viewer.cli view empty --paused $(ARGS)

## Empty authored scene with New/Open/Save and Entity creation workflows.
editor:
	$(PY) -m forge_viewer.cli canvas --demo empty $(ARGS)

## Programmatic scene, Forge rendering, and the standard UI.
canvas:
	$(PY) -m forge_viewer.cli canvas $(ARGS)

## Independent physics adapter with gravity, collision, controls, and pose editing.
toy-physics:
	$(PY) -m forge_viewer.cli toy $(ARGS)

ADAPTER ?= toy
CONFORMANCE_ASSET ?=
## Headless contract report for third-party adapters. MuJoCo example:
## make adapter-conformance ADAPTER=mujoco CONFORMANCE_ASSET=test_scene
adapter-conformance:
	$(PY) -m forge_viewer.cli conformance $(ADAPTER) $(if $(CONFORMANCE_ASSET),--asset $(CONFORMANCE_ASSET),) $(ARGS)

## Editable lights and Environment controls for ambient light, fog, haze, and headlight.
lighting:
	$(PY) -m forge_viewer.cli canvas --demo lighting $(ARGS)

image-light:
	$(PY) -m forge_viewer.cli view assets/image_light.xml --paused $(ARGS)

many-lights:
	$(PY) -m forge_viewer.tools.mujoco_many_lights $(ARGS)

material-parity:
	$(PYTEST) -q tests/test_builder.py tests/test_scene.py
	$(PY) -m forge_viewer.tools.material_parity $(ARGS)

material-parity-accept:
	$(PY) -m forge_viewer.tools.material_parity --accept $(ARGS)

shadow-scheduling:
	$(PYTEST) -q tests/test_light_schedule.py
	$(PYTEST) -q -m gpu tests/gpu/test_shadows.py -k 'eight_local or local_light_indices'
	$(PY) -m forge_viewer.tools.shadow_scheduling $(ARGS)

scene-icons:
	$(PY) -m forge_viewer.cli canvas --demo lighting \
		--enable-render camera --enable-render light $(ARGS)

## World anchors, screen offsets, alignment, and depth modes with the UI font.
text-overlay:
	$(PY) -m forge_viewer.cli canvas --demo text $(ARGS)

OUTPUT ?= output/recording.mp4
SCREENSHOT ?= output/capture.png
## Capture at any resolution. Example: make capture SCENE=humanoid ARGS="--width 1920 --height 1080".
capture:
	$(PY) -m forge_viewer.cli capture $(SCENE) -o $(SCREENSHOT) $(ARGS)

## Stream video encoding. Example: make record SCENE=humanoid ARGS="--frames 120".
record:
	$(PY) -m forge_viewer.cli record $(SCENE) -o $(OUTPUT) $(ARGS)

LIVE_HOST ?= 127.0.0.1
LIVE_PORT ?= 47650
LIVE_SCENE ?= gizmo
## Run physics headlessly and publish the latest snapshot.
serve:
	$(PY) -m forge_viewer.cli serve $(LIVE_SCENE) --host $(LIVE_HOST) --port $(LIVE_PORT) $(ARGS)

## Open an independent remote viewer.
attach:
	$(PY) -m forge_viewer.cli attach --host $(LIVE_HOST) --port $(LIVE_PORT) $(ARGS)

## Start one physics publisher, one effect viewer, and one normal-debug viewer.
live-view:
	@$(PY) -m forge_viewer.cli serve $(LIVE_SCENE) --host $(LIVE_HOST) --port $(LIVE_PORT) $(ARGS) & server=$$!; \
	$(PY) -m forge_viewer.cli attach --host $(LIVE_HOST) --port $(LIVE_PORT) --title "forge effect" & effect=$$!; \
	$(PY) -m forge_viewer.cli attach --host $(LIVE_HOST) --port $(LIVE_PORT) --title "forge debug" --debug-view normal & debug=$$!; \
	trap 'kill $$effect $$debug $$server 2>/dev/null || true' EXIT INT TERM; \
	wait $$effect; wait $$debug

SNAPSHOT ?= output/session.fvs
## Record structure revisions, physics frames, and debug commands.
snapshot-record:
	$(PY) -m forge_viewer.cli serve $(LIVE_SCENE) --host $(LIVE_HOST) --port $(LIVE_PORT) --record-snapshot $(SNAPSHOT) $(ARGS)

## Replay a snapshot loop through the remote protocol.
snapshot-replay:
	@$(PY) -m forge_viewer.cli replay $(SNAPSHOT) --host $(LIVE_HOST) --port $(LIVE_PORT) --loop & server=$$!; \
	trap 'kill $$server 2>/dev/null || true' EXIT INT TERM; \
	$(PY) -m forge_viewer.cli attach --host $(LIVE_HOST) --port $(LIVE_PORT) --title "forge replay" $(ARGS)

camera-state:
	$(PY) -m pytest -q -m physics tests/test_scene_state.py -k camera
	$(PY) -m forge_viewer.tools.scene_state $(ARGS)

scene-snapshot:
	$(PY) -m pytest -q -m physics tests/test_scene_state.py
	$(PY) -m forge_viewer.tools.scene_state $(ARGS)

cli:
	$(PYTEST) -q -m physics tests/test_control_rpc.py

rpc: cli
	$(PYTEST) -q -m "gpu or physics" tests/gpu/test_control_rpc_capture.py
	$(PY) -m forge_viewer.tools.control_rpc

## Native gizmo acceptance: G position, R rotation, T frame, F9 settings.
gizmo:
	$(PY) -m forge_viewer.cli view gizmo --paused $(ARGS)

## Perturbation acceptance: Ctrl+left translates and Ctrl+right rotates a selected free body.
perturb:
	$(PY) -m forge_viewer.cli view gizmo $(ARGS)

## Selection outline acceptance across multiple geoms and occlusion.
outline:
	$(PY) -m forge_viewer.cli view outline --paused

## Multiple planar reflection acceptance for height, clipping, and winding.
reflect:
	$(PY) -m forge_viewer.cli view reflection_multiple --paused

## Sparse checkout of one Google DeepMind MuJoCo Menagerie model.
ROBOT ?= unitree_go2
MENAGERIE_DIR ?= output/mujoco_menagerie
robot:
	@if [ ! -d "$(MENAGERIE_DIR)/.git" ]; then \
		git clone --depth 1 --filter=blob:none --sparse \
			https://github.com/google-deepmind/mujoco_menagerie.git "$(MENAGERIE_DIR)"; \
	fi
	@git -C "$(MENAGERIE_DIR)" sparse-checkout add assets "$(ROBOT)"
	$(PY) -m forge_viewer.cli view "$(MENAGERIE_DIR)/$(ROBOT)/scene.xml" $(ARGS)

AUDIT_SCENE ?= mujoco_visuals
## Full MuJoCo adapter and simulation regression suite.
mujoco-physics:
	$(PYTEST) -q -m physics

## Headless MuJoCo visualization coverage report.
mujoco-audit:
	$(PY) -m forge_viewer.cli audit $(AUDIT_SCENE) --strict

## Interactive heightfield, site, tendon, and contact scene.
mujoco-visuals:
	$(PY) -m forge_viewer.cli view mujoco_visuals \
		--enable-render tendon --enable-render contactpoint --enable-render contactforce $(ARGS)

## MuJoCo joint markers, root subtree COM and body inertia boxes.
mujoco-debug:
	$(PY) -m forge_viewer.cli view joint_types --paused \
		--enable-render joint --enable-render com --enable-render inertia $(ARGS)

mujoco-actuators:
	$(PY) -m forge_viewer.cli view actuator_visuals --paused \
		--camera overview --enable-render actuator --enable-render activation $(ARGS)

mujoco-slider-crank:
	$(PY) -m forge_viewer.tools.mujoco_slider_crank

mujoco-solver-diagnostics:
	$(PY) -m forge_viewer.tools.mujoco_solver_diagnostics

mujoco-islands:
	$(PY) -m forge_viewer.tools.mujoco_islands

mujoco-bvh:
	$(PY) -m forge_viewer.tools.mujoco_bvh $(ARGS)

mujoco-convex-hull:
	$(PY) -m forge_viewer.tools.mujoco_convex_hull $(ARGS)

mujoco-rangefinder:
	$(PY) -m forge_viewer.cli view rangefinder --paused \
		--enable-render rangefinder $(ARGS)

mujoco-constraints:
	$(PY) -m forge_viewer.cli view constraints --paused \
		--camera overview --enable-render constraint $(ARGS)

mujoco-editing:
	$(PY) -m forge_viewer.cli view mocap_equality --paused $(ARGS)

## Open Camera with F6; calibrated_shift demonstrates an off-center principal point.
cameras:
	$(PY) -m forge_viewer.cli view mujoco_visuals $(ARGS)

camera-intrinsics:
	$(PY) -m forge_viewer.cli capture mujoco_visuals --camera calibrated_shift \
		--width 1600 --height 1000 -o output/cameras/calibrated-shift.png $(ARGS)

## Inspect MuJoCo visual groups and synchronized picking filters in Settings.
geom-groups:
	$(PY) -m forge_viewer.cli view mujoco_visuals --paused $(ARGS)

## Inspect 1D, 2D, and 3D flex geometry plus skinned meshes.
deformables:
	$(PY) -m forge_viewer.cli view deformables --paused $(ARGS)

## Capture flex topology, scene labels, and coordinate-frame overlays.
mujoco-overlays:
	$(PY) -m forge_viewer.tools.mujoco_overlays $(ARGS)

mujoco-ik:
	$(PY) -m pytest -q tests/test_mujoco_ik.py
	$(PY) -m forge_viewer.tools.mujoco_ik $(ARGS)

## List assets, free-body metadata, and optional dependency status.
assets:
	$(PY) -m forge_viewer.cli assets

backends:
	$(PY) -m forge_viewer.cli backends

doctor:
	$(PY) -m forge_viewer.cli doctor $(SCENE) $(ARGS)

clean:
	rm -rf out .pytest_cache **/__pycache__
