PY := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
.DEFAULT_GOAL := help

.PHONY: help setup check lint fmt test gpu golden golden-accept parity calibrate gallery gizmo-gallery bench showcase probe reverse viewer canvas lighting scene-icons text-overlay capture record serve attach pvd snapshot-record snapshot-replay toy-physics adapter-conformance gizmo perturb reflect outline robot mujoco-audit mujoco-visuals mujoco-debug mujoco-actuators mujoco-rangefinder mujoco-constraints mujoco-editing musculoskeletal musculoskeletal-video musculoskeletal-check cameras camera-intrinsics geom-groups deformables assets backends doctor clean

help:
	@printf '%s\n' \
		'Interactive:' \
		'  make viewer             default MuJoCo scene' \
		'  make robot             Unitree Go2; downloads on first run' \
		'  make outline           selection and antialiased outline' \
		'  make gizmo             2D/3D position/rotation gizmo' \
		'  make gizmo-gallery     enlarged 2D/3D gizmo reference images' \
		'  make perturb           MuJoCo translation/rotation perturbation' \
		'  make text-overlay      GPU world-space text' \
		'  make mujoco-visuals    hfield/site/tendon/contact' \
		'  make mujoco-debug      joint/COM/inertia debug visuals' \
		'  make mujoco-actuators  joint/site/body actuator visuals' \
		'  make mujoco-rangefinder site/camera rays, hits, and normals' \
		'  make mujoco-constraints  equality constraint endpoint markers' \
		'  make mujoco-editing     mocap pose and equality controls' \
		'  make musculoskeletal   musculoskeletal model, tendons, and keyframes' \
		'  make musculoskeletal-video  300 keyframes → 60 fps MP4' \
		'  make deformables       flex/skin dynamic meshes' \
		'' \
		'Rendering and output:' \
		'  make lighting          editable spot/point/area lights, fog, and haze' \
		'  make scene-icons       camera and light scene icons' \
		'  make reflect           planar reflection' \
		'  make cameras           free, named, and orthographic cameras' \
		'  make capture           write PNG' \
		'  make record            stream MP4' \
		'  make showcase          render feature overview' \
		'' \
		'Backends and remote viewing:' \
		'  make canvas            standalone 3D canvas' \
		'  make toy-physics       minimal independent physics backend' \
		'  make pvd               one physics process and two viewers' \
		'  make snapshot-record   record remote scene snapshots' \
		'  make snapshot-replay   replay recorded snapshots' \
		'' \
		'Verification:' \
		'  make check             lint and CPU tests' \
		'  make gpu               real OpenGL tests' \
		'  make doctor            window-path smoke test' \
		'  make mujoco-audit      MuJoCo visualization coverage' \
		'  make adapter-conformance  adapter contract report' \
		'' \
		'Example: make viewer SCENE=humanoid ARGS="--paused"'

setup:
	uv venv --python 3.11
	uv pip install -e ".[dev,mujoco]"

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

## Editable spot, point, and area lights with fog and haze.
lighting:
	$(PY) -m forge_viewer.cli canvas --demo lighting $(ARGS)

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

PVD_HOST ?= 127.0.0.1
PVD_PORT ?= 47650
PVD_SCENE ?= gizmo
## Run physics headlessly and publish the latest snapshot.
serve:
	$(PY) -m forge_viewer.cli serve $(PVD_SCENE) --host $(PVD_HOST) --port $(PVD_PORT) $(ARGS)

## Open an independent remote viewer.
attach:
	$(PY) -m forge_viewer.cli attach --host $(PVD_HOST) --port $(PVD_PORT) $(ARGS)

## Start one physics publisher, one effect viewer, and one normal-debug viewer.
pvd:
	@$(PY) -m forge_viewer.cli serve $(PVD_SCENE) --host $(PVD_HOST) --port $(PVD_PORT) $(ARGS) & server=$$!; \
	$(PY) -m forge_viewer.cli attach --host $(PVD_HOST) --port $(PVD_PORT) --title "forge effect" & effect=$$!; \
	$(PY) -m forge_viewer.cli attach --host $(PVD_HOST) --port $(PVD_PORT) --title "forge debug" --debug-view normal & debug=$$!; \
	trap 'kill $$effect $$debug $$server 2>/dev/null || true' EXIT INT TERM; \
	wait $$effect; wait $$debug

SNAPSHOT ?= out/session.fvs
## Record structure revisions, physics frames, and debug commands.
snapshot-record:
	$(PY) -m forge_viewer.cli serve $(PVD_SCENE) --host $(PVD_HOST) --port $(PVD_PORT) --record-snapshot $(SNAPSHOT) $(ARGS)

## Replay a snapshot loop through the remote protocol.
snapshot-replay:
	@$(PY) -m forge_viewer.cli replay $(SNAPSHOT) --host $(PVD_HOST) --port $(PVD_PORT) --loop & server=$$!; \
	trap 'kill $$server 2>/dev/null || true' EXIT INT TERM; \
	$(PY) -m forge_viewer.cli attach --host $(PVD_HOST) --port $(PVD_PORT) --title "forge replay" $(ARGS)

## Native gizmo acceptance: G position, R rotation, T frame, F9 settings.
gizmo:
	$(PY) -m forge_viewer.cli view gizmo --paused $(ARGS)

## Perturbation acceptance: Ctrl+left translates and Ctrl+right rotates a selected free body.
perturb:
	$(PY) -m forge_viewer.cli view gizmo $(ARGS)

## Selection outline acceptance across multiple geoms and occlusion.
outline:
	$(PY) -m forge_viewer.cli view outline --paused

## Planar reflection acceptance for height, clipping, and winding.
reflect:
	$(PY) -m forge_viewer.cli view reflection --paused

## Sparse checkout of one Google DeepMind MuJoCo Menagerie model.
ROBOT ?= unitree_go2
MENAGERIE_DIR ?= out/mujoco_menagerie
robot:
	@if [ ! -d "$(MENAGERIE_DIR)/.git" ]; then \
		git clone --depth 1 --filter=blob:none --sparse \
			https://github.com/google-deepmind/mujoco_menagerie.git "$(MENAGERIE_DIR)"; \
	fi
	@git -C "$(MENAGERIE_DIR)" sparse-checkout add assets "$(ROBOT)"
	$(PY) -m forge_viewer.cli view "$(MENAGERIE_DIR)/$(ROBOT)/scene.xml" $(ARGS)

AUDIT_SCENE ?= mujoco_visuals
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

mujoco-rangefinder:
	$(PY) -m forge_viewer.cli view rangefinder --paused \
		--enable-render rangefinder $(ARGS)

mujoco-constraints:
	$(PY) -m forge_viewer.cli view constraints --paused \
		--camera overview --enable-render constraint $(ARGS)

mujoco-editing:
	$(PY) -m forge_viewer.cli view mocap_equality --paused $(ARGS)

MYO_SCENE ?= ../lowerlimb-refactor/lowerlimb-main/assets/models/myo_sim_latest/myo_sim/body/fullbody_kit_9_10_walk_forward_60.xml
MYO_VIDEO ?= output/musculoskeletal-keyframes-60fps.mp4
## Paused musculoskeletal model with 300 keyframes and tendon paths.
musculoskeletal:
	$(PY) -m forge_viewer.cli view "$(MYO_SCENE)" --paused $(ARGS)

## Render 300 keyframes as a five-second, 60 fps video.
musculoskeletal-video:
	$(PY) -m forge_viewer.cli keyframes "$(MYO_SCENE)" -o "$(MYO_VIDEO)" --fps 60 \
		--camera cam_track $(ARGS)

## Verify structure, full frame data, dynamic geometry, and visualization coverage.
musculoskeletal-check:
	$(PY) -m forge_viewer.cli conformance mujoco --asset "$(MYO_SCENE)"
	$(PY) -m forge_viewer.cli audit "$(MYO_SCENE)" --strict

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

## List assets, free-body metadata, and optional dependency status.
assets:
	$(PY) -m forge_viewer.cli assets

backends:
	$(PY) -m forge_viewer.cli backends

doctor:
	$(PY) -m forge_viewer.cli doctor $(SCENE) $(ARGS)

clean:
	rm -rf out .pytest_cache **/__pycache__
