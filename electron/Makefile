.DEFAULT_GOAL := help

.PHONY: help package bundle-smoke clean
help:
	@printf "%s\n" "make install       Install locked Node dependencies" "make run           Build and launch Electron" "make dev           Start the renderer development server" "make check         Build and run regression tests" "make acceptance    Check and capture desktop workflows" "make physics-review Verify WASM, assets and state replay" "make package       Build a macOS app in ../output/electron/release" "make bundle-smoke  Verify the packaged application" "make clean         Remove generated Electron artifacts"

package:
	npm run package
bundle-smoke:
	npm run bundle-smoke
	node scripts/physics-review.mjs --packaged
clean:
	rm -rf dist public/menu-icons ../output/electron

.PHONY: ensure-deps install dev run check acceptance design viewcube-review ui-review viewport-review timeline-review controls-review layout-review camera-review camera-self-review project-review scrollbar-review switch-review actuator-review
ensure-deps:
	node scripts/ensure-deps.cjs
install:
	node scripts/ensure-deps.cjs --force
package bundle-smoke dev run check design actuator-review viewcube-review ui-review viewport-review timeline-review timeline-wheel-review controls-review layout-review camera-review camera-self-review project-review scrollbar-review switch-review: ensure-deps
dev:
	npm run dev
run:
	npm start
check:
	npm run build
	npm test
acceptance: check
	npm run acceptance
design:
	node scripts/sync-design.mjs
actuator-review:
	npm run build
	node --test --test-concurrency=1 tests/joint-heading.test.mjs tests/joint-panel.test.mjs tests/panel-responsive.test.mjs
	node scripts/actuator-cards-review.mjs
viewcube-review:
	node scripts/review-viewcube.mjs
ui-review:
	npm run build
	node scripts/workspace-acceptance.mjs
viewport-review:
	node --test --test-concurrency=1 tests/gizmo-feedback.test.mjs tests/gizmo-precise.test.mjs tests/gizmo-configuration.test.mjs tests/gizmo-interactions.test.mjs tests/gizmo-ribbon.test.mjs tests/joint-interactions.test.mjs tests/hinge-axis-interactions.test.mjs tests/viewport-selection.test.mjs tests/selection-ownership.test.mjs

timeline-review:
	node --test --test-concurrency=1 tests/timeline-navigation.test.mjs tests/timeline.test.mjs tests/timeline-workflows.test.mjs tests/scene-timeline.test.mjs

.PHONY: timeline-wheel-review
timeline-wheel-review:
	npm run build
	$(MAKE) timeline-review
	node scripts/timeline-wheel-review.mjs

controls-review:
	node --test --test-concurrency=1 tests/shared-controls.test.mjs tests/numeric-interaction.test.mjs
	node scripts/review-production-controls.mjs
	node scripts/review-numeric-input.mjs

layout-review:
	node --test --test-concurrency=1 tests/inspector-layout.test.mjs tests/inspector-spacing.test.mjs tests/inspector-lifecycle.test.mjs tests/panel-responsive.test.mjs tests/shared-controls.test.mjs tests/switch-row.test.mjs tests/switch-row-integration.test.mjs tests/workspace-chrome.test.mjs tests/settings-layout.test.mjs tests/operation-notice.test.mjs tests/notice-history.test.mjs tests/rail-buttons.test.mjs tests/joint-heading.test.mjs tests/scrollbars.test.mjs

camera-review:
	node --test --test-concurrency=1 tests/scene-camera.test.mjs tests/camera-preview.test.mjs tests/camera-self-gizmo.test.mjs tests/camera-cards.test.mjs tests/camera-align-icon.test.mjs tests/camera-navigation.test.mjs tests/camera-transition.test.mjs tests/scene-camera-transitions.test.mjs tests/camera-focus.test.mjs tests/joint-focus.test.mjs tests/camera-focus-integration.test.mjs

camera-self-review:
	npm run build
	node scripts/camera-self-review.mjs

project-review:
	npm run compile-menu-icons
	node --test --test-concurrency=1 tests/file-access.test.mjs tests/file-dialog.test.mjs tests/native-menu.test.mjs tests/project-workflows.test.mjs tests/project-directory.test.mjs tests/project-desktop.test.mjs tests/scene.test.mjs

scrollbar-review:
	npm run build
	node --test tests/scrollbars.test.mjs
	node scripts/review-scrollbars.mjs

switch-review:
	npm run build
	node --test --test-concurrency=1 tests/switch-row.test.mjs tests/switch-row-integration.test.mjs
	node scripts/review-switch-rows.mjs

.PHONY: menu-review
menu-review: ensure-deps
	npm run build
	node --test --test-concurrency=1 tests/menu-interactions.test.mjs tests/nested-corners.test.mjs tests/input-hints.test.mjs tests/input-bindings.test.mjs tests/inspector-layout.test.mjs tests/confirmation-output.test.mjs
	node scripts/workspace-acceptance.mjs

.PHONY: localization-review
localization-review: ensure-deps
	node --test --test-concurrency=1 tests/localization.test.mjs tests/native-menu.test.mjs tests/notice-history.test.mjs

.PHONY: corners-review
corners-review: ensure-deps
	node --test --test-concurrency=1 tests/smooth-geometry.test.mjs tests/smooth-capsule.test.mjs tests/capsule-layout.test.mjs tests/nested-corners.test.mjs tests/shared-controls.test.mjs tests/gizmo-feedback.test.mjs

.PHONY: physics-review
physics-review: ensure-deps
	npm run build
	node --test --test-concurrency=1 tests/physics-backend.test.mjs tests/physics-scene.test.mjs tests/physics-worker.test.mjs tests/physics-workspace.test.mjs tests/simulation-history.test.mjs tests/simulation-keyboard.test.mjs tests/assets.test.mjs tests/perturbation-input.test.mjs tests/camera-focus-integration.test.mjs
	node scripts/physics-review.mjs
