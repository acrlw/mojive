import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// A local scene for evaluating editor interactions. It does not run a physics adapter.
export function createViewport(host, state, onSelect, onFrame, onTransform) {
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.setClearColor(0x2a3038);
  renderer.domElement.className = "scene-canvas";
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute(
    "aria-label",
    "3D viewport. Drag to orbit, scroll to zoom, click an object to select.",
  );
  host.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x2a3038, 17, 46);
  const perspective = new THREE.PerspectiveCamera(42, 1, 0.05, 100);
  const orthographic = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.05, 100);
  for (const cam of [perspective, orthographic]) {
    cam.up.set(0, 0, 1);
    cam.position.set(4.3, -5.9, 4.2);
  }
  let camera = perspective;
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.55);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.maxDistance = 25;
  controls.minDistance = 1;
  controls.maxPolarAngle = Math.PI * 0.96;
  scene.add(new THREE.HemisphereLight(0xe5eefc, 0x303640, 2.1));
  const sun = new THREE.DirectionalLight(0xffeedb, 3.0);
  sun.position.set(-3, -4, 8);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, {
    left: -6,
    right: 6,
    top: 6,
    bottom: -6,
    near: 0.1,
    far: 25,
  });
  sun.shadow.bias = -0.001;
  scene.add(sun);
  const texCanvas = document.createElement("canvas");
  texCanvas.width = 128;
  texCanvas.height = 128;
  const ctx = texCanvas.getContext("2d");
  ctx.fillStyle = "#343b45";
  ctx.fillRect(0, 0, 128, 128);
  ctx.fillStyle = "#414b58";
  ctx.fillRect(0, 0, 64, 64);
  ctx.fillRect(64, 64, 64, 64);
  const floorTexture = new THREE.CanvasTexture(texCanvas);
  floorTexture.wrapS = floorTexture.wrapT = THREE.RepeatWrapping;
  floorTexture.repeat.set(40, 40);
  floorTexture.colorSpace = THREE.SRGBColorSpace;
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(80, 80),
    new THREE.MeshStandardMaterial({ map: floorTexture, roughness: 1 }),
  );
  floor.receiveShadow = true;
  floor.position.z = -0.015;
  scene.add(floor);
  const grid = new THREE.GridHelper(40, 80, 0x57616d, 0x444e5b);
  grid.rotation.x = Math.PI / 2;
  grid.visible = false;
  scene.add(grid);
  const group = new THREE.Group();
  scene.add(group);
  const selection = new THREE.BoxHelper(new THREE.Object3D(), 0xe8b04f);
  selection.visible = false;
  selection.material.depthTest = true;
  selection.material.transparent = true;
  selection.material.opacity = 0.85;
  selection.renderOrder = 10;
  scene.add(selection);
  const transform = new TransformControls(camera, renderer.domElement);
  transform.setSize(0.8);
  scene.add(transform.getHelper());
  let usedGizmo = false;
  transform.addEventListener("dragging-changed", (e) => {
    controls.enabled = !e.value;
  });
  transform.addEventListener("mouseDown", () => {
    usedGizmo = true;
    onTransform("start");
  });
  function transformValues() {
    const o = transform.object,
      e = state.entities.find((e) => e.id === state.selected);
    if (!o || !e) return null;
    const position = o.position.toArray(),
      rotation = [o.rotation.x, o.rotation.y, o.rotation.z].map(
        (x) => (x * 180) / Math.PI,
      );
    if (e.id === "hinge_body") rotation[0] -= state.pose.hinge;
    if (e.id === "ball_body") rotation[1] -= state.pose.ball;
    if (e.id === "slide_body") position[1] -= state.pose.slide;
    return {
      id: e.id,
      position,
      rotation,
      scale: o.scale.toArray().map((x) => Math.max(0.01, x)),
    };
  }
  transform.addEventListener("objectChange", () =>
    onTransform("preview", transformValues()),
  );
  transform.addEventListener("mouseUp", () =>
    onTransform("commit", transformValues()),
  );
  const objects = new Map();
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const materials = [];
  function mat(color, roughness = 0.55) {
    const m = new THREE.MeshStandardMaterial({
      color,
      roughness,
      metalness: 0.08,
    });
    m.userData.authored = color !== "#535c67";
    materials.push(m);
    return m;
  }
  function mesh(geo, material, parent, x = 0, y = 0, z = 0) {
    const m = new THREE.Mesh(geo, material);
    m.position.set(x, y, z);
    m.castShadow = true;
    m.receiveShadow = true;
    parent.add(m);
    return m;
  }
  function cylinder(r, length, material, parent, x = 0, y = 0, z = 0) {
    const m = mesh(
      new THREE.CylinderGeometry(r, r, length, 32),
      material,
      parent,
      x,
      y,
      z,
    );
    m.rotation.x = Math.PI / 2;
    return m;
  }
  function rebuild() {
    transform.detach();
    group.traverse((o) => {
      if (o.isMesh) o.geometry.dispose();
    });
    group.clear();
    materials.splice(0).forEach((m) => m.dispose());
    objects.clear();
    for (const e of state.entities) {
      if (
        !["link", "geom", "light", "camera"].includes(e.type) ||
        e.shape === "floor"
      )
        continue;
      const g = new THREE.Group();
      g.userData.entityId = e.id;
      group.add(g);
      objects.set(e.id, g);
      const color = mat(e.color, e.roughness ?? 0.5),
        dark = mat("#535c67");
      if (e.shape === "hinge") {
        cylinder(0.15, 0.15, dark, g);
        const rod = cylinder(0.072, 0.85, color, g, 0, 0.4, 0);
        rod.rotation.x = 0;
        mesh(new THREE.SphereGeometry(0.072, 24, 16), color, g, 0, 0.83, 0);
      } else if (e.shape === "ball") {
        mesh(new THREE.BoxGeometry(0.28, 0.28, 0.25), dark, g, 0, 0, 0.42);
        const rod = cylinder(0.068, 0.75, color, g, 0, 0, -0.05);
        rod.rotation.y = -0.2;
        mesh(new THREE.SphereGeometry(0.14, 32, 24), color, g, -0.07, 0, -0.43);
      } else if (e.shape === "slide") {
        const rail = cylinder(0.029, 1.3, dark, g);
        rail.rotation.x = 0;
        mesh(new THREE.BoxGeometry(0.35, 0.35, 0.35), color, g);
      } else if (e.shape === "chain") {
        cylinder(0.055, 1.1, color, g, 0, 0, 0.38);
        mesh(new THREE.BoxGeometry(0.24, 0.24, 0.24), dark, g, 0, 0, -0.2);
        mesh(new THREE.SphereGeometry(0.058, 16, 16), color, g, 0, 0, 0.93);
      } else if (e.type === "light") {
        mesh(new THREE.SphereGeometry(0.08, 12, 12), mat("#e5d08c"), g);
        const h = new THREE.PointLightHelper(
          new THREE.PointLight(0xe5d08c),
          0.2,
        );
        g.add(h);
      } else if (e.type === "camera") {
        mesh(new THREE.BoxGeometry(0.22, 0.16, 0.14), dark, g);
        const lens = cylinder(0.09, 0.13, dark, g, 0, 0.12);
        lens.rotation.x = 0;
      } else if (e.shape === "sphere")
        mesh(new THREE.SphereGeometry(0.3, 40, 32), color, g);
      else if (e.shape === "cylinder") cylinder(0.3, 0.7, color, g);
      else if (e.shape === "capsule") {
        const part = mesh(
          new THREE.CapsuleGeometry(0.2, 0.45, 8, 24),
          color,
          g,
        );
        part.rotation.x = Math.PI / 2;
      } else {
        mesh(new THREE.BoxGeometry(0.5, 0.5, 0.5), color, g);
        if (e.id === "free_body") cylinder(0.07, 0.32, color, g, 0, 0, 0.36);
      }
    }
    update();
  }
  function update() {
    for (const e of state.entities) {
      const g = objects.get(e.id);
      if (!g) continue;
      g.position.fromArray(e.position);
      g.rotation.set(...e.rotation.map((v) => (v * Math.PI) / 180));
      g.scale.fromArray(e.scale);
      if (e.id === "hinge_body")
        g.rotation.x += (state.pose.hinge * Math.PI) / 180;
      if (e.id === "ball_body")
        g.rotation.y += (state.pose.ball * Math.PI) / 180;
      if (e.id === "slide_body") g.position.y += state.pose.slide;
      g.visible = !e.hidden && state.layers[e.type] !== false;
      g.traverse((o) => {
        if (o.isMesh) {
          o.material.wireframe = state.shading === "Wireframe";
          if (o.material.userData.authored) o.material.color.set(e.color);
          o.material.roughness = e.roughness;
        }
      });
    }
    floor.visible = state.layers.floor !== false;
    const nextMap = state.checker ? floorTexture : null;
    if (floor.material.map !== nextMap) {
      floor.material.map = nextMap;
      floor.material.needsUpdate = true;
    }
    floor.material.color.set(state.checker ? 0xffffff : 0x333a44);
    grid.visible = state.grid;
    renderer.shadowMap.enabled = state.shadows;
    sun.intensity = state.entities.find((e) => e.type === "light" && !e.hidden)
      ? state.lightIntensity
      : 0;
    const lightEntity = state.entities.find((e) => e.type === "light");
    if (lightEntity) {
      sun.position.fromArray(lightEntity.position);
      sun.color.set(lightEntity.color);
    }
    transform.setMode(
      { move: "translate", rotate: "rotate", scale: "scale" }[state.tool] ||
        "translate",
    );
    transform.setSpace(state.frame === "Body" ? "local" : "world");
    transform.setTranslationSnap(state.snap ? 0.1 : null);
    transform.setRotationSnap(state.snap ? Math.PI / 12 : null);
    transform.setScaleSnap(state.snap ? 0.1 : null);
    const target = objects.get(state.selected);
    if (
      target &&
      target.visible &&
      state.tool !== "select" &&
      !state.playing &&
      state.gizmos
    )
      transform.attach(target);
    else transform.detach();
    refreshSelection();
  }
  function refreshSelection() {
    const e = state.entities.find((e) => e.id === state.selected);
    const selected = objects.get(state.selected);
    selection.visible = !!selected && selected.visible && state.outlines;
    if (selection.visible) {
      selected.updateWorldMatrix(true, true);
      selection.setFromObject(selected);
    }
  }

  function resize() {
    const w = host.clientWidth,
      h = host.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    perspective.aspect = w / h;
    perspective.updateProjectionMatrix();
    orthographic.left = (-5 * w) / h;
    orthographic.right = (5 * w) / h;
    orthographic.updateProjectionMatrix();
  }
  const observer = new ResizeObserver(resize);
  observer.observe(host);
  function projectMode(ortho) {
    const next = ortho ? orthographic : perspective;
    if (next !== camera) {
      next.position.copy(camera.position);
      next.quaternion.copy(camera.quaternion);
      if (ortho) {
        orthographic.zoom =
          5 /
          (Math.tan((perspective.fov * Math.PI) / 360) *
            camera.position.distanceTo(controls.target));
      }
      camera = next;
      transform.camera = camera;
      controls.object = camera;
      controls.update();
      resize();
    }
  }
  function orient(axis) {
    state.cameraLabel =
      {
        X: "Right",
        "-X": "Left",
        Y: "Back",
        "-Y": "Front",
        Z: "Top",
        "-Z": "Bottom",
      }[axis] || "Perspective";
    host.querySelector(".view-name").textContent = state.cameraLabel;
    const distance = camera.position.distanceTo(controls.target);
    const vec = {
      X: [1, 0, 0],
      "-X": [-1, 0, 0],
      Y: [0, 1, 0],
      "-Y": [0, -1, 0],
      Z: [0.0001, 0, 1],
      "-Z": [0.0001, 0, -1],
      Perspective: [0.58, -0.8, 0.57],
    }[axis] || [0.58, -0.8, 0.57];
    camera.position
      .copy(controls.target)
      .add(new THREE.Vector3(...vec).normalize().multiplyScalar(distance));
    controls.update();
  }
  function frame(all = false) {
    const selected = objects.get(state.selected);
    const bounds = new THREE.Box3();
    if (selected && !all) bounds.setFromObject(selected);
    else
      for (const o of objects.values())
        if (
          o.visible &&
          state.entities.find((e) => e.id === o.userData.entityId)?.type !==
            "light"
        )
          bounds.expandByObject(o);
    if (bounds.isEmpty())
      bounds.setFromCenterAndSize(
        new THREE.Vector3(0, 0, 0.5),
        new THREE.Vector3(5, 3, 2),
      );
    const center = bounds.getCenter(new THREE.Vector3()),
      size = bounds.getSize(new THREE.Vector3()).length();
    const direction = camera.position.clone().sub(controls.target).normalize();
    controls.target.copy(center);
    const distance = Math.max(
      1.5,
      ((size / (2 * Math.tan((perspective.fov * Math.PI) / 360))) * 1.35) /
        Math.min(1, perspective.aspect),
    );
    camera.position
      .copy(center)
      .add(direction.multiplyScalar(Math.min(24, distance)));
    if (state.ortho) {
      camera.zoom = 5 / Math.max(0.6, size * 0.7);
      camera.updateProjectionMatrix();
    }
    controls.update();
  }
  let press = null;
  renderer.domElement.addEventListener("pointerdown", (e) => {
    usedGizmo = !!transform.axis;
    press = [e.clientX, e.clientY];
  });
  renderer.domElement.addEventListener("pointermove", (e) => {
    if (
      e.buttons &&
      press &&
      !usedGizmo &&
      !transform.axis &&
      Math.hypot(e.clientX - press[0], e.clientY - press[1]) > 4
    ) {
      state.cameraLabel = state.ortho ? "User orthographic" : "Perspective";
      host.querySelector(".view-name").textContent = state.cameraLabel;
    }
  });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (
      usedGizmo ||
      !!transform.axis ||
      !press ||
      Math.hypot(e.clientX - press[0], e.clientY - press[1]) > 4 ||
      e.button !== 0
    )
      return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      (-(e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster
      .intersectObjects(group.children, true)
      .find((h) => h.object.visible && h.object.parent.visible);
    let o = hit?.object;
    while (o && !o.userData.entityId) o = o.parent;
    onSelect(o?.userData.entityId || null);
  });
  const nav = host.querySelector(".nav-widget");
  nav.innerHTML = `<svg viewBox="0 0 104 104" aria-label="Orientation navigation"><circle class="nav-backdrop" cx="52" cy="52" r="50"/><g class="nav-lines"></g><circle cx="52" cy="52" r="3" fill="#e6e8eb"/><g class="nav-axes"></g></svg>`;
  const svgNS = "http://www.w3.org/2000/svg";
  const navItems = [];
  for (let axis = 0; axis < 3; axis++)
    for (const sign of [-1, 1]) {
      const name = (sign < 0 ? "-" : "") + "XYZ"[axis],
        color = ["#dc7773", "#68b578", "#7b9fea"][axis];
      const g = document.createElementNS(svgNS, "g");
      g.dataset.axis = name;
      g.setAttribute("role", "button");
      g.setAttribute("aria-label", "View from " + name);
      g.setAttribute("tabindex", "0");
      g.innerHTML = `<circle r="10" fill="${sign > 0 ? color : "#24282d"}" stroke="${color}" stroke-width="${sign > 0 ? 0 : 1.5}"/><text y="3.4" font-size="10" font-weight="650" fill="${sign > 0 ? "#15202b" : color}" text-anchor="middle">${sign > 0 ? "XYZ"[axis] : ""}</text><title>View from ${name}</title>`;
      nav.querySelector(".nav-axes").append(g);
      const line = document.createElementNS(svgNS, "line");
      line.setAttribute("x1", "52");
      line.setAttribute("y1", "52");
      line.setAttribute("stroke", color);
      line.setAttribute("stroke-width", "2");
      if (sign > 0) nav.querySelector(".nav-lines").append(line);
      navItems.push({ axis, sign, g, line, z: 0 });
    }
  function drawNavigation() {
    const rotation = new THREE.Matrix4().extractRotation(
      camera.matrixWorldInverse,
    );
    nav
      .querySelector(".nav-backdrop")
      .setAttribute("fill", `rgba(22,25,28,${state.navOpacity})`);
    for (const item of navItems) {
      const v = new THREE.Vector3();
      v.setComponent(item.axis, item.sign);
      v.applyMatrix4(rotation);
      const x = 52 + v.x * 32,
        y = 52 - v.y * 32;
      item.z = v.z;
      item.g.setAttribute("transform", `translate(${x} ${y})`);
      item.line.setAttribute("x2", x);
      item.line.setAttribute("y2", y);
    }
    // Retain focused nodes while updating depth order, so keyboard navigation is stable.
    if (!nav.contains(document.activeElement))
      for (const item of [...navItems].sort((a, b) => a.z - b.z))
        nav.querySelector(".nav-axes").append(item.g);
  }
  nav.addEventListener("keydown", (e) => {
    if (["Enter", " "].includes(e.key)) {
      e.preventDefault();
      const a = e.target.closest("[data-axis]")?.dataset.axis;
      if (a) orient(a);
    }
  });
  let navPress = null,
    navDragged = false,
    navTarget = null;
  nav.addEventListener("pointerdown", (e) => {
    navPress = [e.clientX, e.clientY];
    navDragged = false;
    navTarget =
      e.target.closest("[data-axis]")?.dataset.axis ||
      (e.target.closest(".nav-center") ? "projection" : null);
    nav.setPointerCapture(e.pointerId);
  });
  nav.addEventListener("pointermove", (e) => {
    if (!navPress) return;
    const dx = e.clientX - navPress[0],
      dy = e.clientY - navPress[1];
    if (Math.abs(dx) + Math.abs(dy) > 2) navDragged = true;
    if (navDragged) {
      state.cameraLabel = state.ortho ? "User orthographic" : "Perspective";
      host.querySelector(".view-name").textContent = state.cameraLabel;
      const offset = camera.position.clone().sub(controls.target);
      const spherical = new THREE.Spherical().setFromVector3(
        offset.clone().applyAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2),
      );
      spherical.theta -= dx * 0.008;
      spherical.phi = Math.max(
        0.01,
        Math.min(Math.PI - 0.01, spherical.phi - dy * 0.008),
      );
      offset
        .setFromSpherical(spherical)
        .applyAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
      camera.position.copy(controls.target).add(offset);
      controls.update();
      navPress = [e.clientX, e.clientY];
    }
  });
  nav.addEventListener("pointerup", () => {
    if (!navDragged && navTarget) {
      if (navTarget === "projection") {
        state.ortho = !state.ortho;
        projectMode(state.ortho);
      } else {
        orient(navTarget);
      }
      onSelect(undefined);
    }
    navPress = null;
    navTarget = null;
  });
  nav.addEventListener("pointercancel", () => (navPress = null));
  let last = performance.now(),
    lastNav = "";
  function tick(now) {
    requestAnimationFrame(tick);
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    onFrame(dt);
    controls.update();
    refreshSelection();
    renderer.render(scene, camera);
    const navKey = camera.quaternion.toArray().join(",") + state.navOpacity;
    if (navKey !== lastNav) {
      drawNavigation();
      lastNav = navKey;
    }
  }
  rebuild();
  resize();
  requestAnimationFrame(tick);
  return {
    rebuild,
    update,
    frame,
    orient,
    projectMode,
    saveView: () => ({
      position: camera.position.toArray(),
      target: controls.target.toArray(),
      ortho: state.ortho,
    }),
    restoreView(view) {
      if (!view) return;
      state.ortho = view.ortho;
      projectMode(view.ortho);
      camera.position.fromArray(view.position);
      controls.target.fromArray(view.target);
      controls.update();
    },
    capture: () => renderer.domElement.toDataURL("image/png"),
    metrics: () => ({
      triangles: renderer.info.render.triangles,
      calls: renderer.info.render.calls,
    }),
    cancelTransform() {
      transform.reset();
    },
    setColor(id, color) {
      objects.get(id)?.traverse((o) => {
        if (o.isMesh) {
          o.material.color.set(color);
          o.material.roughness =
            state.entities.find((e) => e.id === id)?.roughness ?? 0.5;
        }
      });
    },
  };
}
