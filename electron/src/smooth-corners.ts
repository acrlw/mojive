/// <reference types="vite/client" />
import {
  cornerBudgets,
  normalizeRadii,
  derivedCorner,
  maximumCurvature,
  shiftedCorner,
  surfacePath,
  smoothCapsule,
  capsuleEndPadding,
  offsetCapsule,
  type SurfaceGeometry,
  type Capsule,
  type SurfaceCorners,
} from './smooth-geometry';
import './corners.css';

let smoothing = 0.6;
let capsuleAmount = 0.382;
const owners = new Set<HTMLElement>();
const pending = new Set<HTMLElement>();
const written = new WeakMap<HTMLElement, string | null>();
let frame = 0;
const radii = ['TopLeft', 'TopRight', 'BottomRight', 'BottomLeft'] as const;
const geometryRoots = new Set<Element>();
const geometry = new Map<HTMLElement, SurfaceGeometry>();
type Surface = {
  element: HTMLElement;
  width: number;
  height: number;
  radius: number[];
  budgets: number[];
  bounds: DOMRect;
  scale: number;
  indices: number[];
  strokeInset: number;
  capsule: boolean;
};

function measureGeometry(batch: HTMLElement[]) {
  const surfaces: Surface[] = [];
  const byElement = new Map<HTMLElement, Surface>();
  const nodes: { surface: Surface; corner: number; parent: number }[] = [];
  const roots = new Set<Element>();
  for (const changed of geometryRoots) {
    if (!changed.isConnected) continue;
    let root = changed;
    // Corner constraints join ancestors and descendants. Rebuild that complete
    // rounded family without measuring unrelated panels on every slider update.
    for (let ancestor = changed.parentElement; ancestor; ancestor = ancestor.parentElement)
      if (owners.has(ancestor)) root = ancestor;
    roots.add(root);
  }
  geometryRoots.clear();
  const rootList = [...roots];
  const scopes = rootList.filter(
    (root) => !rootList.some((other) => other !== root && other.contains(root)),
  );
  const candidates = [...new Set([...owners, ...batch])].filter((element) =>
    scopes.some((root) => root.contains(element)),
  );
  const markers = candidates.map((element) => element.getAttribute('data-smooth-surface'));
  for (const element of candidates) element.removeAttribute('data-smooth-surface');
  for (const element of candidates) {
    if (!element.isConnected || !element.hasAttribute('data-capsule-layout')) continue;
    // Intrinsic chip width includes its end padding. Resolve that layout before
    // measuring child bounds or building the shared surface geometry.
    for (let pass = 0; pass < 8; pass++) {
      const style = getComputedStyle(element);
      const width = parseFloat(style.width),
        height = parseFloat(style.height);
      if (!(width > 0 && height > 0)) break;
      const center = capsuleEndPadding(smoothCapsule(width, height, capsuleAmount));
      const previous = parseFloat(element.style.getPropertyValue('--capsule-end'));
      if (Math.abs(center - previous) < 0.0001) break;
      element.style.setProperty('--capsule-end', `${center}px`);
    }
    written.set(element, element.getAttribute('style'));
  }
  for (const element of candidates) {
    if (!element.isConnected) {
      geometry.delete(element);
      continue;
    }
    const style = getComputedStyle(element);
    const width =
      parseFloat(style.width) +
      (style.boxSizing === 'border-box'
        ? 0
        : parseFloat(style.paddingLeft) +
          parseFloat(style.paddingRight) +
          parseFloat(style.borderLeftWidth) +
          parseFloat(style.borderRightWidth));
    const height =
      parseFloat(style.height) +
      (style.boxSizing === 'border-box'
        ? 0
        : parseFloat(style.paddingTop) +
          parseFloat(style.paddingBottom) +
          parseFloat(style.borderTopWidth) +
          parseFloat(style.borderBottomWidth));
    const raw = radii.map((corner) => style[`border${corner}Radius`]);
    if (
      !width ||
      !height ||
      raw.every((value) => parseFloat(value) === 0) ||
      raw.some((value) => value.includes('%')) ||
      element.hasAttribute('data-round-surface')
    ) {
      geometry.delete(element);
      delete element.dataset.smoothGeometry;
      continue;
    }
    element.dataset.smoothGeometry = '';
    const radius = normalizeRadii(width, height, raw.map(parseFloat));
    const bounds = element.getBoundingClientRect();
    if (!bounds.width || !bounds.height) {
      geometry.delete(element);
      continue;
    }
    const surface: Surface = {
      element,
      width,
      height,
      radius,
      budgets: cornerBudgets(width, height, radius),
      bounds,
      scale: bounds.width / width,
      indices: [],
      strokeInset: Math.max(0, ...paintEdges(style).map((edge) => edge.inset)),
      capsule: radius.every((r) => Math.abs(r - Math.min(width, height) / 2) < 1e-6),
    };
    for (let corner = 0; !surface.capsule && corner < 4; corner++) {
      surface.indices.push(nodes.length);
      nodes.push({ surface, corner, parent: nodes.length });
    }
    surfaces.push(surface);
    byElement.set(element, surface);
    if (!owners.has(element)) {
      owners.add(element);
      resize.observe(element);
    }
  }
  candidates.forEach((element, i) => {
    if (markers[i] !== null) element.setAttribute('data-smooth-surface', markers[i]!);
  });
  const find = (index: number): number =>
    nodes[index].parent === index ? index : (nodes[index].parent = find(nodes[index].parent));
  const join = (a: number, b: number) => {
    nodes[find(a)].parent = find(b);
  };
  const pairs = [
    [0, 1],
    [2, 1],
    [2, 3],
    [0, 3],
  ];
  for (const child of surfaces) {
    if (child.capsule) continue;
    const matched = new Set<number>();
    for (let ancestor = child.element.parentElement; ancestor; ancestor = ancestor.parentElement) {
      const parent = byElement.get(ancestor);
      if (!parent || parent.capsule) continue;
      const a = parent.bounds,
        b = child.bounds,
        scale = child.scale;
      const gaps = [
        (b.left - a.left) / scale,
        (b.top - a.top) / scale,
        (a.right - b.right) / scale,
        (a.bottom - b.bottom) / scale,
      ];
      for (let i = 0; i < 4; i++) {
        const [x, y] = pairs[i];
        if (
          matched.has(i) ||
          child.radius[i] === 0 ||
          gaps[x] < 0 ||
          Math.abs(gaps[x] - gaps[y]) > 0.15 ||
          Math.abs(parent.radius[i] - child.radius[i] - gaps[x]) > 0.15
        )
          continue;
        join(child.indices[i], parent.indices[i]);
        matched.add(i);
      }
    }
  }
  // Equal radii on one surface share a contour even when only some corners
  // touch an ancestor or child. Different radii retain independent constraints.
  for (const surface of surfaces.filter((s) => !s.capsule))
    for (let i = 0; i < 4; i++)
      for (let j = i + 1; j < 4; j++)
        if (surface.radius[i] > 0 && surface.radius[i] === surface.radius[j])
          join(surface.indices[i], surface.indices[j]);
  const groups = new Map<number, number[]>();
  nodes.forEach((_, i) => {
    const root = find(i);
    const group = groups.get(root) ?? [];
    group.push(i);
    groups.set(root, group);
  });
  const resolved = new Map<number, ReturnType<typeof derivedCorner>>();
  for (const group of groups.values()) {
    const tolerance = Math.min(0.01, ...group.map((i) => 0.01 / nodes[i].surface.scale));
    const radii = [...new Set(group.map((i) => nodes[i].surface.radius[nodes[i].corner]))].sort(
      (a, b) => b - a,
    );
    const extra = Math.max(
      0,
      Math.min(
        ...group.map(
          (i) =>
            nodes[i].surface.budgets[nodes[i].corner] - nodes[i].surface.radius[nodes[i].corner],
        ),
      ),
    );
    const minimum = Math.min(
      ...group.map((i) => nodes[i].surface.radius[nodes[i].corner] - nodes[i].surface.strokeInset),
    );
    // Preserve the outer reference whenever every nested contour (including stroke
    // centerlines) is regular. A deeper reference keeps over-deep chains well-defined.
    const referenceRadius = radii.find((radius) => {
      if (radius === 0) return true;
      const reference = derivedCorner(radius, smoothing, radius + extra, 0, tolerance);
      const peak = Math.max(1 / radius, ...reference.leading.map(maximumCurvature));
      return 1 + (minimum - radius) * peak > 1e-5;
    });
    if (referenceRadius === undefined) throw new Error('No regular nested corner reference');
    const budget = referenceRadius + extra;
    for (const i of group) {
      const { surface, corner } = nodes[i];
      resolved.set(
        i,
        derivedCorner(
          referenceRadius,
          smoothing,
          budget,
          surface.radius[corner] - referenceRadius,
          tolerance,
        ),
      );
    }
  }
  const capsules = surfaces.filter((surface) => surface.capsule);
  const capsuleParents = new Map(capsules.map((surface) => [surface, surface]));
  const capsuleRoot = (surface: Surface): Surface => {
    const parent = capsuleParents.get(surface)!;
    if (parent === surface) return surface;
    const root = capsuleRoot(parent);
    capsuleParents.set(surface, root);
    return root;
  };
  for (const child of capsules) {
    for (let ancestor = child.element.parentElement; ancestor; ancestor = ancestor.parentElement) {
      const parent = byElement.get(ancestor);
      if (!parent?.capsule) continue;
      const a = parent.bounds,
        b = child.bounds;
      const gaps = [b.left - a.left, b.top - a.top, a.right - b.right, a.bottom - b.bottom].map(
        (v) => v / child.scale,
      );
      if (gaps[0] >= 0 && Math.max(...gaps) - Math.min(...gaps) < 0.15) {
        capsuleParents.set(capsuleRoot(child), capsuleRoot(parent));
        break;
      }
    }
  }
  const capsuleGroups = new Map<Surface, Surface[]>();
  for (const surface of capsules) {
    const root = capsuleRoot(surface),
      group = capsuleGroups.get(root) ?? [];
    group.push(surface);
    capsuleGroups.set(root, group);
  }
  const capsuleGeometry = new Map<Surface, Capsule>();
  for (const group of capsuleGroups.values()) {
    group.sort((a, b) => b.radius[0] - a.radius[0]);
    const tolerance = Math.min(0.01, ...group.map((s) => 0.01 / s.scale));
    const minimum = Math.min(...group.map((s) => s.radius[0] - s.strokeInset));
    const reference = group
      .map((s) => smoothCapsule(s.width, s.height, capsuleAmount, tolerance))
      .find((c) => c.minimumRadius + minimum - Math.min(c.width, c.height) / 2 > 1e-5);
    if (!reference) throw new Error('No regular nested capsule reference');
    for (const surface of group)
      capsuleGeometry.set(
        surface,
        offsetCapsule(
          reference,
          surface.radius[0] - Math.min(reference.width, reference.height) / 2,
          tolerance,
        ),
      );
  }
  const key = (shape: SurfaceGeometry) =>
    Array.isArray(shape) ? shape.map((c) => c.key).join('|') : shape.key;
  for (const surface of surfaces) {
    const shape = surface.capsule
      ? capsuleGeometry.get(surface)!
      : (surface.indices.map((i) => resolved.get(i)!) as SurfaceCorners);
    const previous = geometry.get(surface.element);
    if (!previous || key(shape) !== key(previous)) batch.push(surface.element);
    geometry.set(surface.element, shape);
  }
}

export function setCornerSmoothing(value: number, capsuleValue = capsuleAmount) {
  const rectangle = Math.max(0, Math.min(1, value));
  const capsule = Math.max(0, Math.min(1, capsuleValue));
  if (smoothing === rectangle && capsuleAmount === capsule) return;
  smoothing = rectangle;
  capsuleAmount = capsule;
  geometryRoots.add(document.documentElement);
  for (const element of owners) pending.add(element);
  schedule();
}
export function cornerSmoothing() {
  return smoothing;
}
export function capsuleSmoothing() {
  return capsuleAmount;
}

const escape = (value: string) =>
  value.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;');
function shadows(value: string) {
  return value.split(/,(?![^()]*\))/).map((part) => {
    const color = part.match(/(?:rgba?|color)\([^)]*\)/)?.[0] ?? 'transparent';
    const lengths =
      part
        .replace(color, '')
        .match(/-?[\d.]+px/g)
        ?.map(parseFloat) ?? [];
    return { color, lengths, inset: part.includes('inset') };
  });
}
type SurfaceEdge = {
  kind: 'border' | 'inset' | 'outline';
  width: number;
  color: string;
  inset: number;
};
function paintEdges(style: CSSStyleDeclaration): SurfaceEdge[] {
  const edges: SurfaceEdge[] = [];
  for (const shadow of shadows(style.boxShadow)) {
    const [x, y, blur, spread = 0] = shadow.lengths;
    if (shadow.inset && x === 0 && y === 0 && blur === 0 && spread > 0)
      edges.push({ kind: 'inset', width: spread, color: shadow.color, inset: spread / 2 });
  }
  const border = parseFloat(style.borderTopWidth);
  if (border > 0)
    edges.push({ kind: 'border', width: border, color: style.borderTopColor, inset: border / 2 });
  const outline = parseFloat(style.outlineWidth);
  if (outline > 0 && style.outlineStyle !== 'none')
    edges.push({
      kind: 'outline',
      width: outline,
      color: style.outlineColor,
      inset: -parseFloat(style.outlineOffset) - outline / 2,
    });
  return edges;
}
function paint(element: HTMLElement) {
  if (!element.isConnected) {
    owners.delete(element);
    resize.unobserve(element);
    return;
  }
  const previous = element.getAttribute('data-smooth-surface');
  element.removeAttribute('data-smooth-surface');
  element.removeAttribute('data-smooth-position');
  const style = getComputedStyle(element);
  const width =
    parseFloat(style.width) +
    (style.boxSizing === 'border-box'
      ? 0
      : parseFloat(style.paddingLeft) +
        parseFloat(style.paddingRight) +
        parseFloat(style.borderLeftWidth) +
        parseFloat(style.borderRightWidth));
  const height =
    parseFloat(style.height) +
    (style.boxSizing === 'border-box'
      ? 0
      : parseFloat(style.paddingTop) +
        parseFloat(style.paddingBottom) +
        parseFloat(style.borderTopWidth) +
        parseFloat(style.borderBottomWidth));
  const raw = radii.map((corner) => style[`border${corner}Radius`]);
  if (
    !width ||
    !height ||
    raw.every((radius) => parseFloat(radius) === 0) ||
    raw.some((radius) => radius.includes('%')) ||
    element.hasAttribute('data-round-surface')
  ) {
    delete element.dataset.smoothRadii;
    delete element.dataset.smoothAmount;
    delete element.dataset.smoothPaint;
    return;
  }
  if (!owners.has(element)) {
    owners.add(element);
    resize.observe(element);
  }
  const radius = normalizeRadii(width, height, raw.map(parseFloat));
  const corners = geometry.get(element)!;
  if (!corners) return;
  const scale = element.getBoundingClientRect().width / width;
  const localSmoothing = Array.isArray(corners) ? smoothing : capsuleAmount;
  const makePath = (inset: number) =>
    surfacePath(
      width - 2 * inset,
      height - 2 * inset,
      Array.isArray(corners)
        ? (corners.map((corner) =>
            shiftedCorner(corner, -inset, Math.min(0.01, 0.01 / scale)),
          ) as SurfaceCorners)
        : offsetCapsule(corners, -inset, Math.min(0.01, 0.01 / scale)),
    );
  const sourceStyle = {
    backgroundColor: style.backgroundColor,
    backgroundImage: style.backgroundImage,
    borderRadius: style.borderRadius,
    boxShadow: style.boxShadow,
    outlineStyle: style.outlineStyle,
    outlineWidth: style.outlineWidth,
    outlineColor: style.outlineColor,
    outlineOffset: style.outlineOffset,
    ...Object.fromEntries(radii.map((corner, index) => [`border${corner}Radius`, raw[index]])),
  };
  const background = style.backgroundColor;
  const checker = style.backgroundImage.startsWith('repeating-conic-gradient')
    ? [...new Set(style.backgroundImage.match(/rgba?\([^)]*\)/g) ?? [])]
    : null;
  const boxShadows = shadows(style.boxShadow);
  const edges = paintEdges(style);
  const solid = background !== 'rgba(0, 0, 0, 0)' && background !== 'transparent';
  if (!solid && !checker && !edges.length) {
    if (previous) element.style.removeProperty('--smooth-paint');
    return;
  }
  const defs = checker
    ? `<defs><pattern id="checker" width="8" height="8" patternUnits="userSpaceOnUse"><path fill="${escape(checker[0])}" d="M0 0h8v8H0z"/><path fill="${escape(checker[1])}" d="M0 0h4v4H0zM4 4h4v4H4z"/></pattern></defs>`
    : '';
  const path = makePath(0);
  const rings = boxShadows.filter(
    ({ inset, lengths: [x, y, blur, spread] }) =>
      !inset && x === 0 && y === 0 && blur === 0 && spread > 0,
  );
  const outset = Math.max(
    0,
    ...rings.map(({ lengths }) => lengths[3]),
    ...edges.map((edge) => edge.width / 2 - edge.inset),
  );
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width + 2 * outset}" height="${height + 2 * outset}" viewBox="${-outset} ${-outset} ${width + 2 * outset} ${height + 2 * outset}">${defs}${rings.map(({ color, lengths }) => `<path d="${path}" fill="none" stroke="${escape(color)}" stroke-width="${lengths[3] * 2}"/>`).join('')}<path d="${path}" fill="${checker ? 'url(#checker)' : escape(background)}"/>${edges.map((edge) => `<path data-edge="${edge.kind}" transform="translate(${edge.inset} ${edge.inset})" d="${makePath(edge.inset)}" fill="none" stroke="${escape(edge.color)}" stroke-width="${edge.width}"/>`).join('')}</svg>`;
  const dropShadows = boxShadows
    .filter((shadow) => !shadow.inset && shadow.lengths.length >= 3 && shadow.lengths[2] > 0)
    .map(({ lengths: [x, y, blur], color }) => `drop-shadow(${x}px ${y}px ${blur / 2}px ${color})`)
    .join(' ');
  const image = `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
  const native = element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement;
  const media =
    element instanceof HTMLCanvasElement ||
    element instanceof HTMLImageElement ||
    element instanceof HTMLVideoElement;
  const layer = getComputedStyle(element, '::after').content === 'none' ? 'after' : 'before';
  element.style.setProperty('--smooth-paint', image);
  element.style.setProperty(
    '--smooth-shadow',
    [style.filter === 'none' ? '' : style.filter, dropShadows].filter(Boolean).join(' ') || 'none',
  );
  element.style.setProperty(
    '--smooth-inset',
    `${-parseFloat(style.borderTopWidth) - outset}px ${-parseFloat(style.borderRightWidth) - outset}px ${-parseFloat(style.borderBottomWidth) - outset}px ${-parseFloat(style.borderLeftWidth) - outset}px`,
  );
  if (media) {
    // Replaced elements paint their own pixels and cannot host a pseudo-element.
    element.style.setProperty('--smooth-clip', `path("${path}")`);
    element.style.setProperty('--smooth-media-background', background);
  }
  element.dataset.smoothPaint = JSON.stringify(sourceStyle);
  element.dataset.smoothRadii = JSON.stringify(radius);
  element.dataset.smoothAmount = String(localSmoothing);
  element.dataset.smoothKind = Array.isArray(corners) ? 'rectangle' : 'capsule';
  element.dataset.smoothReferences = JSON.stringify(
    Array.isArray(corners)
      ? corners.map((corner) => ({
          radius: (corner.reference ?? corner).radius,
          offset: corner.offset,
          error: corner.sampledError,
        }))
      : Array.from({ length: 4 }, () => ({
          radius:
            Math.min((corners.reference ?? corners).width, (corners.reference ?? corners).height) /
            2,
          offset: corners.offset,
          smoothing: corners.effectiveSmoothing,
        })),
  );
  if (style.position === 'static') element.dataset.smoothPosition = 'static';
  element.dataset.smoothSurface = media ? 'media' : native ? 'native' : layer;
  written.set(element, element.getAttribute('style'));
}
function flush() {
  frame = 0;
  const batch = [...pending];
  pending.clear();
  if (geometryRoots.size) measureGeometry(batch);
  const unique = [...new Set(batch)];
  unique.sort((a, b) => (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1));
  for (const element of unique) paint(element);
}
function schedule() {
  if (!frame) frame = requestAnimationFrame(flush);
}
function scan(root: Element, geometryChange = true) {
  if (geometryChange) geometryRoots.add(root);
  if (root instanceof HTMLElement) pending.add(root);
  for (const element of root.querySelectorAll<HTMLElement>('*'))
    if (element instanceof HTMLElement) pending.add(element);
  schedule();
}
const resize = new ResizeObserver((entries) => {
  for (const entry of entries) {
    geometryRoots.add(entry.target);
    pending.add(entry.target as HTMLElement);
  }
  schedule();
});
const previousInlineStyle = document.createElement('div').style;
const geometricProperties =
  /^(?:--|width$|height$|min-|max-|padding|margin|border.*(?:radius|width)$|font|line-height$|gap$|row-gap$|column-gap$|grid|flex|display$|box-sizing$|zoom$)/;
function styleChangesGeometry(element: HTMLElement, previous: string | null) {
  previousInlineStyle.cssText = previous ?? '';
  const names = new Set([...Array.from(previousInlineStyle), ...Array.from(element.style)]);
  const rounded =
    element.hasAttribute('data-smooth-geometry') ||
    !!element.querySelector('[data-smooth-geometry]');
  const ownsGeometry = element.hasAttribute('data-smooth-geometry');
  const roundedAncestor = !!element.parentElement?.closest('[data-smooth-geometry]');
  for (const name of names) {
    if (previousInlineStyle.getPropertyValue(name) === element.style.getPropertyValue(name))
      continue;
    if (/^border.*radius$/.test(name)) return true;
    // ResizeObserver covers dimensions of standalone descendant surfaces. Moving
    // or resizing their unrounded wrapper cannot change a nested corner relation.
    // Inherited shape/font variables and layout inside a rounded ancestor can.
    if (
      (/^--|^font|^line-height$/.test(name) && rounded) ||
      (geometricProperties.test(name) && (ownsGeometry || (rounded && roundedAncestor))) ||
      (rounded && roundedAncestor && /^(?:position|top|left|right|bottom|transform)$/.test(name))
    )
      return true;
  }
  return false;
}
const mutations = new MutationObserver((records) => {
  for (const record of records) {
    if (record.type === 'childList') {
      if ([...record.removedNodes].some((node) => node instanceof Element)) {
        if (record.target instanceof Element) geometryRoots.add(record.target);
        schedule();
      }
      for (const node of record.addedNodes) if (node instanceof Element) scan(node);
    } else if (
      record.target instanceof HTMLElement &&
      (record.attributeName !== 'style' ||
        record.target.getAttribute('style') !== written.get(record.target))
    )
      scan(
        record.target,
        record.attributeName === 'style'
          ? styleChangesGeometry(record.target, record.oldValue)
          : ['class', 'open', 'aria-expanded'].includes(record.attributeName!),
      );
  }
  for (const owner of owners)
    if (!owner.isConnected) {
      owners.delete(owner);
      geometry.delete(owner);
      resize.unobserve(owner);
    }
});
const held = new Set<HTMLElement>();
const heldKeys = new Map<string, Set<HTMLElement>>();
const interaction = (event: Event) => {
  if (event.type === 'blur' && event.target !== window) return;
  if ((event.type === 'focusin' || event.type === 'focusout') && event.target instanceof Element)
    geometryRoots.add(event.target);
  const affected = new Set<HTMLElement>();
  for (const element of event.composedPath()) {
    if (!(element instanceof HTMLElement)) continue;
    affected.add(element);
    // Track colors can inherit hover/press variables from a compound control.
    if (element.matches('button, input, textarea, [role="slider"], .number-field'))
      for (const child of element.querySelectorAll<HTMLElement>('[data-smooth-surface]'))
        affected.add(child);
  }
  if (event.type === 'pointerdown') for (const element of affected) held.add(element);
  if (event.type === 'keydown' && event instanceof KeyboardEvent) {
    const key = event.code || event.key;
    const targets = heldKeys.get(key) ?? new Set<HTMLElement>();
    for (const element of affected) targets.add(element);
    heldKeys.set(key, targets);
  }
  if (['pointerup', 'pointercancel', 'blur'].includes(event.type)) {
    for (const element of held) affected.add(element);
    held.clear();
  }
  if (event.type === 'keyup' && event instanceof KeyboardEvent) {
    const key = event.code || event.key;
    for (const element of heldKeys.get(key) ?? []) affected.add(element);
    heldKeys.delete(key);
  }
  if (event.type === 'blur') {
    for (const targets of heldKeys.values()) for (const element of targets) affected.add(element);
    heldKeys.clear();
  }
  for (const element of affected) pending.add(element);
  schedule();
};
mutations.observe(document.documentElement, {
  childList: true,
  subtree: true,
  attributes: true,
  attributeOldValue: true,
  attributeFilter: ['class', 'style', 'open', 'aria-checked', 'aria-expanded', 'disabled'],
});
for (const event of [
  'pointerover',
  'pointerout',
  'pointerdown',
  'pointerup',
  'pointercancel',
  'keydown',
  'keyup',
  'blur',
  'focusin',
  'focusout',
  'toggle',
])
  document.addEventListener(event, interaction, true);
window.addEventListener('blur', interaction);
scan(document.documentElement);
if (import.meta.hot)
  import.meta.hot.dispose(() => {
    mutations.disconnect();
    resize.disconnect();
    cancelAnimationFrame(frame);
    window.removeEventListener('blur', interaction);
    for (const event of [
      'pointerover',
      'pointerout',
      'pointerdown',
      'pointerup',
      'pointercancel',
      'keydown',
      'keyup',
      'blur',
      'focusin',
      'focusout',
      'toggle',
    ])
      document.removeEventListener(event, interaction, true);
  });
