/** Native scrollbar visibility policy; the browser retains every scroll and capture operation. */
export const SCROLLBAR_HIDE_DELAY = 1000;

type Axis = 'x' | 'y';
type State = { owner: HTMLElement; timer: number; held: boolean };

function scrolls(element: HTMLElement, axis: Axis, style = getComputedStyle(element)) {
  return (
    /^(auto|scroll)$/.test(axis === 'x' ? style.overflowX : style.overflowY) &&
    (axis === 'x'
      ? element.scrollWidth > element.clientWidth
      : element.scrollHeight > element.clientHeight)
  );
}

/** Resolve the real native gutter, including borders, both axes and CSS zoom. */
function trackAt(element: HTMLElement, x: number, y: number): Axis | null {
  const rect = element.getBoundingClientRect();
  if (
    !element.offsetWidth ||
    !element.offsetHeight ||
    x < rect.left ||
    x >= rect.right ||
    y < rect.top ||
    y >= rect.bottom
  )
    return null;
  const style = getComputedStyle(element),
    sx = rect.width / element.offsetWidth,
    sy = rect.height / element.offsetHeight;
  const left = parseFloat(style.borderLeftWidth),
    right = parseFloat(style.borderRightWidth);
  const top = parseFloat(style.borderTopWidth),
    bottom = parseFloat(style.borderBottomWidth);
  const vertical = element.offsetWidth - element.clientWidth - left - right;
  const horizontal = element.offsetHeight - element.clientHeight - top - bottom;
  const contentLeft = rect.left + element.clientLeft * sx;
  const contentRight = contentLeft + element.clientWidth * sx;
  const contentBottom = rect.top + (element.clientTop + element.clientHeight) * sy;
  if (
    vertical > 0 &&
    scrolls(element, 'y', style) &&
    y < contentBottom &&
    (style.direction === 'rtl'
      ? x >= rect.left + left * sx && x < contentLeft
      : x >= contentRight && x < rect.right - right * sx)
  )
    return 'y';
  if (
    horizontal > 0 &&
    scrolls(element, 'x', style) &&
    x >= contentLeft &&
    x < contentRight &&
    y >= contentBottom &&
    y < rect.bottom - bottom * sy
  )
    return 'x';
  return null;
}

/** Delegation covers mounted, virtualized and top-layer popup scroll owners without a registry scan. */
export function installScrollbarBehavior(doc: Document = document) {
  const win = doc.defaultView!;
  const states = new Map<HTMLElement, State>();
  let hovered: HTMLElement | null = null,
    held: State | null = null;
  let heldPointer: number | null = null;
  let pointerX = NaN,
    pointerY = NaN;
  const hide = (state: State) => {
    win.clearTimeout(state.timer);
    delete state.owner.dataset.scrollbarVisible;
    if (!state.held) states.delete(state.owner);
  };
  const reveal = (owner: HTMLElement) => {
    let state = states.get(owner);
    if (!state) states.set(owner, (state = { owner, timer: 0, held: false }));
    win.clearTimeout(state.timer);
    owner.dataset.scrollbarVisible = '';
    if (!state.held) state.timer = win.setTimeout(() => hide(state), SCROLLBAR_HIDE_DELAY);
    return state;
  };
  const clearHover = () => {
    if (hovered) delete hovered.dataset.scrollbarHover;
    hovered = null;
  };
  const release = () => {
    if (!held) return;
    const state = held;
    held = null;
    heldPointer = null;
    state.held = false;
    delete state.owner.dataset.scrollbarPressed;
    if (state.owner.isConnected) reveal(state.owner);
    else hide(state);
  };
  const track = (event: PointerEvent) => {
    for (const target of event.composedPath()) {
      if (!(target instanceof HTMLElement)) continue;
      const axis = trackAt(target, event.clientX, event.clientY);
      if (axis) return { owner: target, axis };
    }
    return null;
  };
  const move = (event: PointerEvent) => {
    // Native scrollbar capture suppresses move events; never replace that capture.
    if (held) {
      if (event.pointerId !== heldPointer) return;
      if (!event.buttons) release();
      else return;
    }
    if (event.clientX === pointerX && event.clientY === pointerY) return;
    pointerX = event.clientX;
    pointerY = event.clientY;
    const hit = event.buttons ? null : track(event);
    if (hovered !== hit?.owner) clearHover();
    if (hit) {
      hovered = hit.owner;
      hovered.dataset.scrollbarHover = hit.axis;
      reveal(hovered);
    }
  };
  const down = (event: PointerEvent) => {
    if (held && event.pointerId !== heldPointer) return;
    clearHover();
    if (event.button !== 0) return;
    const hit = track(event);
    if (!hit) return;
    release();
    held = reveal(hit.owner);
    heldPointer = event.pointerId;
    held.held = true;
    win.clearTimeout(held.timer);
    hit.owner.dataset.scrollbarPressed = hit.axis;
  };
  const up = (event: PointerEvent) => {
    if (held && event.pointerId !== heldPointer) return;
    release();
    clearHover();
    const hit = !event.buttons && track(event);
    if (hit) {
      hovered = hit.owner;
      hovered.dataset.scrollbarHover = hit.axis;
    }
  };
  const cancel = (event: PointerEvent) => {
    if (held && event.pointerId !== heldPointer) return;
    release();
    clearHover();
  };
  const wheel = (event: WheelEvent) => {
    const axis = Math.abs(event.deltaX) > Math.abs(event.deltaY) || event.shiftKey ? 'x' : 'y';
    for (const target of event.composedPath()) {
      if (!(target instanceof HTMLElement) || !scrolls(target, axis)) continue;
      reveal(target);
      return;
    }
  };
  const scroll = (event: Event) => {
    const owner = event.target;
    if (owner instanceof HTMLElement && (scrolls(owner, 'x') || scrolls(owner, 'y'))) reveal(owner);
  };
  const reset = () => {
    release();
    clearHover();
    for (const state of states.values()) hide(state);
    pointerX = pointerY = NaN;
  };
  const leave = (event: PointerEvent) => {
    if (!event.relatedTarget) clearHover();
  };
  const observer = new MutationObserver(() => {
    const hidden = (owner: HTMLElement) =>
      !owner.isConnected ||
      !owner.getClientRects().length ||
      getComputedStyle(owner).visibility !== 'visible' ||
      !!owner.closest('[inert]');
    if (held && hidden(held.owner)) release();
    if (hovered && hidden(hovered)) clearHover();
    for (const state of states.values()) if (hidden(state.owner)) hide(state);
  });
  observer.observe(doc, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: ['hidden', 'style', 'open', 'inert'],
  });
  doc.addEventListener('pointermove', move, true);
  doc.addEventListener('pointerdown', down, true);
  doc.addEventListener('pointerup', up, true);
  doc.addEventListener('pointercancel', cancel, true);
  doc.addEventListener('pointerout', leave, true);
  doc.addEventListener('wheel', wheel, { capture: true, passive: true });
  doc.addEventListener('scroll', scroll, true);
  doc.addEventListener('visibilitychange', reset);
  win.addEventListener('blur', reset);
  return () => {
    reset();
    observer.disconnect();
    doc.removeEventListener('pointermove', move, true);
    doc.removeEventListener('pointerdown', down, true);
    doc.removeEventListener('pointerup', up, true);
    doc.removeEventListener('pointercancel', cancel, true);
    doc.removeEventListener('pointerout', leave, true);
    doc.removeEventListener('wheel', wheel, true);
    doc.removeEventListener('scroll', scroll, true);
    doc.removeEventListener('visibilitychange', reset);
    win.removeEventListener('blur', reset);
  };
}
