/** Read the SVG surface paint together with layout properties from its DOM owner. */
export function installPaintStyle() {
  window.paintStyle = (element, pseudo) => {
    const style = getComputedStyle(element, pseudo);
    const paint =
      !pseudo && element.hasAttribute('data-smooth-surface')
        ? JSON.parse(element.dataset.smoothPaint)
        : {};
    return new Proxy(style, {
      get(target, key) {
        if (key in paint) return paint[key];
        const value = target[key];
        return typeof value === 'function' ? value.bind(target) : value;
      },
    });
  };
}

/** Let state and the shared SVG paint batch reach the next presented frame. */
export function settlePaint(page) {
  return page.evaluate(
    () => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))),
  );
}
