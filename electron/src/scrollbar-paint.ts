import { smoothCapsule, surfacePath } from './smooth-geometry';

const previous = new WeakMap<HTMLElement, string>();
const roles = ['field-active', 'primary-dim', 'primary'] as const;

/** Paint fixed-size smooth end caps while Chromium owns thumb length, scrolling and capture. */
export function applyScrollbarPaint(root: HTMLElement, smoothing: number) {
  const style = getComputedStyle(root);
  const colors = roles.map((role) => style.getPropertyValue(`--ui-${role}`).trim());
  const key = JSON.stringify([smoothing, ...colors]);
  if (previous.get(root) === key) return;
  previous.set(root, key);
  const shape = smoothCapsule(3, 32, smoothing, 0.002);
  const contour = surfacePath(shape.width, shape.height, shape);
  const cap = Math.ceil(shape.depth);
  // The extra straight pixel overlaps the center strip, avoiding seams at fractional zoom.
  const extent = cap + 1;
  const image = (color: string, end: boolean, horizontal: boolean) => {
    const width = horizontal ? extent : 3;
    const height = horizontal ? 3 : extent;
    const shift = end ? 32 - extent : 0;
    const transform = horizontal ? 'matrix(0 1 1 0 0 0)' : '';
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="${horizontal ? shift : 0} ${horizontal ? 0 : shift} ${width} ${height}"><path d="${contour}" transform="${transform}" fill="${color}"/></svg>`;
    return `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
  };
  root.style.setProperty('--scrollbar-cap', `${cap}px`);
  root.style.setProperty('--scrollbar-cap-extent', `${extent}px`);
  colors.forEach((color, index) => {
    for (const axis of ['x', 'y'])
      root.style.setProperty(
        `--scrollbar-${roles[index]}-${axis}`,
        `${image(color, false, axis === 'x')}, linear-gradient(${color}, ${color}), ${image(color, true, axis === 'x')}`,
      );
  });
}
