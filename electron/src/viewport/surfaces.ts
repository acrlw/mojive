/** Ordinary UI surfaces; spatial gizmo geometry has a separate native contour owner. */
const canvasPaths = new Map<string, Path2D>();
export function fillSurface(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const key = `${width}:${height}:${radius}`;
  let path = canvasPaths.get(key);
  if (!path) {
    path = new Path2D();
    path.roundRect(0, 0, width, height, Math.max(0, radius));
    if (canvasPaths.size >= 64) canvasPaths.delete(canvasPaths.keys().next().value!);
    canvasPaths.set(key, path);
  }
  ctx.save();
  ctx.translate(x, y);
  ctx.fill(path);
  ctx.restore();
}

/** Compatibility entry point: CSS owns radii without masks or size observers. */
export function observeSurfaces(elements: readonly HTMLElement[]) {
  for (const element of elements) {
    element.style.removeProperty('clip-path');
    element.style.removeProperty('corner-shape');
  }
  return () => {};
}
