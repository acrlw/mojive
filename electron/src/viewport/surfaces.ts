import { derivedCorner, smoothCapsule, surfacePath } from '../smooth-geometry';
import { cornerSmoothing, capsuleSmoothing } from '../smooth-corners';

const canvasPaths = new Map<string, Path2D>();
export function fillSurface(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) {
  const smoothing = radius >= Math.min(width, height) / 2 ? capsuleSmoothing() : cornerSmoothing();
  const key = `${width}:${height}:${radius}:${smoothing}`;
  let path = canvasPaths.get(key);
  if (!path) {
    const effectiveRadius = Math.min(radius, width / 2, height / 2);
    const corner = derivedCorner(effectiveRadius, smoothing, Math.min(width, height) / 2, 0, 0.01);
    path = new Path2D(
      surfacePath(
        width,
        height,
        radius >= Math.min(width, height) / 2
          ? smoothCapsule(width, height, smoothing)
          : [corner, corner, corner, corner],
      ),
    );
    if (canvasPaths.size >= 64) canvasPaths.delete(canvasPaths.keys().next().value!);
    canvasPaths.set(key, path);
  }
  ctx.save();
  ctx.translate(x, y);
  ctx.fill(path);
  ctx.restore();
}
