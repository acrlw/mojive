/** Measured Three scene/compositor/helper workload; UI canvas batches use separate contexts. */
export interface ViewportStats {
  fps: number;
  frameMs: number;
  drawCalls: number;
  triangles: number;
  lines: number;
  points: number;
  backend: 'Three.js · WebGL 2';
  idle: boolean;
}
export class ViewportMetrics {
  private frames: { at: number; duration: number }[] = [];
  private lastAt = -Infinity;
  private lastFrameMs = 0;
  private workload = { drawCalls: 0, triangles: 0, lines: 0, points: 0 };
  record(
    start: number,
    end: number,
    render: { calls: number; triangles: number; lines: number; points: number },
  ) {
    this.lastAt = end;
    this.lastFrameMs = Math.max(0, end - start);
    this.frames.push({ at: end, duration: this.lastFrameMs });
    this.workload = {
      drawCalls: render.calls,
      triangles: render.triangles,
      lines: render.lines,
      points: render.points,
    };
  }
  sample(now: number): ViewportStats {
    while (this.frames.length && this.frames[0].at <= now - 1000) this.frames.shift();
    const idle = now - this.lastAt >= 500;
    return {
      fps: idle ? 0 : this.frames.length,
      frameMs: this.frames.length
        ? this.frames.reduce((total, frame) => total + frame.duration, 0) / this.frames.length
        : this.lastFrameMs,
      ...this.workload,
      backend: 'Three.js · WebGL 2',
      idle,
    };
  }
}
