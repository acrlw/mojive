// Independent intersections against the serialized C/A path. SVG getPointAtLength
// flattens arcs, which otherwise adds about 0.00027 * radius to this measurement.
export function parseCorner(d) {
  const tokens = d.match(/[A-Z]|[-+]?\d*\.?\d+(?:e[-+]?\d+)?/gi) ?? [];
  const segments = [];
  let current;
  while (tokens.length) {
    const command = tokens.shift();
    if (command === 'M') {
      current = { x: Number(tokens.shift()), y: Number(tokens.shift()) };
      continue;
    }
    if (command === 'C') {
      const v = tokens.splice(0, 6).map(Number),
        end = { x: v[4], y: v[5] };
      segments.push({
        type: 'cubic',
        points: [current, { x: v[0], y: v[1] }, { x: v[2], y: v[3] }, end],
      });
      current = end;
      continue;
    }
    if (command === 'A') {
      const v = tokens.splice(0, 7).map(Number),
        end = { x: v[5], y: v[6] },
        dx = end.x - current.x,
        dy = end.y - current.y,
        chord = Math.hypot(dx, dy),
        height = Math.sqrt(Math.max(0, v[0] ** 2 - chord ** 2 / 4));
      segments.push({
        type: 'arc',
        start: current,
        end,
        radius: v[0],
        center: {
          x: (current.x + end.x) / 2 - (dy / chord) * height,
          y: (current.y + end.y) / 2 + (dx / chord) * height,
        },
      });
      current = end;
      continue;
    }
    throw new Error(`Unsupported corner command ${command}`);
  }
  return segments;
}
function cubicPoint(points, t) {
  const u = 1 - t,
    weights = [u ** 3, 3 * u * u * t, 3 * u * t * t, t ** 3];
  return {
    x: points.reduce((v, p, i) => v + p.x * weights[i], 0),
    y: points.reduce((v, p, i) => v + p.y * weights[i], 0),
  };
}
export function normalGapError(segments, samples, expected) {
  let worst = 0;
  for (const { p, n } of samples) {
    const distances = [];
    for (const segment of segments) {
      if (segment.type === 'arc') {
        const v = { x: p.x - segment.center.x, y: p.y - segment.center.y },
          dot = v.x * n.x + v.y * n.y,
          disc = dot ** 2 - (v.x * v.x + v.y * v.y - segment.radius ** 2);
        if (disc < 0) continue;
        const distance = -dot + Math.sqrt(disc),
          q = { x: p.x + distance * n.x, y: p.y + distance * n.y };
        const angle = (q) => Math.atan2(q.y - segment.center.y, q.x - segment.center.x);
        if (angle(q) >= angle(segment.start) - 1e-9 && angle(q) <= angle(segment.end) + 1e-9)
          distances.push(distance);
      } else {
        const f = (t) => {
          const q = cubicPoint(segment.points, t);
          return (q.x - p.x) * n.y - (q.y - p.y) * n.x;
        };
        let a = 0,
          b = 1,
          fa = f(0),
          fb = f(1);
        if (Math.abs(fa) < 1e-10) b = 0;
        else if (Math.abs(fb) < 1e-10) a = 1;
        else if (fa * fb > 0) continue;
        for (let i = 0; i < 45; i++) {
          const mid = (a + b) / 2,
            value = f(mid);
          if (fa * value <= 0) b = mid;
          else {
            a = mid;
            fa = value;
          }
        }
        const q = cubicPoint(segment.points, (a + b) / 2);
        distances.push((q.x - p.x) * n.x + (q.y - p.y) * n.y);
      }
    }
    if (!distances.length) throw new Error(`No normal intersection at ${JSON.stringify(p)}`);
    worst = Math.max(worst, Math.min(...distances.map((d) => Math.abs(d - expected))));
  }
  return worst;
}
