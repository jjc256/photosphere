// Feature pixels are top-left/down; poses and WebGL use right/up/-Z.
// Keep normalized film coordinates in the device camera's axes, including
// the user's quarter-turn correction for a sideways video feed.
export function filmPoint(x, y, shot) {
  const scale = Math.max(shot.w, shot.h);
  const sx = (x - shot.w / 2) / scale;
  const sy = (shot.h / 2 - y) / scale;
  const a = (shot.vidRot || 0) * Math.PI / 2;
  return [Math.cos(a) * sx + Math.sin(a) * sy,
    -Math.sin(a) * sx + Math.cos(a) * sy];
}

export function filmSize(shot) {
  return (shot.vidRot || 0) % 2 ? [shot.h, shot.w] : [shot.w, shot.h];
}

export function renderCamera(shot, camera, focal) {
  const scale = Math.max(shot.w, shot.h);
  const [w, h] = filmSize(shot);
  const a = (shot.vidRot || 0) * Math.PI / 2;
  const sx = Math.cos(a) * camera.cx - Math.sin(a) * camera.cy;
  const sy = Math.sin(a) * camera.cx + Math.cos(a) * camera.cy;
  return {
    focalScale: camera.focalScale,
    // Texture uploads flip Y, so these are bottom-left UV coordinates.
    center: [0.5 + sx * scale / shot.w, 0.5 + sy * scale / shot.h],
    tan: [w / (2 * scale * focal), h / (2 * scale * focal)],
  };
}

// Maximum angular film radius on the first, monotonic branch of PTLens.
// A negative polynomial coefficient can send remote rays back into the image
// after its turning point. Merely checking the final UV paints circular copies
// of the image at those angles, even though the real lens never sees them.
export function projectionRadiusLimit(tanX, tanY, center, vidRot, a, b, c) {
  const angle = vidRot * Math.PI / 2, co = Math.cos(angle), si = Math.sin(angle);
  let corner = 0;
  for (const u of [0, 1]) for (const v of [0, 1]) {
    const x = 2 * (u - center[0]), y = 2 * (v - center[1]);
    corner = Math.max(corner, Math.hypot((co * x + si * y) * tanX, (-si * x + co * y) * tanY));
  }
  if (!a && !b && !c) return corner;
  const value = (r) => r * (1 + r * (a + r * (b + r * c)));
  const slope = (r) => 1 + r * (2 * a + r * (3 * b + r * 4 * c));
  const bisect = (lo, hi, fn, target) => {
    for (let i = 0; i < 48; i++) {
      const mid = (lo + hi) / 2;
      if (fn(mid) < target) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  };
  // Extrema of the cubic derivative split it into monotonic intervals. Check
  // each interval so a later positive branch cannot conceal an earlier fold.
  const critical = [];
  if (Math.abs(c) > 1e-14) {
    const disc = 36 * b * b - 96 * a * c;
    if (disc >= 0) for (const sign of [-1, 1]) critical.push((-6 * b + sign * Math.sqrt(disc)) / (24 * c));
  } else if (Math.abs(b) > 1e-14) critical.push(-a / (3 * b));
  const stops = critical.filter((r) => r > 0 && Number.isFinite(r)).sort((x, y) => x - y);
  let lo = 0;
  const limitAt = (hi) => {
    if (slope(hi) <= 0) {
      const fold = bisect(lo, hi, (r) => -slope(r), 0);
      return value(fold) >= corner ? bisect(0, fold, value, corner) : fold;
    }
    return value(hi) >= corner ? bisect(0, hi, value, corner) : null;
  };
  for (const hi of stops) {
    const limit = limitAt(hi);
    if (limit !== null) return limit;
    lo = hi;
  }
  let hi = Math.max(1, corner * 2, lo * 2);
  for (let i = 0; i < 48; i++, hi *= 2) {
    const limit = limitAt(hi);
    if (limit !== null) return limit;
    lo = hi;
  }
  return corner; // guard against non-finite/invalid calibration
}
