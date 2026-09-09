// Residual registration on small spherical previews. The global camera solve
// stays responsible for projection; this smooth, bounded field only reconciles
// nearby overlapping details displaced by handheld parallax.
export function localAlignment(images, w, h, { radius = 6, spacing = 16 } = {}) {
  const gw = Math.ceil(w / spacing),
    gh = Math.ceil(h / spacing),
    size = gw * gh;
  const count = images.length * size;
  const planes = images.map((p) => {
    const gray = new Float32Array(w * h),
      valid = new Uint8Array(w * h);
    for (let k = 0; k < gray.length; k++) {
      gray[k] = (p[k * 4] + p[k * 4 + 1] + p[k * 4 + 2]) / 3;
      valid[k] = p[k * 4 + 3] > 250;
    }
    return { gray, valid };
  });
  const wrap = (x) => (x + w) % w;
  function patch(im, x, y) {
    if (y < 4 || y >= h - 4) return null;
    const out = [];
    let sum = 0,
      sq = 0;
    for (let dy = -4; dy <= 4; dy += 2)
      for (let dx = -4; dx <= 4; dx += 2) {
        const k = (y + dy) * w + wrap(x + dx);
        if (!im.valid[k]) return null;
        const v = im.gray[k];
        out.push(v);
        sum += v;
        sq += v * v;
      }
    const norm = Math.sqrt(Math.max(0, sq - (sum * sum) / 25));
    if (norm < 15) return null;
    return out.map((v) => (v - sum / 25) / norm);
  }
  function match(a, im, x, y) {
    let best = -Infinity,
      bx = 0,
      by = 0,
      raw = -1,
      atZero = -1;
    for (let dy = -radius; dy <= radius; dy++)
      for (let dx = -radius; dx <= radius; dx++) {
        const b = patch(im, wrap(x + dx), y + dy);
        if (!b) continue;
        let corr = 0;
        for (let k = 0; k < 25; k++) corr += a[k] * b[k];
        if (!dx && !dy) atZero = corr;
        const score = corr - 0.001 * (dx * dx + dy * dy);
        if (score > best) {
          best = score;
          raw = corr;
          bx = dx;
          by = dy;
        }
      }
    if (atZero > 0.9995) return { dx: 0, dy: 0, ix: 0, iy: 0, score: atZero };
    // Quadratic interpolation gives subpixel offsets without enlarging the
    // deformation grid or rounding every alignment to a visible stair step.
    const corrAt = (dx, dy) => {
      const b = patch(im, wrap(x + dx), y + dy);
      return b ? a.reduce((sum, v, k) => sum + v * b[k], 0) - 0.001 * (dx * dx + dy * dy) : -1;
    };
    const peak = (lo, mid, hi) => {
      const curve = lo - 2 * mid + hi;
      return curve < -1e-5 ? Math.max(-0.5, Math.min(0.5, (0.5 * (lo - hi)) / curve)) : 0;
    };
    const sx = peak(corrAt(bx - 1, by), best, corrAt(bx + 1, by));
    const sy = peak(corrAt(bx, by - 1), best, corrAt(bx, by + 1));
    return { dx: bx + sx, dy: by + sy, ix: bx, iy: by, score: raw, atZero };
  }
  function nodes(frame, x, y) {
    const fx = ((x + 0.5) / w) * gw - 0.5,
      fy = ((y + 0.5) / h) * gh - 0.5;
    const ix = Math.floor(fx),
      iy = Math.floor(fy),
      ax = fx - ix,
      ay = fy - iy;
    const out = [];
    for (let dy = 0; dy < 2; dy++)
      for (let dx = 0; dx < 2; dx++)
        out.push([
          frame * size + Math.max(0, Math.min(gh - 1, iy + dy)) * gw + ((ix + dx + gw) % gw),
          (dx ? ax : 1 - ax) * (dy ? ay : 1 - ay),
        ]);
    return out;
  }
  const rows = [];
  let matches = 0,
    before = 0;
  for (let i = 0; i < images.length; i++)
    for (let j = i + 1; j < images.length; j++) {
      for (let y = 8; y < h - 8; y += 8)
        for (let x = 8; x < w; x += 8) {
          if (!planes[i].valid[y * w + x] || !planes[j].valid[y * w + x]) continue;
          const a = patch(planes[i], x, y);
          if (!a) continue;
          const m = match(a, planes[j], x, y);
          if (m.score < 0.94 || Math.abs(m.ix) === radius || Math.abs(m.iy) === radius) continue;
          const b = patch(planes[j], wrap(x + m.ix), y + m.iy);
          if (!b) continue;
          const back = match(b, planes[i], wrap(x + m.ix), y + m.iy);
          if (back.score < 0.94 || Math.hypot(m.ix + back.ix, m.iy + back.iy) > 1) continue;
          // Exact or already coherent overlap anchors the field at zero too.
          const coeff = [...nodes(i, x, y).map(([k, v]) => [k, -v]), ...nodes(j, wrap(x + m.ix), y + m.iy)];
          rows.push({ coeff, dx: m.dx, dy: m.dy, weight: 1 });
          matches++;
          before += m.dx * m.dx + m.dy * m.dy;
        }
    }
  // Smooth neighboring vertices and shrink unconstrained regions toward the
  // original projection. Longitude is periodic; latitude never wraps.
  for (let f = 0; f < images.length; f++)
    for (let y = 0; y < gh; y++)
      for (let x = 0; x < gw; x++) {
        const k = f * size + y * gw + x;
        rows.push({
          coeff: [
            [k, 1],
            [f * size + y * gw + ((x + 1) % gw), -1],
          ],
          dx: 0,
          dy: 0,
          weight: 0.2,
        });
        if (y + 1 < gh)
          rows.push({
            coeff: [
              [k, 1],
              [k + gw, -1],
            ],
            dx: 0,
            dy: 0,
            weight: 0.2,
          });
      }
  function solve(axis) {
    const b = new Float64Array(count),
      diag = new Float64Array(count).fill(0.02);
    for (const row of rows)
      for (const [k, v] of row.coeff) {
        b[k] += row.weight * v * row[axis];
        diag[k] += row.weight * v * v;
      }
    const mul = (x) => {
      const out = Float64Array.from(x, (v) => v * 0.02);
      for (const row of rows) {
        let dot = 0;
        for (const [k, v] of row.coeff) dot += v * x[k];
        for (const [k, v] of row.coeff) out[k] += row.weight * v * dot;
      }
      return out;
    };
    const x = new Float64Array(count),
      r = b.slice(),
      z = Float64Array.from(r, (v, k) => v / diag[k]),
      p = z.slice();
    let rz = r.reduce((s, v, k) => s + v * z[k], 0);
    for (let iter = 0; iter < 80 && rz > 1e-12; iter++) {
      const ap = mul(p),
        pap = p.reduce((s, v, k) => s + v * ap[k], 0);
      if (pap <= 0) break;
      const alpha = rz / pap;
      for (let k = 0; k < count; k++) {
        x[k] += alpha * p[k];
        r[k] -= alpha * ap[k];
        z[k] = r[k] / diag[k];
      }
      const next = r.reduce((s, v, k) => s + v * z[k], 0),
        beta = next / rz;
      for (let k = 0; k < count; k++) p[k] = z[k] + beta * p[k];
      rz = next;
    }
    return x;
  }
  const dx = solve('dx'),
    dy = solve('dy');
  // Bound both displacement and its spatial derivative. This keeps the mesh
  // invertible, even if an accidental repeated-pattern match slips through.
  for (let f = 0; f < images.length; f++) {
    let scale = 1;
    for (let y = 0; y < gh; y++)
      for (let x = 0; x < gw; x++) {
        const k = f * size + y * gw + x;
        scale = Math.min(scale, radius / Math.max(radius, Math.abs(dx[k]), Math.abs(dy[k])));
        for (const [n, step] of [
          [f * size + y * gw + ((x + 1) % gw), w / gw],
          ...(y + 1 < gh ? [[k + gw, h / gh]] : []),
        ]) {
          const distance = Math.hypot(dx[k] - dx[n], dy[k] - dy[n]);
          if (distance > step * 0.25) scale = Math.min(scale, (step * 0.25) / distance);
        }
      }
    for (let k = f * size; k < (f + 1) * size; k++) {
      dx[k] *= scale;
      dy[k] *= scale;
    }
  }
  let after = 0,
    maxShift = 0;
  for (const row of rows.slice(0, matches)) {
    let x = 0,
      y = 0;
    for (const [k, v] of row.coeff) {
      x += v * dx[k];
      y += v * dy[k];
    }
    after += (x - row.dx) ** 2 + (y - row.dy) ** 2;
  }
  const fields = images.map((_, f) => {
    const a = new Float32Array(size * 2);
    for (let k = 0; k < size; k++) {
      a[k * 2] = dx[f * size + k] / w;
      a[k * 2 + 1] = dy[f * size + k] / h;
      maxShift = Math.max(maxShift, Math.hypot(dx[f * size + k], dy[f * size + k]));
    }
    return a;
  });
  return {
    fields,
    width: gw,
    height: gh,
    diagnostics: {
      matches,
      beforeRms: Math.sqrt(before / Math.max(1, matches)),
      afterRms: Math.sqrt(after / Math.max(1, matches)),
      maxShift,
    },
  };
}
