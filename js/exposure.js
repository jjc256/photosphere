import { solveSPD } from './ba.js';

const median = (values) => {
  values.sort((a, b) => a - b);
  return values[values.length >> 1];
};

// Robust per-channel gain from geometrically overlapping, warped images.
// Descriptor matches are not required: flat walls still carry exposure data.
// Warps are bottom-up RGBA8, with alpha indicating real captured coverage.
export function overlapExposure(warps, width, height, initialGains = warps.map(() => 1)) {
  const n = warps.length;
  if (!n) return [];
  const edges = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const a = warps[i], b = warps[j], ratios = [[], [], []];
    for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
      const p = (y * width + x) * 4;
      if (!a[p + 3] || !b[p + 3]) continue;
      // Prefer smooth patches: a small pose error at a window or chair edge
      // must not masquerade as a change of exposure or white balance.
      if (Math.abs(a[p - 4] - a[p + 4]) + Math.abs(a[p - width * 4] - a[p + width * 4]) > 30 ||
          Math.abs(b[p - 4] - b[p + 4]) + Math.abs(b[p - width * 4] - b[p + width * 4]) > 30) continue;
      for (let c = 0; c < 3; c++) {
        if (a[p + c] < 16 || b[p + c] < 16 || a[p + c] > 235 || b[p + c] > 235) continue;
        ratios[c].push(Math.log(b[p + c] / a[p + c]));
      }
    }
    for (let c = 0; c < 3; c++) {
      if (ratios[c].length < 24) continue;
      const delta = median(ratios[c]);
      const mad = median(ratios[c].map((r) => Math.abs(r - delta)));
      if (mad > 0.12 || Math.abs(delta) > Math.log(3)) continue;
      edges.push({ i, j, c, delta, weight: Math.min(1, ratios[c].length / 200) / (1 + (mad / 0.04) ** 2) });
    }
  }
  if (!edges.length) return initialGains.map((g) => [g, g, g]);
  const gains = Array.from({ length: n }, () => [1, 1, 1]);
  for (let c = 0; c < 3; c++) {
    const H = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i === j ? 0.01 : 0));
    const rhs = initialGains.map((g) => 0.01 * Math.log(g));
    for (const e of edges) {
      if (e.c !== c) continue;
      H[e.i][e.i] += e.weight; H[e.j][e.j] += e.weight;
      H[e.i][e.j] -= e.weight; H[e.j][e.i] -= e.weight;
      rhs[e.i] += e.weight * e.delta; rhs[e.j] -= e.weight * e.delta;
    }
    const x = solveSPD(H, rhs, n);
    for (let i = 0; i < n; i++) gains[i][c] = Math.exp(Math.max(-Math.log(2), Math.min(Math.log(2), x[i])));
  }
  // Preserve highlight detail when bringing dark auto-exposures up to match
  // their neighbours. One common scale keeps the solved relative gains/WB.
  const highlights = [];
  for (let i = 0; i < n; i++) for (let p = 0; p < warps[i].length; p += 64) {
    if (!warps[i][p + 3]) continue;
    highlights.push(Math.max(...gains[i].map((g, c) => g * warps[i][p + c])));
  }
  if (highlights.length) {
    highlights.sort((a, b) => a - b);
    const scale = Math.min(1, 250 / highlights[Math.floor(highlights.length * 0.99)]);
    for (const gain of gains) for (let c = 0; c < 3; c++) gain[c] *= scale;
  }
  return gains;
}
