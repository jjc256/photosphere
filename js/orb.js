// Compact ORB: FAST-9 keypoints + oriented BRIEF-256 descriptors + Hamming
// matcher with Lowe ratio test and cross-check. Pure JS, works on a plain
// grayscale Uint8Array so it runs the same in Node (for the self-test) and
// in the browser.

const CIRCLE = [
  [0, -3], [1, -3], [2, -2], [3, -1], [3, 0], [3, 1], [2, 2], [1, 3],
  [0, 3], [-1, 3], [-2, 2], [-3, 1], [-3, 0], [-3, -1], [-2, -2], [-1, -3],
];

export function toGray(rgba, w, h) {
  const g = new Uint8ClampedArray(w * h);
  for (let i = 0, p = 0; i < g.length; i++, p += 4) {
    g[i] = (rgba[p] * 77 + rgba[p + 1] * 150 + rgba[p + 2] * 29) >> 8;
  }
  return g;
}

// Separable binomial blur (approx Gaussian), `passes` times.
export function blur(src, w, h, passes = 2) {
  let a = Float32Array.from(src);
  let b = new Float32Array(w * h);
  for (let k = 0; k < passes; k++) {
    for (let y = 0; y < h; y++) {
      const r = y * w;
      for (let x = 0; x < w; x++) {
        const l = x > 0 ? a[r + x - 1] : a[r + x];
        const rr = x < w - 1 ? a[r + x + 1] : a[r + x];
        b[r + x] = (l + 2 * a[r + x] + rr) * 0.25;
      }
    }
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        const u = y > 0 ? b[(y - 1) * w + x] : b[y * w + x];
        const d = y < h - 1 ? b[(y + 1) * w + x] : b[y * w + x];
        a[y * w + x] = (u + 2 * b[y * w + x] + d) * 0.25;
      }
    }
  }
  return a; // Float32
}

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Fixed BRIEF sampling pattern: 256 point pairs in a radius-15 disc.
const PATCH = 15;
const PATTERN = (() => {
  const rnd = mulberry32(0x9e3779b9);
  const g = () => {
    // sum of 3 uniforms ≈ gaussian, scaled into the patch
    const v = (rnd() + rnd() + rnd() - 1.5) * (PATCH / 1.5);
    return Math.max(-PATCH, Math.min(PATCH, Math.round(v)));
  };
  const p = new Int8Array(256 * 4);
  for (let i = 0; i < 256; i++) { p[i * 4] = g(); p[i * 4 + 1] = g(); p[i * 4 + 2] = g(); p[i * 4 + 3] = g(); }
  return p;
})();

const POP = (() => { const t = new Uint8Array(256); for (let i = 0; i < 256; i++) t[i] = (i & 1) + t[i >> 1]; return t; })();

export function detectAndDescribe(gray, w, h, {
  fastThresh = 20, maxFeatures = 700, nmsRadius = 7, blurPasses = 1, minKeypoints = 90, _depth = 0,
} = {}) {
  const g = blur(gray, w, h, blurPasses);
  const border = PATCH + 4;
  const cand = [];

  for (let y = border; y < h - border; y++) {
    for (let x = border; x < w - border; x++) {
      const v = g[y * w + x];
      const hi = v + fastThresh, lo = v - fastThresh;
      // quick reject on the 4 compass points
      let brighter = 0, darker = 0;
      for (const idx of [0, 4, 8, 12]) {
        const s = g[(y + CIRCLE[idx][1]) * w + x + CIRCLE[idx][0]];
        if (s > hi) brighter++; else if (s < lo) darker++;
      }
      if (brighter < 3 && darker < 3) continue;

      // contiguous arc of >=9 around the 16-ring
      let best = 0, run = 0, sign = 0;
      for (let i = 0; i < 24; i++) {
        const c = CIRCLE[i % 16];
        const s = g[(y + c[1]) * w + x + c[0]];
        const cur = s > hi ? 1 : s < lo ? -1 : 0;
        if (cur !== 0 && cur === sign) run++;
        else { sign = cur; run = cur !== 0 ? 1 : 0; }
        if (run > best) best = run;
      }
      if (best < 9) continue;

      let score = 0;
      for (let i = 0; i < 16; i++) {
        const c = CIRCLE[i];
        score += Math.abs(g[(y + c[1]) * w + x + c[0]] - v);
      }
      cand.push({ x, y, score });
    }
  }

  cand.sort((a, b) => b.score - a.score);

  // grid-accelerated non-max suppression
  const cell = Math.max(1, nmsRadius);
  const gw = Math.ceil(w / cell);
  const grid = new Map();
  const kps = [];
  for (const k of cand) {
    if (kps.length >= maxFeatures) break;
    const cx = (k.x / cell) | 0, cy = (k.y / cell) | 0;
    let ok = true;
    for (let dy = -1; dy <= 1 && ok; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const arr = grid.get((cy + dy) * gw + (cx + dx));
        if (!arr) continue;
        for (const j of arr) {
          if ((j.x - k.x) ** 2 + (j.y - k.y) ** 2 < nmsRadius * nmsRadius) { ok = false; break; }
        }
        if (!ok) break;
      }
    }
    if (!ok) continue;
    const key = cy * gw + cx;
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(k);
    kps.push(k);
  }

  // adaptive: low-contrast frame -> retry with a gentler corner threshold
  if (kps.length < minKeypoints && fastThresh > 6 && _depth < 2) {
    return detectAndDescribe(gray, w, h, {
      fastThresh: Math.max(6, Math.round(fastThresh * 0.55)),
      maxFeatures, nmsRadius, blurPasses, minKeypoints, _depth: _depth + 1,
    });
  }

  // orientation (intensity centroid) + descriptor
  const desc = new Uint8Array(kps.length * 32);
  const r2 = PATCH * PATCH;
  for (let n = 0; n < kps.length; n++) {
    const { x, y } = kps[n];
    let m01 = 0, m10 = 0;
    for (let dy = -PATCH; dy <= PATCH; dy++) {
      const yy = (y + dy) * w;
      for (let dx = -PATCH; dx <= PATCH; dx++) {
        if (dx * dx + dy * dy > r2) continue;
        const I = g[yy + x + dx];
        m10 += dx * I; m01 += dy * I;
      }
    }
    const ang = Math.atan2(m01, m10);
    kps[n].angle = ang;
    const c = Math.cos(ang), s = Math.sin(ang);
    for (let i = 0; i < 256; i++) {
      const ax = PATTERN[i * 4], ay = PATTERN[i * 4 + 1];
      const bx = PATTERN[i * 4 + 2], by = PATTERN[i * 4 + 3];
      const rax = (c * ax - s * ay) | 0, ray = (s * ax + c * ay) | 0;
      const rbx = (c * bx - s * by) | 0, rby = (s * bx + c * by) | 0;
      const pa = g[(y + ray) * w + x + rax];
      const pb = g[(y + rby) * w + x + rbx];
      if (pa < pb) desc[n * 32 + (i >> 3)] |= 1 << (i & 7);
    }
  }
  return { kps, desc };
}

function hamming(d, a, e, b) {
  let s = 0;
  for (let i = 0; i < 32; i++) s += POP[d[a + i] ^ e[b + i]];
  return s;
}

// Returns [[iA, iB], ...] surviving ratio test + cross-check.
export function matchDescriptors(dA, dB, ratio = 0.8) {
  const nA = dA.length / 32, nB = dB.length / 32;
  const bestBforA = new Int32Array(nA).fill(-1);
  const bestAforB = new Int32Array(nB).fill(-1);
  const scoreAforB = new Int32Array(nB).fill(257);

  for (let i = 0; i < nA; i++) {
    let b1 = 300, b2 = 300, j1 = -1;
    const off = i * 32;
    for (let j = 0; j < nB; j++) {
      const s = hamming(dA, off, dB, j * 32);
      if (s < b1) { b2 = b1; b1 = s; j1 = j; }
      else if (s < b2) b2 = s;
      if (s < scoreAforB[j]) { scoreAforB[j] = s; bestAforB[j] = i; }
    }
    if (j1 >= 0 && b1 < ratio * b2) bestBforA[i] = j1;
  }
  const out = [];
  for (let i = 0; i < nA; i++) {
    const j = bestBforA[i];
    if (j >= 0 && bestAforB[j] === i) out.push([i, j]);
  }
  return out;
}
