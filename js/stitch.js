// Feature-based panorama solver, run once when the user taps "Done".
//
//   ORB features  ->  match IMU-adjacent pairs  ->  RANSAC homography
//   ->  focal length from the homographies (median)
//   ->  per-pair relative-rotation refinement (Gauss-Newton)
//   ->  global L2 rotation averaging, anchored to the gyro
//   ->  gain compensation
//
// Input  shots: [{ gray:Uint8, w, h, quat:[x,y,z,w], hfovDeg }]  (gray = a
//        downscaled luma copy; quat is the device pose, camera->world).
// Output { rotations:[mat3 row-major per shot], focalScale, gains:[per shot],
//          connected:[bool], ok, log }

import { detectAndDescribe, matchDescriptors } from './orb.js';
import {
  ransacHomography, focalFromHomography, relRotFromHomography, refineRelRot,
  rotationAverage, gainCompensate, qToR, matMul3, matT3, logSO3,
} from './ba.js';

const DEG = Math.PI / 180;
const fwd = (R) => [-R[6], -R[7], -R[8]];
const angBetween = (a, b) => Math.acos(Math.min(1, Math.max(-1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])));
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const tick = () => new Promise((r) => setTimeout(r, 0));

export async function stitch(shots, { onProgress = () => {} } = {}) {
  const N = shots.length;
  const log = [];
  const gyroR = shots.map((s) => qToR(s.quat));
  const hfov = (shots[0].hfovDeg || 55) * DEG;
  const w = shots[0].w, h = shots[0].h;
  const cx = w / 2, cy = h / 2;
  const focal0 = (w / 2) / Math.tan(hfov / 2);
  const bail = () => ({ rotations: gyroR, focalScale: 1, gains: shots.map(() => 1), connected: shots.map(() => true), ok: false, log });

  if (N < 2) { log.push('need >= 2 frames'); return bail(); }

  // 1. features
  const feats = [];
  for (let i = 0; i < N; i++) {
    onProgress('features', i / N);
    feats.push(detectAndDescribe(shots[i].gray, shots[i].w, shots[i].h, { fastThresh: 14, maxFeatures: 800 }));
    await tick();
  }
  log.push('features: ' + feats.map((f) => f.kps.length).join(','));

  // 2. candidate pairs: IMU-adjacent (bounded fan-out) + every consecutive pair
  // (capture order tracks a sweep, and covers the case where the gyro is dead).
  const maxAng = Math.min(hfov * 1.15, 1.4);
  const seen = new Set();
  const cand = [];
  const addPair = (i, j) => {
    const a = Math.min(i, j), b = Math.max(i, j);
    if (a === b || seen.has(a * N + b)) return;
    seen.add(a * N + b); cand.push([a, b]);
  };
  for (let i = 0; i < N; i++) {
    const near = [];
    for (let j = 0; j < N; j++) if (j !== i) {
      const a = angBetween(fwd(gyroR[i]), fwd(gyroR[j]));
      if (a < maxAng) near.push([j, a]);
    }
    near.sort((p, q) => p[1] - q[1]);
    for (const [j] of near.slice(0, 7)) addPair(i, j);
  }
  for (let i = 0; i + 1 < N; i++) { addPair(i, i + 1); addPair(i, Math.min(i + 2, N - 1)); }

  // 3. match + homography verification
  const verified = []; // { i, j, mc:[centered matches], inl, Ii, Ij, H }
  const focals = [];
  let bestRaw = 0, bestInl = 0;
  for (let c = 0; c < cand.length; c++) {
    onProgress('matching', c / cand.length);
    const [i, j] = cand[c];
    const raw = matchDescriptors(feats[i].desc, feats[j].desc, 0.82);
    bestRaw = Math.max(bestRaw, raw.length);
    if (raw.length >= 10) {
      const pa = raw.map(([a]) => [feats[i].kps[a].x, feats[i].kps[a].y]);
      const pb = raw.map(([, b]) => [feats[j].kps[b].x, feats[j].kps[b].y]);
      const ca = pa.map((p) => [p[0] - cx, p[1] - cy]);
      const cb = pb.map((p) => [p[0] - cx, p[1] - cy]);
      const { H, inliers } = ransacHomography(ca, cb, { iters: 500, thresh: 3.5 });
      bestInl = Math.max(bestInl, inliers.length);
      if (inliers.length >= 12 && H) {
        const step = Math.max(1, Math.floor(inliers.length / 70));
        const mc = [];
        for (let k = 0; k < inliers.length; k += step) {
          const t = inliers[k];
          mc.push([ca[t][0], ca[t][1], cb[t][0], cb[t][1]]);
        }
        let si = 0, sj = 0;
        const gi = shots[i].gray, gj = shots[j].gray;
        for (const t of inliers) {
          si += gi[Math.round(pa[t][1]) * shots[i].w + Math.round(pa[t][0])] || 0;
          sj += gj[Math.round(pb[t][1]) * shots[j].w + Math.round(pb[t][0])] || 0;
        }
        verified.push({ i, j, mc, inl: inliers.length, Ii: si / inliers.length / 255, Ij: sj / inliers.length / 255, H });
        for (const f of focalFromHomography(H)) if (f > 0.3 * focal0 && f < 3 * focal0) focals.push(f);
      }
    }
    await tick();
  }
  log.push(`verified pairs: ${verified.length} (best raw matches ${bestRaw}, best inliers ${bestInl}, ${cand.length} pairs tried)`);
  if (verified.length === 0) return bail();

  // 4. focal length: median of the per-homography estimates
  let focal = focals.length >= 3 ? median(focals) : focal0;
  focal = clamp(focal, 0.5 * focal0, 2.0 * focal0);
  const focalScale = focal / focal0;
  log.push(`focal ${focal.toFixed(1)}px (init ${focal0.toFixed(1)}, ${focals.length} votes, scale ${focalScale.toFixed(3)})`);

  // 5. refine each pair's relative rotation, seeded from the gyro
  onProgress('optimizing', 0);
  const edges = [];
  const UF = new UnionFind(N);
  for (let e = 0; e < verified.length; e++) {
    onProgress('optimizing', e / verified.length * 0.6);
    const v = verified[e];
    const gyroSeed = matMul3(matT3(gyroR[v.j]), gyroR[v.i]); // cam_i ray -> cam_j ray
    // seed from the homography (already fits the matches); sanity-check vs gyro
    let seed = gyroSeed;
    try {
      const hSeed = relRotFromHomography(v.H, focal);
      const off = Math.hypot(...logSO3(matMul3(hSeed, matT3(gyroSeed))));
      if (hSeed.every((x) => isFinite(x)) && off < 0.6) seed = hSeed;
    } catch { /* keep gyro seed */ }
    const { Rrel, rms, inl } = refineRelRot(seed, v.mc, focal);
    if (rms < 4 && inl >= 10) { edges.push({ i: v.i, j: v.j, Rrel, w: Math.min(inl, 120) }); UF.union(v.i, v.j); }
    v.rms = rms;
    await tick();
  }
  log.push(`good edges: ${edges.length}`);
  log.push('pairs: ' + verified.map((v) => `${v.i}-${v.j}:${(v.rms || 9).toFixed(1)}/${v.inl}`).join(' '));
  if (edges.length === 0) return bail();

  // 6. largest connected component
  const root = Array.from({ length: N }, (_, i) => UF.find(i));
  const cnt = {};
  root.forEach((r) => (cnt[r] = (cnt[r] || 0) + 1));
  const big = +Object.entries(cnt).sort((a, b) => b[1] - a[1])[0][0];
  const connected = root.map((r) => r === big);
  log.push(`connected ${connected.filter(Boolean).length}/${N}`);

  // 7. global rotation averaging (anchored softly to the gyro)
  onProgress('optimizing', 0.7);
  const subEdges = edges.filter((e) => connected[e.i] && connected[e.j]);
  const avg = rotationAverage(N, subEdges, gyroR, { priorW: 0.2, iters: 60 });
  const rotations = gyroR.map((g, k) => {
    if (!connected[k]) return g;
    const drift = Math.hypot(...logSO3(matMul3(avg[k], matT3(g))));
    return drift < 0.5 && avg[k].every((v) => isFinite(v)) ? avg[k] : g;
  });
  onProgress('optimizing', 1);

  // 8. gain compensation over connected pairs
  const gains = shots.map(() => 1);
  const idxOf = [];
  const sub = [];
  connected.forEach((ok, i) => { if (ok) { idxOf[i] = sub.length; sub.push(i); } });
  const gPairs = verified.filter((v) => connected[v.i] && connected[v.j]);
  if (gPairs.length) {
    const gg = gainCompensate(sub.length,
      gPairs.map((v) => ({ i: idxOf[v.i], j: idxOf[v.j], Ii: v.Ii, Ij: v.Ij, n: v.inl })),
      { sigmaN: 0.012, sigmaG: 0.1 });
    sub.forEach((orig, k) => { gains[orig] = clamp(gg[k], 0.6, 1.6); });
  }

  return { rotations, focalScale, gains, connected, ok: true, log };
}

class UnionFind {
  constructor(n) { this.p = Array.from({ length: n }, (_, i) => i); }
  find(x) { while (this.p[x] !== x) { this.p[x] = this.p[this.p[x]]; x = this.p[x]; } return x; }
  union(a, b) { this.p[this.find(a)] = this.find(b); }
}
