// Feature-based panorama solver, run once when the user taps "Done".
//
//   SIFT/ORB features -> all-pairs matching -> calibrated-ray RANSAC
//   -> shared lens calibration -> robust pair refinement
//   -> incremental and global bundle adjustment -> gain compensation
//
// Input  shots: [{ gray:Uint8, w, h, quat:[x,y,z,w], hfovDeg, vidRot }]  (gray = a
//        downscaled luma copy; quat is the device pose, camera->world).
// Output { rotations:[mat3 row-major per shot], focalScale, gains:[per shot],
//          connected:[bool], ok, log }

import { filmPoint, filmSize, renderCamera } from './camera-geometry.js';
import { featureBackend } from './cv-features.js';
import { generalizedLensSolver } from './solver-wasm.js';
import { globalBundleAdjust } from './solver-worker-client.js';
import {
  ransacHomography, focalFromHomography, relRotFromHomography, ransacGeneralizedRotation, refineRelRot,
  bundleAdjust, gainCompensate, anchorComponents, qToR, matMul3, matT3, logSO3, setGeneralizedLensKernel,
} from './ba.js';

const DEG = Math.PI / 180;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const tick = () => new Promise((r) => setTimeout(r, 0));
const I3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

function polishPair(matchBlock, Rseed, focal, linearity, {
  k1 = 0, k2 = 0, k3 = 0, coordinateScale = 512, iters = 8,
} = {}) {
  const Rj = matT3(Rseed);
  const fit = bundleAdjust([{ w: 1, h: 1 }, { w: 1, h: 1 }], [I3, Rj], [{ i: 0, j: 1, m: matchBlock }], {
    focal0: focal, cx: 0, cy: 0, k1, k2, k3, linearity,
    optimizeFocal: true, optimizeLens: true, optimizeDistortion: false,
    optimizePerFrame: true, optimizePerFrameCenter: false,
    minLinearity: -1.2, maxLinearity: 1.8,
    priorW: 0.005, huber: 4 / coordinateScale, iters, anchor: 0,
  });
  const rel = matMul3(matT3(fit.R[1]), fit.R[0]);
  const scale0 = fit.cameras?.[0]?.focalScale || 1;
  const scale1 = fit.cameras?.[1]?.focalScale || 1;
  return {
    Rrel: rel,
    focal: fit.focal * Math.sqrt(scale0 * scale1),
    linearity: fit.linearity,
    cost: fit.cost,
  };
}

function incrementalBuild(N, edges, gyroR, frames, {
  focal, cx, cy, k1, k2, k3, linearity, coordinateScale,
}) {
  const rotations = gyroR.map((R) => R.slice());
  const built = new Array(N).fill(false);
  const activeEdges = [];
  const uf = new UnionFind(N);
  const remaining = edges.slice().sort((a, b) => (b.w || 0) - (a.w || 0));
  let cameras = frames.map(() => ({ focalScale: 1, cx, cy }));
  let curFocal = focal, curLinearity = linearity, curK1 = k1, curK2 = k2, curK3 = k3;

  const addEdge = (edge) => {
    if (built[edge.i] && !built[edge.j]) rotations[edge.j] = matMul3(rotations[edge.i], matT3(edge.Rrel));
    else if (!built[edge.i] && built[edge.j]) rotations[edge.i] = matMul3(rotations[edge.j], edge.Rrel);
    else if (!built[edge.i] && !built[edge.j]) rotations[edge.j] = matMul3(rotations[edge.i], matT3(edge.Rrel));
    built[edge.i] = true;
    built[edge.j] = true;
    uf.union(edge.i, edge.j);
    activeEdges.push(edge);
  };

  const runBA = (iters, optimizeDistortion = false) => {
    const ba = bundleAdjust(frames, rotations, activeEdges.map((e) => ({ i: e.i, j: e.j, m: e.mc })), {
      focal0: curFocal, initialCameras: cameras, cx, cy, k1: curK1, k2: curK2, k3: curK3, linearity: curLinearity,
      optimizeFocal: true, optimizeLens: !optimizeDistortion, optimizeDistortion,
      optimizePerFrame: true, optimizePerFrameCenter: false,
      minLinearity: -1.2, maxLinearity: 1.8,
      priorW: 0.06, huber: 5 / coordinateScale, iters,
      anchor: activeEdges[0]?.i || 0,
    });
    if (!ba.R.every((R) => R.every((v) => isFinite(v))) || !isFinite(ba.focal)) return false;
    for (let k = 0; k < N; k++) rotations[k] = ba.R[k];
    cameras = ba.cameras;
    curFocal = ba.focal;
    curLinearity = ba.linearity;
    curK1 = ba.k1; curK2 = ba.k2; curK3 = ba.k3;
    return true;
  };

  while (remaining.length) {
    let idx = remaining.findIndex((e) => built[e.i] !== built[e.j]);
    if (idx < 0) idx = remaining.findIndex((e) => built[e.i] && built[e.j] && uf.find(e.i) !== uf.find(e.j));
    if (idx < 0) idx = remaining.findIndex((e) => !built[e.i] && !built[e.j]);
    if (idx < 0) {
      for (const e of remaining.splice(0)) if (built[e.i] && built[e.j] && uf.find(e.i) === uf.find(e.j)) activeEdges.push(e);
      break;
    }
    const edge = remaining.splice(idx, 1)[0];
    addEdge(edge);
    runBA(activeEdges.length === 1 ? 12 : 8, false);

    for (let i = remaining.length - 1; i >= 0; i--) {
      const e = remaining[i];
      if (built[e.i] && built[e.j] && uf.find(e.i) === uf.find(e.j)) {
        activeEdges.push(e);
        remaining.splice(i, 1);
      }
    }
  }

  if (activeEdges.length) runBA(N > 24 ? 10 : 18, false);
  return { rotations, cameras, focal: curFocal, linearity: curLinearity, k1: curK1, k2: curK2, k3: curK3, built, activeEdges, uf };
}

export async function stitch(shots, { onProgress = () => {} } = {}) {
  const N = shots.length;
  const log = [];
  const gyroR = shots.map((s) => qToR(s.quat));
  if (!N) return { rotations: [], cameras: [], gains: [], connected: [], reliable: [], focalScale: 1, ok: false, log: ['need >= 2 frames'] };
  const hfov = (shots[0].hfovDeg || 55) * DEG;
  const w = shots[0].w, h = shots[0].h;
  // Panorama stores feature/film coordinates centred and divided by the
  // longer image edge.  Its camera, PTLens coefficients and RANSAC thresholds
  // all live in that unit system, so retain it through bundle adjustment.
  const coordinateScale = Math.max(w, h);
  let cx = 0, cy = 0;
  const focal0 = ((filmSize(shots[0])[0] / 2) / Math.tan(hfov / 2)) / coordinateScale;
  const bail = () => ({ rotations: gyroR, focalScale: 1, k1: 0, k2: 0, k3: 0, gains: shots.map(() => 1), connected: shots.map(() => true), ok: false, log });

  if (N < 2) { log.push('need >= 2 frames'); return bail(); }

  // 1. features
  onProgress('loading vision', 0);
  try { setGeneralizedLensKernel(await generalizedLensSolver()); } catch { /* JS equations remain available */ }
  const vision = await featureBackend();
  log.push(`vision: ${vision.kind}`);
  const feats = [];
  for (let i = 0; i < N; i++) {
    onProgress('features', i / N);
    feats.push(vision.detect(shots[i].gray, shots[i].w, shots[i].h));
    await tick();
  }
  const counts = feats.map((f) => f.kps.length);
  const usable = counts.filter((c) => c >= 60).length;
  log.push(`features: ${counts.join(',')}`);
  log.push(`usable frames (>=60 feat): ${usable}/${N}`);

  // 2. Candidate pairs. Panorama deliberately tries every pair after it has
  // established its ordered backbone: IMU estimates are only priors and must
  // not decide which overlaps exist.
  const seen = new Set();
  const cand = [];
  const addPair = (i, j) => {
    const a = Math.min(i, j), b = Math.max(i, j);
    if (a === b || seen.has(a * N + b)) return;
    seen.add(a * N + b); cand.push([a, b]);
  };
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    // Frames below the feature floor cannot produce a geometric edge, so skip
    // only those. This leaves no IMU/capture-order blind spots in the graph.
    if (counts[i] >= 12 && counts[j] >= 12) addPair(i, j);
  }

  // 3. match + homography verification
  const verified = []; // { i, j, mc:[centered matches], inl, Ii, Ij, H }
  const focals = [];
  let bestRaw = 0, bestInl = 0;
  for (let c = 0; c < cand.length; c++) {
    onProgress('matching', c / cand.length);
    const [i, j] = cand[c];
    const raw = vision.match(feats[i], feats[j]);
    bestRaw = Math.max(bestRaw, raw.length);
    // Panorama's PairAlignment refuses underspecified descriptor sets before
    // RANSAC (40 raw matches, then at least 25 coarse inliers).  The lower
    // browser thresholds admitted weak texture as geometry and produced cuts.
    if (raw.length >= 40) {
      const pa = raw.map(([a]) => [feats[i].kps[a].x, feats[i].kps[a].y]);
      const pb = raw.map(([, b]) => [feats[j].kps[b].x, feats[j].kps[b].y]);
      const ca = pa.map((p) => filmPoint(p[0], p[1], shots[i]));
      const cb = pb.map((p) => filmPoint(p[0], p[1], shots[j]));
      // Direct calibrated-ray RANSAC decides whether the images overlap.
      const gyroSeed = matMul3(matT3(gyroR[j]), gyroR[i]);
      const geometric = ransacGeneralizedRotation(ca.map((p, k) => [p[0], p[1], cb[k][0], cb[k][1]]), focal0, gyroSeed,
        { thresh: 4 / coordinateScale });
      const inliers = geometric.inliers;
      bestInl = Math.max(bestInl, inliers.length);
      if (inliers.length >= 25) {
        // Keep a homography only as a weak focal vote; it no longer controls
        // pair acceptance or the relative-rotation seed.
        const { H } = ransacHomography(ca, cb, { iters: 180, thresh: 4.5 / coordinateScale });
        const step = Math.max(1, Math.floor(inliers.length / 70));
        const mc = [];
        for (let k = 0; k < inliers.length; k += step) {
          const t = inliers[k];
          mc.push([ca[t][0], ca[t][1], cb[t][0], cb[t][1]]);
        }
        let Rseed = geometric.Rrel;
        let pairFocal = focal0;
        let pairLinearity = 1;
        try {
          const coarse = polishPair(mc, Rseed, focal0, 1, { coordinateScale, iters: 6 });
          if (coarse.Rrel.every((x) => isFinite(x)) && isFinite(coarse.focal) && coarse.focal > 0.25 * focal0 && coarse.focal < 4 * focal0) {
            Rseed = coarse.Rrel;
            pairFocal = coarse.focal;
            pairLinearity = coarse.linearity;
          }
          const fine = polishPair(mc, Rseed, pairFocal, pairLinearity, { coordinateScale, iters: 8 });
          if (fine.Rrel.every((x) => isFinite(x)) && isFinite(fine.focal) && fine.focal > 0.25 * focal0 && fine.focal < 4 * focal0) {
            Rseed = fine.Rrel;
            pairFocal = fine.focal;
            pairLinearity = fine.linearity;
          }
        } catch { /* keep the RANSAC seed */ }
        let si = 0, sj = 0;
        const gi = shots[i].gray, gj = shots[j].gray;
        for (const t of inliers) {
          si += gi[Math.round(pa[t][1]) * shots[i].w + Math.round(pa[t][0])] || 0;
          sj += gj[Math.round(pb[t][1]) * shots[j].w + Math.round(pb[t][0])] || 0;
        }
        verified.push({ i, j, mc, inl: inliers.length, Ii: si / inliers.length / 255, Ij: sj / inliers.length / 255, H, Rseed, pairFocal, pairLinearity });
        if (isFinite(pairFocal)) focals.push(pairFocal);
        if (H) for (const f of focalFromHomography(H)) if (f > 0.3 * focal0 && f < 3 * focal0) focals.push(f);
      }
    }
    await tick();
  }
  log.push(`verified pairs: ${verified.length} (best raw matches ${bestRaw}, best inliers ${bestInl}, ${cand.length} pairs tried)`);
  if (verified.length === 0) return bail();

  // 4. focal length: median of the per-homography estimates
  let focal = focals.length >= 3 ? median(focals) : focal0;
  focal = clamp(focal, 0.5 * focal0, 2.0 * focal0);
  log.push(`focal ${(focal * coordinateScale).toFixed(1)}px (init ${(focal0 * coordinateScale).toFixed(1)}, ${focals.length} votes)`);

  // 4b. calibrate the lens: solve radial distortion k1 (and polish the focal)
  // by minimising total pairwise reprojection error. A phone lens is not a
  // pinhole, and the residual grows toward the frame edges - exactly where
  // seams land - so leaving k1 at 0 misaligns every seam and loses pairs.
  onProgress('optimizing', 0);
  const seedFor = (v, fl) => {
    const gyroSeed = matMul3(matT3(gyroR[v.j]), gyroR[v.i]); // cam_i ray -> cam_j ray
    if (v.Rseed) return v.Rseed;
    try {
      const hSeed = relRotFromHomography(v.H, fl);
      const off = Math.hypot(...logSO3(matMul3(hSeed, matT3(gyroSeed))));
      if (hSeed.every((x) => isFinite(x)) && off < 0.6) return hSeed;
    } catch { /* keep gyro seed */ }
    return gyroSeed;
  };
  const calibSet = verified.slice().sort((a, b) => b.inl - a.inl).slice(0, 12);
  const scoreCalib = (fl, kk, kk2, kk3, ll) => {
    let sse = 0, n = 0;
    for (const v of calibSet) {
      const r = refineRelRot(seedFor(v, fl), v.mc, fl, {
        k1: kk, k2: kk2, k3: kk3, linearity: ll, iters: 8,
        outlierFloor: 2.5 / coordinateScale,
      });
      if (r.inl >= 10 && r.rms < 20 / coordinateScale) { sse += r.rms * r.rms * r.inl; n += r.inl; }
    }
    return n ? Math.sqrt(sse / n) : 1e9;
  };
  let k1 = 0, k2 = 0, k3 = 0, linearity = 1;
  if (calibSet.length >= 3) {
    let best = scoreCalib(focal, 0, 0, 0, linearity);
    for (let round = 0; round < 2; round++) {
      const kStep = round === 0 ? 0.06 : 0.02;
      for (let i = -4; i <= 4; i++) {
        const kk = k1 + i * kStep;
        if (kk < -0.45 || kk > 0.2 || i === 0) continue;
        const sc = scoreCalib(focal, kk, k2, k3, linearity);
        if (sc < best) { best = sc; k1 = kk; }
      }
      const fStep = round === 0 ? 0.04 : 0.015;
      for (let i = -3; i <= 3; i++) {
        if (i === 0) continue;
        const fl = focal * (1 + i * fStep);
        if (fl < 0.5 * focal0 || fl > 2 * focal0) continue;
        const sc = scoreCalib(fl, k1, k2, k3, linearity);
        if (sc < best) { best = sc; focal = fl; }
      }
      const k2Step = round === 0 ? 0.03 : 0.01;
      for (let i = -3; i <= 3; i++) {
        if (i === 0) continue;
        const kk2 = k2 + i * k2Step;
        if (kk2 < -0.18 || kk2 > 0.18) continue;
        const sc = scoreCalib(focal, k1, kk2, k3, linearity);
        if (sc < best) { best = sc; k2 = kk2; }
      }
      const k3Step = round === 0 ? 0.01 : 0.003;
      for (let i = -2; i <= 2; i++) {
        if (i === 0) continue;
        const kk3 = k3 + i * k3Step;
        if (kk3 < -0.06 || kk3 > 0.06) continue;
        const sc = scoreCalib(focal, k1, k2, kk3, linearity);
        if (sc < best) { best = sc; k3 = kk3; }
      }
      for (const ll of [linearity - 0.5, linearity - 0.25, linearity - 0.1, linearity + 0.1, linearity + 0.25, linearity + 0.5]) {
        if (ll < -1.2 || ll > 1.8 || Math.abs(ll) < 1e-4) continue;
        const sc = scoreCalib(focal, k1, k2, k3, ll);
        if (sc < best) { best = sc; linearity = ll; }
      }
      await tick();
    }
    log.push(`lens: L ${linearity.toFixed(3)}, a ${k1.toFixed(5)}, b ${k2.toFixed(5)}, c ${k3.toFixed(5)}, focal ${(focal * coordinateScale).toFixed(1)} ` +
      `(${(focal / focal0).toFixed(3)}x), pair rms ${(best * coordinateScale).toFixed(2)}px`);
  }

  // 5. refine each pair's relative rotation with the calibrated lens
  const edges = [];
  for (let e = 0; e < verified.length; e++) {
    onProgress('optimizing', 0.2 + e / verified.length * 0.5);
    const v = verified[e];
    const { Rrel, rms, inl, matches } = refineRelRot(seedFor(v, focal), v.mc, focal, {
      k1, k2, k3, linearity, outlierFloor: 2.5 / coordinateScale,
    });
    // accept a pair if it aligns tightly, OR loosely but with lots of inliers
    // (wide-baseline / mild parallax pairs still anchor the graph)
    if (inl >= 10 && (rms < 4 / coordinateScale || (rms < 6.5 / coordinateScale && inl >= 25))) {
      edges.push({ i: v.i, j: v.j, Rrel, w: Math.min(inl, 120), mc: matches });
    }
    v.rms = rms;
    await tick();
  }
  log.push(`good edges: ${edges.length}`);
  log.push('pairs: ' + verified.map((v) => `${v.i}-${v.j}:${v.rms == null ? '9.0' : (v.rms * coordinateScale).toFixed(1)}/${v.inl}`).join(' '));
  if (edges.length === 0) return bail();

  // 6. Build the panorama the same way Panorama does: start from a pair, add
  // images through verified edges, and bundle-adjust the growing model after
  // each insertion. This is slower than one rotation average, but failure here
  // is more expensive than runtime.
  onProgress('optimizing', 0.7);
  const inc = incrementalBuild(N, edges, gyroR, shots.map(() => ({ w, h })), {
    focal, cx, cy, k1, k2, k3, linearity, coordinateScale,
  });
  let rotations = inc.rotations;
  focal = inc.focal; k1 = inc.k1; k2 = inc.k2; k3 = inc.k3; linearity = inc.linearity;
  const root = Array.from({ length: N }, (_, i) => inc.uf.find(i));
  const sizeOf = {};
  inc.built.forEach((ok, i) => { if (ok) sizeOf[root[i]] = (sizeOf[root[i]] || 0) + 1; });
  const connected = inc.built.map((ok, i) => ok && sizeOf[root[i]] > 1);
  const comps = Object.values(sizeOf).filter((n) => n > 1).sort((a, b) => b - a);
  log.push(`incremental BA: ${connected.filter(Boolean).length}/${N} in ${comps.length} clusters [${comps.join(',')}]`);
  const mainRoot = Object.keys(sizeOf).sort((a, b) => sizeOf[b] - sizeOf[a])[0];
  const reliable = inc.built.map((ok, i) => ok && String(root[i]) === mainRoot);
  log.push(`coherent frames: ${reliable.filter(Boolean).length}/${N}`);
  let cameras = inc.cameras;

  // 7. Final reprojection solve over every edge the incremental builder folded
  // into the panorama.
  const allPairs = inc.activeEdges.map((e) => ({
    i: e.i, j: e.j, m: e.mc,
  }));
  if (allPairs.length) {
    try {
      const ba = await globalBundleAdjust(shots.map(() => ({ w, h })), rotations, allPairs, {
        focal0: focal, initialCameras: cameras, cx, cy, k1, k2, k3, linearity, optimizeFocal: true, optimizeLens: true, optimizeDistortion: false,
        optimizePerFrame: true, optimizePerFrameCenter: false,
        minLinearity: -1.2, maxLinearity: 1.8,
        priorW: 0.08, huber: 5 / coordinateScale, iters: N > 24 ? 4 : 10,
      });
      if (ba.R.every((R) => R.every((v) => isFinite(v))) && isFinite(ba.focal)) {
        rotations = ba.R; focal = ba.focal; k1 = ba.k1; k2 = ba.k2; k3 = ba.k3; linearity = ba.linearity;
        cameras = ba.cameras;
        log.push(`global BA: focal ${(focal * coordinateScale).toFixed(1)}, L ${linearity.toFixed(3)}, a ${k1.toFixed(5)}, b ${k2.toFixed(5)}, c ${k3.toFixed(5)}`);
      }
    } catch { /* retain the complete incremental calibration */ }
  }
  rotations = anchorComponents(rotations, gyroR, root, mainRoot);
  log.push(`coverage: ${N}/${N} frames retained (${reliable.filter(Boolean).length} main, ${connected.filter((ok, i) => ok && !reliable[i]).length} secondary, ${connected.filter((ok) => !ok).length} sensor-only)`);
  onProgress('optimizing', 1);

  // 8. gain compensation over every matched pair
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

  // Convert normalized film calibration to source texture UVs and angular
  // distortion coefficients. Per-frame focal powers are applied in WebGL.
  const renderCameras = cameras.map((camera, i) => renderCamera(shots[i], camera, focal));
  return { rotations, focalScale: focal / focal0, k1: k1 * focal, k2: k2 * focal * focal, k3: k3 * focal * focal * focal,
    linearity, center: renderCameras[0].center, cameras: renderCameras, gains, connected, reliable, ok: true, log };
}

class UnionFind {
  constructor(n) { this.p = Array.from({ length: n }, (_, i) => i); }
  find(x) { while (this.p[x] !== x) { this.p[x] = this.p[this.p[x]]; x = this.p[x]; } return x; }
  union(a, b) { this.p[this.find(a)] = this.find(b); }
}
