// Synthetic end-to-end test of the feature-based stitcher (no camera/GPU).
// Render N perspective views of a procedural equirect scene at KNOWN rotations,
// feed the solver NOISY gyro poses + per-frame gain, check it recovers truth.

import { stitch } from './js/stitch.js';
import {
  quatFromAxisAngle, multiplyQuat, normalizeQuat, quatAngle,
} from './js/orientation.js';
import { qToR, matVec3, matMul3, matT3, logSO3, expSO3 } from './js/ba.js';

const DEG = Math.PI / 180;

// ---- procedural scene -------------------------------------------------------
function hash(ix, iy) {
  let h = (ix * 374761393 + iy * 668265263) | 0;
  h = ((h ^ (h >>> 13)) * 1274126177) | 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function vnoise(x, y) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const a = hash(ix, iy), b = hash(ix + 1, iy), c = hash(ix, iy + 1), d = hash(ix + 1, iy + 1);
  const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
// aperiodic multi-octave texture — corner-rich for FAST, no repeats to fool
// matching, band-limited so 2x supersampling removes aliasing.
function scene(lon, lat) {
  const x = lon * 30 + 12.3, y = lat * 30 - 7.7;
  let s = 0, amp = 1, f = 1;
  for (let k = 0; k < 5; k++) { s += amp * (vnoise(x * f + k * 13.1, y * f - k * 9.7) - 0.5); amp *= 0.62; f *= 2; }
  return Math.max(0, Math.min(255, 128 + s * 190));
}

// ---- perspective render (row-major R, camera looks -Z), 2x supersampled ---
function renderView(R, f, w, h) {
  const g = new Uint8ClampedArray(w * h);
  const cx = w / 2, cy = h / 2;
  const S = 2, fs = f * S;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let sy = 0; sy < S; sy++) for (let sx = 0; sx < S; sx++) {
        const px = (x * S + sx + 0.5) - cx * S, py = (y * S + sy + 0.5) - cy * S;
        const world = matVec3(R, [px / fs, py / fs, -1]);
        const n = Math.hypot(...world);
        acc += scene(Math.atan2(world[0] / n, -world[2] / n),
          Math.asin(Math.max(-1, Math.min(1, world[1] / n))));
      }
      g[y * w + x] = acc / (S * S);
    }
  }
  return g;
}

const qYawPitch = (yaw, pitch) => normalizeQuat(multiplyQuat(
  quatFromAxisAngle(0, 1, 0, yaw), quatFromAxisAngle(1, 0, 0, pitch)));
function randSmallQuat(rad) {
  const a = [Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1];
  const n = Math.hypot(...a) || 1;
  return quatFromAxisAngle(a[0] / n, a[1] / n, a[2] / n, (Math.random() * 2 - 1) * rad);
}

// ---- build frames --------------------------------------------------------------
const W = 480, H = 360, HFOV = 55 * DEG;
const F = (W / 2) / Math.tan(HFOV / 2);
const yaws = [-75, -45, -15, 15, 45, 75].map((d) => d * DEG);
const pitches = [-13 * DEG, 13 * DEG];

const truthQ = [];
for (const p of pitches) for (const yw of yaws) truthQ.push(qYawPitch(yw, p));

const GYRO_NOISE = 2.5 * DEG;
const shots = truthQ.map((qt) => {
  const noisy = normalizeQuat(multiplyQuat(randSmallQuat(GYRO_NOISE), qt));
  const gain = 0.85 + Math.random() * 0.3;
  const g = renderView(qToR(qt), F, W, H);
  for (let i = 0; i < g.length; i++) g[i] = Math.min(255, g[i] * gain);
  return { gray: g, w: W, h: H, quat: noisy, hfovDeg: 55, _R: qToR(qt), _gain: gain };
});

console.log(`${shots.length} frames, gyro noise ${(GYRO_NOISE / DEG).toFixed(1)}deg, true focal ${F.toFixed(1)}px`);

const t0 = Date.now();
const res = await stitch(shots, { onProgress: (s, f) => process.stdout.write(`\r${s} ${(f * 100) | 0}%    `) });
process.stdout.write('\n');
console.log('elapsed', ((Date.now() - t0) / 1000).toFixed(1) + 's');
res.log.forEach((l) => console.log('  ·', l));
console.log('ok:', res.ok, ' focalScale:', res.focalScale.toFixed(4), '(want ~1.000)');

let gyroErr = 0, estErr = 0, nConn = 0, worst = 0;
for (let i = 0; i < shots.length; i++) {
  const ge = quatAngle(shots[i].quat, truthQ[i]) / DEG;
  const ee = Math.hypot(...logSO3(matMul3(res.rotations[i], matT3(shots[i]._R)))) / DEG;
  gyroErr += ge;
  if (res.connected[i]) { estErr += ee; nConn++; worst = Math.max(worst, ee); }
  console.log(`  f${String(i).padStart(2)} ${res.connected[i] ? 'Y' : 'n'}  gyro ${ge.toFixed(2)}°  est ${ee.toFixed(2)}°  gain ${shots[i]._gain.toFixed(2)}→${res.gains[i].toFixed(2)}`);
}
const meanEst = estErr / Math.max(1, nConn), meanGyro = gyroErr / shots.length;
console.log(`\nmean gyro ${meanGyro.toFixed(2)}°  |  mean est(connected) ${meanEst.toFixed(2)}°  worst ${worst.toFixed(2)}°`);

// est error is measured against absolute truth; the solution lives in the
// gyro-anchored frame, so ~gyro-noise of global offset is expected. What must
// hold: everything connected, focal recovered, and no frame wildly worse.
const pass = res.ok && nConn >= shots.length - 1 && res.focalScale > 0.85 && res.focalScale < 1.18 && worst < 3.5;
console.log(pass ? '\nPASS ✅' : '\nFAIL ❌');
process.exit(pass ? 0 : 1);
