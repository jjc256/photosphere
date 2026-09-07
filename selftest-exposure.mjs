import assert from 'node:assert/strict';
import { overlapExposure } from './js/exposure.js';
import { distributedMatches, cycleSupportedPairs } from './js/stitch.js';
import { ransacGeneralizedRotation, expSO3, matVec3, matMul3, matT3, logSO3 } from './js/ba.js';

const w = 96, h = 48;
const exposures = [[1, 1, 1], [0.7, 0.75, 0.8], [1.25, 1.2, 1.15]];
const warps = exposures.map((exposure, k) => {
  const rgba = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = k * 18; x < Math.min(w, k * 18 + 60); x++) {
    const p = (y * w + x) * 4;
    for (let c = 0; c < 3; c++) rgba[p + c] = (80 + 0.5 * x + 0.3 * y + c * 5) * exposure[c];
    rgba[p + 3] = 255;
  }
  return rgba;
});
// A moving dark patch in one overlap must not determine its exposure.
for (let y = 10; y < 16; y++) for (let x = 24; x < 36; x++) for (let c = 0; c < 3; c++) warps[1][(y * w + x) * 4 + c] = 20;
const gains = overlapExposure(warps, w, h);
for (let k = 1; k < gains.length; k++) for (let c = 0; c < 3; c++) {
  assert.ok(Math.abs(gains[k][c] * exposures[k][c] / gains[0][c] - 1) < 0.015, 'overlap exposure/WB was not recovered');
}
const blank = new Uint8Array(w * h * 4);
assert.deepEqual(overlapExposure([blank], w, h), [[1, 1, 1]]);
console.log('PASS: overlap-only exposure and white balance recovered within 1.5%, despite an outlier patch');

const f = 0.9, truth = expSO3([0.04, 0.2, -0.03]);
const matches = [];
for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) {
  const u = (x - 1.5) * 0.09, v = (y - 1.5) * 0.08;
  const d = matVec3(truth, [u / f, v / f, -1]);
  matches.push([u, v, f * d[0] / -d[2], f * d[1] / -d[2]]);
}
assert.ok(distributedMatches(matches));
assert.equal(distributedMatches(matches.map((m) => [m[0], 0, m[2], 0])), false);
const seed = matMul3(expSO3([0.01, -0.01, 0.01]), truth);
const fit = ransacGeneralizedRotation(matches, f, seed, { minMatches: 12, minInliers: 12, iters: 40, thresh: 3 / 512 });
assert.ok(fit.inliers.length === matches.length);
assert.ok(Math.hypot(...logSO3(matMul3(fit.Rrel, matT3(truth)))) < 0.001);
const falseMatches = matches.map((m, i) => [...m.slice(0, 2), ...matches[(i * 7 + 3) % matches.length].slice(2)]);
assert.ok(ransacGeneralizedRotation(falseMatches, f, seed, { minMatches: 12, minInliers: 12, iters: 40, thresh: 3 / 512 }).inliers.length < 12);
console.log('PASS: small distributed overlap is recovered; collinear and incorrect matches are rejected');
const ra = expSO3([0.03, 0.1, 0.02]), rb = expSO3([0.02, 0.2, -0.03]);
const link = { i: 1, j: 2, Rrel: matMul3(rb, matT3(ra)) };
const candidates = [{ i: 0, j: 1, Rrel: ra }, { i: 0, j: 2, Rrel: rb }];
assert.equal(cycleSupportedPairs([link], candidates).length, 2);
assert.equal(cycleSupportedPairs([link], [candidates[0], { ...candidates[1], Rrel: matMul3(expSO3([0.1, 0, 0]), rb) }]).length, 0);
console.log('PASS: low-ratio matches require a consistent three-image cycle');
