import assert from 'node:assert/strict';
import { seamLabels } from './js/seams.js';
const w = 80, h = 48;
const warps = [0, 1].map((k) => {
  const pixels = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = k ? 20 : 0; x < (k ? w : 60); x++) {
    const object = x >= 34 + k * 5 && x <= 43 + k * 5 && y > 8 && y < 38;
    const value = object ? 220 : 80;
    pixels.set([value, value, value, 255], (y * w + x) * 4);
  }
  return pixels;
});
const gains = [[1, 1, 1], [1, 1, 1]];
const labels = seamLabels(warps, w, h, gains, [true, true]);
let cuts = 0;
for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
  const p = y * w + x;
  assert.ok(labels[p] >= 0 && warps[labels[p]][4 * p + 3], 'coverage lost or invalid source selected');
  // Ignore the longitude wrap here: these two views have no overlap there.
  for (const q of [x + 1 < w ? p + 1 : -1, y + 1 < h ? p + w : -1]) {
    if (q < 0 || labels[p] === labels[q] || !warps[0][p * 4 + 3] || !warps[1][p * 4 + 3] || !warps[0][q * 4 + 3] || !warps[1][q * 4 + 3]) continue;
    cuts++;
    assert.equal(warps[0][p * 4], warps[1][p * 4], 'seam crossed a displaced object');
    assert.equal(warps[0][q * 4], warps[1][q * 4], 'seam crossed a displaced object');
  }
}
assert.ok(cuts >= h, 'fixture did not exercise a joining seam');
// Put the same overlapping object across longitude zero. The cylindrical
// graph must apply the same boundary cost there as anywhere else.
const wrapped = warps.map((frame) => {
  const out = new Uint8Array(frame.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const from = (y * w + x) * 4, to = (y * w + (x + w / 2) % w) * 4;
    out.set(frame.subarray(from, from + 4), to);
  }
  return out;
});
const wrappedLabels = seamLabels(wrapped, w, h, gains, [true, true]);
for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
  const p = y * w + x, q = y * w + (x + 1) % w;
  if (wrappedLabels[p] === wrappedLabels[q]) continue;
  for (const r of [p, q]) if (wrapped[0][r * 4 + 3] && wrapped[1][r * 4 + 3]) {
    assert.equal(wrapped[0][r * 4], wrapped[1][r * 4], 'longitude seam cut through the displaced object');
  }
}
const fallback = seamLabels(warps, w, h, gains, [true, false]);
for (let p = 0; p < fallback.length; p++) if (warps[0][p * 4 + 3]) assert.equal(fallback[p], 0, 'sensor-only frame displaced a verified source');
const blank = new Uint8Array(w * h * 4);
assert.ok(seamLabels([blank], w, h, [[1, 1, 1]], [true]).every((v) => v === -1));
assert.ok(seamLabels([warps[0]], w, h, [[1, 1, 1]], [true]).every((v, p) => v === (warps[0][p * 4 + 3] ? 0 : -1)));
console.log(`PASS: ${cuts} seam edges avoid the displaced object; coverage and verified-source priority preserved`);

// Four overlapping views can leave a detour from an earlier insertion even
// after a better source becomes available. Refinement must actually remove
// that detour without creating a new cut through the displaced object.
const rw = 96, rh = 48;
const sequence = Array.from({ length: 4 }, (_, k) => {
  const pixels = new Uint8Array(rw * rh * 4);
  for (let y = 0; y < rh; y++) for (let x = k * 14; x < Math.min(rw, k * 14 + 54); x++) {
    const object = x > 33 + k * 5 && x < 48 + k * 5 && y > 10 && y < 36;
    const value = object ? 180 : 80 + Math.floor(y / 6) * 2 + Math.floor(x / 5);
    pixels.set([value, value, value, 255], (y * rw + x) * 4);
  }
  return pixels;
});
const sequenceGains = sequence.map(() => [1, 1, 1]), trusted = sequence.map(() => true);
const firstPass = seamLabels(sequence, rw, rh, sequenceGains, trusted, { refine: false });
let diagnostics;
const refined = seamLabels(sequence, rw, rh, sequenceGains, trusted, { onDiagnostics: (d) => { diagnostics = d; } });
const boundaryLength = (labels) => {
  let count = 0;
  for (let p = 0; p < labels.length; p++) for (const q of [Math.floor(p / rw) * rw + (p % rw + 1) % rw, p + rw]) {
    if (q < labels.length && labels[p] >= 0 && labels[q] >= 0 && labels[p] !== labels[q]) count++;
  }
  return count;
};
assert.ok(boundaryLength(refined) < boundaryLength(firstPass), 'refinement retained an unnecessary seam detour');
assert.ok(diagnostics.acceptedMoves > 0);
assert.equal(diagnostics.after.mismatch, 0, 'refinement introduced a photometric cut');
for (let p = 0; p < refined.length; p++) assert.ok(refined[p] >= 0 && sequence[refined[p]][p * 4 + 3], 'refinement removed coverage');
console.log('PASS: refinement shortens an existing seam without increasing source disagreement');
