import assert from 'node:assert/strict';
import { localAlignment } from './js/local-alignment.js';
const w = 128, h = 64;
let seed = 42;
const texture = Array.from({ length: w * h }, () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return 30 + (seed >>> 25);
});
function photo(dx, dy, gain = 1, offset = 0) {
  const out = new Uint8Array(w * h * 4);
  for (let y = 8; y < h - 8; y++) for (let x = 8; x < w - 8; x++) {
    const value = texture[(y + dy) * w + x + dx] * gain + offset;
    out.set([value, value, value, 255], (y * w + x) * 4);
  }
  return out;
}
const a = photo(0, 0);
const shifted = localAlignment([a, photo(-3, 2, 1.2, 12)], w, h);
assert.ok(shifted.diagnostics.matches >= 20, 'failed to register exposure-shifted overlap');
assert.ok(shifted.diagnostics.afterRms < .3, 'known displacement was not corrected');
const mean = (field, axis) => field.reduce((s, v, k) => s + (k % 2 === axis ? v : 0), 0) / (field.length / 2);
assert.ok(Math.abs((mean(shifted.fields[1], 0) - mean(shifted.fields[0], 0)) * w - 3) < .3);
assert.ok(Math.abs((mean(shifted.fields[1], 1) - mean(shifted.fields[0], 1)) * h + 2) < .3);
for (const result of [localAlignment([a, a], w, h), localAlignment([new Uint8Array(w*h*4), a], w, h)]) {
  assert.ok(result.fields.every(f => f.every(v => v === 0)), 'coherent/unsupported pixels must not move');
}
const flat = new Uint8Array(w*h*4).fill(255);
assert.equal(localAlignment([flat, flat], w, h).diagnostics.matches, 0, 'blank walls must not create evidence');
for (const field of shifted.fields) {
  for (let y = 0; y < shifted.height; y++) for (let x = 0; x < shifted.width; x++) {
    const k = y * shifted.width + x;
    const next = y * shifted.width + (x + 1) % shifted.width;
    assert.ok(Math.hypot((field[k*2] - field[next*2]) * w, (field[k*2+1] - field[next*2+1]) * h) <= w / shifted.width * .25 + 1e-5, 'mesh can fold across longitude wrap');
  }
}
console.log('PASS: local alignment corrects known parallax despite exposure differences, preserves identity/blank regions, and bounds mesh deformation');
