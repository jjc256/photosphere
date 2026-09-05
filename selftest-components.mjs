// Disconnected feature groups must retain their internally solved geometry
// while sharing the main group's sensor reference. No camera/GPU required.
import assert from 'node:assert/strict';
import { anchorComponents, expSO3, matMul3, matT3 } from './js/ba.js';
import { projectionRadiusLimit } from './js/camera-geometry.js';

const groups = [0, 0, 0, 0, 4, 4, 4, 7];
const truth = groups.map((_, i) => expSO3([0.15 * i, 0.6 * i, -0.1 * i]));
const main = expSO3([0.08, -0.12, 0.04]);
const secondary = expSO3([-0.2, 0.3, 0.1]);
const sensor = expSO3([0.1, -0.3, -0.04]);
const gyro = truth.map((R) => matMul3(sensor, R));
const solved = truth.map((R, i) => matMul3(i < 4 ? main : i < 7 ? secondary : sensor, R));
// A single corrupt IMU reading must not rotate every fallback image with it.
gyro[2] = matMul3(expSO3([0.5, 0.8, 0.3]), gyro[2]);
const anchored = anchorComponents(solved, gyro, groups, 0);
const error = (A, B) => Math.max(...A.map((v, i) => Math.abs(v - B[i])));
anchored.forEach((R, i) => assert.ok(error(R, matMul3(main, truth[i])) < 1e-10, `incorrect global placement for frame ${i}`));
for (let i = 0; i < 4; i++) assert.deepEqual(anchored[i], solved[i], 'main component changed');
for (const [i, j] of [[0, 3], [4, 6]]) {
  assert.ok(error(matMul3(matT3(anchored[i]), anchored[j]), matMul3(matT3(solved[i]), solved[j])) < 1e-10, 'component alignment was broken');
}
assert.deepEqual(anchorComponents([], [], [], 0), []);
assert.ok(Math.abs(projectionRadiusLimit(0.3, 0.4, [0.5, 0.5], 0, 0, 0, 0) - 0.5) < 1e-12);
// The cubic turns at r=.2 and becomes positive/increasing again beyond 1/3.
// Choosing the later inverse root used to create circular duplicate images.
assert.ok(Math.abs(projectionRadiusLimit(0.1, 0.1, [0.5, 0.5], 0, -4, 5, 0) - 0.2) < 1e-10);
const quarticLimit = projectionRadiusLimit(0.2, 0.2, [0.5, 0.5], 0, -3, 0, 1);
assert.ok(quarticLimit < 0.2 && Math.abs(1 - 6 * quarticLimit + 4 * quarticLimit ** 3) < 1e-10);
const normalLimit = projectionRadiusLimit(0.2, 0.2, [0.5, 0.5], 0, 0, -1, 0);
assert.ok(normalLimit < 1 / Math.sqrt(3) && Math.abs(normalLimit - normalLimit ** 3 - Math.hypot(0.2, 0.2)) < 1e-10);
console.log('PASS: secondary groups and isolated frames retain coverage in the main sensor reference');
console.log('PASS: lens coverage stops before radial distortion folds');
