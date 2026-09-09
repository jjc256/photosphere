import assert from 'node:assert/strict';
import { capturePlan } from './js/capture-plan.js';
const DEG = Math.PI / 180;
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
function check(hfov, aspect, lensMargin = 1) {
  const tx = Math.tan(hfov * DEG / 2), ty = tx * aspect;
  const plan = capturePlan(tx, ty);
  const cameras = plan.targets.map(({ dir: d, cap }) => {
    const yaw = cap ? Math.PI / 4 : Math.atan2(d[0], -d[2]);
    const pitch = Math.asin(d[1]);
    return { d, right: [Math.cos(yaw), 0, Math.sin(yaw)], up: [-Math.sin(pitch) * Math.sin(yaw), Math.cos(pitch), Math.sin(pitch) * Math.cos(yaw)] };
  });
  let holes = 0, overlap = 0, samples = 0;
  for (let lat = -89; lat <= 89; lat += 2) for (let lon = -180; lon < 180; lon += 2) {
    const p = lat * DEG, y = lon * DEG;
    const d = [Math.cos(p) * Math.sin(y), Math.sin(p), -Math.cos(p) * Math.cos(y)];
    let count = 0;
    for (const c of cameras) {
      const z = dot(c.d, d);
      if (z > 0 && Math.abs(dot(c.right, d)) < tx * lensMargin * z && Math.abs(dot(c.up, d)) < ty * lensMargin * z) count++;
    }
    holes += count === 0; overlap += count > 1; samples++;
  }
  assert.equal(holes, 0, `${hfov}deg aspect=${aspect} has coverage holes`);
  assert.ok(overlap / samples > 0.5, 'too little shared coverage for matching');
  assert.equal(plan.targets.filter((t) => t.cap).length, 2);
  console.log(`${hfov}deg aspect=${aspect.toFixed(2)}: ${plan.targets.length} dots, ${(100 * overlap / samples).toFixed(0)}% shared sphere samples, no holes`);
  return plan;
}
const portrait = check(50, 16 / 9, 0.85);
assert.equal(portrait.targets.length, 33);
assert.equal(new Set(portrait.targets.filter((t) => !t.cap).map((t) => t.dir[1])).size, 3);
assert.ok(portrait.sweepStep >= 5 * DEG && portrait.sweepStep <= 8 * DEG);
for (const fov of [35, 65, 90, 110]) for (const aspect of [16 / 9, 9 / 16, 1]) check(fov, aspect, 0.9);
console.log('PASS: adaptive rings cover the sphere with overlap and reduce default portrait dots from 64 to 33');

const { redundantShot } = await import('./js/capture-plan.js');
const angle = (a, b) => Math.abs(a - b);
const frames = [{ quat: 0, feat: 1, guided: true }, { quat: 0.01, feat: 50, guided: true }, { quat: 10, feat: 100 }];
assert.equal(redundantShot(frames, angle), 2, 'a pair of close guide captures displaced deliberate coverage');
assert.equal(redundantShot(frames.slice(0, 2), angle), -1, 'all-deliberate capture must retain its dots');
assert.equal(redundantShot([{ quat: 0, feat: 1, cap: 'zenith' }, { quat: 0.1, feat: 100 }], angle), 1);
assert.equal(redundantShot([{ quat: 0, feat: 10 }, { quat: 0.1, feat: 100 }], angle), 0);
console.log('PASS: guide and pole captures survive eviction; redundant sweep frames are removed first');
