// Replay a photosphere-debug-*.json bundle from a device and report why the
// stitch failed.  Usage:  node debugreplay.mjs photosphere-debug-XXXX.json
import fs from 'node:fs';
import { stitch } from './js/stitch.js';
import { detectAndDescribe } from './js/orb.js';

const file = process.argv[2];
if (!file) { console.error('usage: node debugreplay.mjs <bundle.json>'); process.exit(1); }
const d = JSON.parse(fs.readFileSync(file, 'utf8'));

console.log('version   ', d.version);
console.log('device    ', d.ua);
console.log('hfovDeg   ', d.hfovDeg, ' video', d.videoWH);
console.log('frames    ', d.shots.length);
console.log('prev log  ', (d.stitchLog || []).join(' | '));
console.log('');

const shots = d.shots.map((s) => ({
  gray: Uint8ClampedArray.from(Buffer.from(s.grayB64, 'base64')),
  w: s.gw, h: s.gh, quat: s.quat, hfovDeg: s.hfovDeg ?? d.hfovDeg,
}));

console.log('per-frame: size  meanGray min..max  stddev  ORBfeatures');
for (let i = 0; i < shots.length; i++) {
  const g = shots[i].gray;
  let sum = 0, mn = 255, mx = 0;
  for (const v of g) { sum += v; if (v < mn) mn = v; if (v > mx) mx = v; }
  const mean = sum / g.length;
  let sd = 0;
  for (const v of g) sd += (v - mean) ** 2;
  sd = Math.sqrt(sd / g.length);
  const f = detectAndDescribe(shots[i].gray, shots[i].w, shots[i].h,
    { fastThresh: 18, maxFeatures: 900, minKeypoints: 120 });
  console.log(
    `  f${String(i).padStart(2)}  ${shots[i].w}x${shots[i].h}  ` +
    `${mean.toFixed(0).padStart(3)}  ${String(mn).padStart(3)}..${String(mx).padEnd(3)}  ` +
    `${sd.toFixed(1).padStart(5)}  ${f.kps.length}`);
}
console.log('');

const res = await stitch(shots, { onProgress: () => {} });
res.log.forEach((l) => console.log('  ', l));
console.log('\nconnected', res.connected.filter(Boolean).length + '/' + shots.length,
  ' focalScale', res.focalScale.toFixed(3), ' ok', res.ok);
