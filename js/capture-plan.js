const DEG = Math.PI / 180;

// Space captures from the real camera footprint. Reserve some FOV for lens
// calibration error, then keep roughly one third overlap in each direction.
export function capturePlan(tanX, tanY) {
  const hfov = 2 * Math.atan(tanX * 0.9) / DEG;
  const vfov = 2 * Math.atan(tanY * 0.9) / DEG;
  const divisions = Math.max(1, Math.ceil(90 / (vfov * 0.7)));
  const pitchStep = 90 / divisions;
  const rings = [{ pitch: 0 }];
  for (let i = 1; i < divisions; i++) rings.push({ pitch: i * pitchStep }, { pitch: -i * pitchStep });
  const targets = [];
  for (const { pitch } of rings) {
    const cp = Math.cos(pitch * DEG);
    const count = Math.max(4, Math.ceil(360 * cp / (hfov * 0.65)));
    const offset = pitch ? 180 / count : 0;
    for (let i = 0; i < count; i++) {
      const yaw = (i * 360 / count + offset) * DEG;
      targets.push({ dir: [cp * Math.sin(yaw), Math.sin(pitch * DEG), -cp * Math.cos(yaw)], done: false, progress: 0 });
    }
  }
  targets.push(
    { dir: [0, 1, 0], cap: 'zenith', done: false, progress: 0 },
    { dir: [0, -1, 0], cap: 'nadir', done: false, progress: 0 },
  );
  // Guide dots control deliberate stops; intermediate sweep photos supply
  // the denser overlap needed by weakly textured real scenes.
  return { targets, sweepStep: Math.max(5, Math.min(8, hfov * 0.18)) * DEG };
}

// Sweep frames are expendable; a completed guide dot must keep its photograph.
// Return -1 when every frame is deliberate coverage, even if that means a
// narrow-FOV capture needs more frames than the usual budget.
export function redundantShot(shots, angle) {
  let best = -1, gap = Infinity;
  for (let i = 0; i < shots.length; i++) for (let j = i + 1; j < shots.length; j++) {
    const a = shots[i], b = shots[j];
    const keepA = a.guided || a.cap, keepB = b.guided || b.cap;
    if (keepA && keepB) continue;
    const distance = angle(a.quat, b.quat);
    if (distance < gap) {
      gap = distance;
      best = keepA ? j : keepB ? i : (a.feat <= b.feat ? i : j);
    }
  }
  return best;
}
