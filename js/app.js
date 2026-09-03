import { PanoEngine } from './pano.js';
import { stitch } from './stitch.js';
import { buildGPanoXMP, embedMetadata } from './xmp.js';
import { buildExifSegment } from './exif.js';
import {
  DEG, deviceQuat, quatToMat3, yawPitchToMat3, quatFromAxisAngle,
  multiplyQuat, normalizeQuat, quatAngle, forwardDir,
} from './orientation.js';

const APP_VERSION = '0.5.1';

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ---- captured-frame budget --------------------------------------------------
const MAX_SHOTS = 32;   // memory-bounded; ImageData is kept for the final blend
const CAP_LONG = 1024;  // long side kept for compositing
const GRAY_LONG = 512;  // long side used for feature detection

const state = {
  stream: null,
  engine: null,
  hasOrientation: false,
  lastOrientQuat: [0, 0, 0, 1],
  quat: [0, 0, 0, 1],
  R: yawPitchToMat3(0, 0),
  lastCapQuat: null,
  headingDeg: null,
  geo: null,          // { lat, lon, alt, acc } — embedded only in the downloaded file
  geoWatch: null,
  hfovDeg: 55,
  vidRot: 0,
  autoCap: true,
  shots: [],          // { imgData, w, h, gray, gw, gh, sharp, quat, hfovDeg }
  targets: [],        // guided-capture dots: { dir, done, progress }
  R0: null,           // camera->world at capture start (target lattice reference)
  activeTarget: -1,
  raf: 0,
  running: false,
  lastPreview: 0,
  lastClassPass: 0,
  speed: 0,
  speedQuat: null,
  speedT: 0,
  sim: { active: false, drag: false, yaw: 0, pitch: 0, lx: 0, ly: 0 },
  view: { yaw: 0, pitch: 0, fov: 72 * DEG, drag: false, lx: 0, ly: 0, pinch: 0 },
  viewFlat: false,
};

const video = $('cam');
let glCanvas = $('gl'); // single WebGL canvas, reparented between screens

// ---- screen switching ---------------------------------------------------
function showScreen(name) {
  for (const s of document.querySelectorAll('.screen')) s.classList.remove('is-active');
  $('screen-' + name).classList.add('is-active');
}

function toast(msg, ms = 2600) {
  const t = $('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}

function note(msg, isErr) {
  const n = $('start-note');
  n.textContent = msg || '';
  n.classList.toggle('err', !!isErr);
}

// ---- orientation ------------------------------------------------------------
function onOrient(e) {
  if (e.alpha === null && e.beta === null && e.gamma === null) return;
  state.hasOrientation = true;
  const screenAngle =
    ((screen.orientation && screen.orientation.angle) || window.orientation || 0) * DEG;
  state.lastOrientQuat = deviceQuat(
    (e.alpha || 0) * DEG, (e.beta || 0) * DEG, (e.gamma || 0) * DEG, screenAngle);
  if (typeof e.webkitCompassHeading === 'number' && !Number.isNaN(e.webkitCompassHeading) &&
      state.headingDeg === null) {
    state.headingDeg = e.webkitCompassHeading;
  }
}

function updateOrientation() {
  if (state.hasOrientation) {
    state.quat = state.lastOrientQuat;
    state.R = quatToMat3(state.quat);
  } else {
    const { yaw, pitch } = state.sim;
    state.quat = normalizeQuat(multiplyQuat(
      quatFromAxisAngle(0, 1, 0, yaw), quatFromAxisAngle(1, 0, 0, pitch)));
    state.R = yawPitchToMat3(yaw, pitch);
  }
}

// ---- location (for Google Maps / Street View) -----------------------------
function updateGeoStatus(kind, detail) {
  const el = $('geo-status');
  el.classList.remove('ok', 'err');
  if (kind === 'locating') { el.textContent = 'Getting location…'; return; }
  if (kind === 'error') {
    el.classList.add('err');
    el.textContent = 'Location unavailable' + (detail ? ' — ' + detail : '');
    return;
  }
  if (state.geo) {
    el.classList.add('ok');
    el.textContent =
      `\u{1F4CD} ${state.geo.lat.toFixed(5)}, ${state.geo.lon.toFixed(5)}` +
      (Number.isFinite(state.geo.acc) ? ` ±${Math.round(state.geo.acc)} m` : '') +
      ' · embedded only in the file you download';
  } else {
    el.textContent = 'Location off';
  }
}

function setGeoEnabled(on) {
  if (!('geolocation' in navigator)) {
    if (on) updateGeoStatus('error', 'no geolocation API');
    return;
  }
  if (on) {
    if (state.geoWatch != null) return;
    updateGeoStatus('locating');
    state.geoWatch = navigator.geolocation.watchPosition(
      (p) => {
        state.geo = {
          lat: p.coords.latitude, lon: p.coords.longitude,
          alt: p.coords.altitude, acc: p.coords.accuracy,
        };
        updateGeoStatus();
      },
      (err) => { updateGeoStatus('error', err.message); },
      { enableHighAccuracy: true, maximumAge: 15000, timeout: 25000 });
  } else {
    if (state.geoWatch != null) navigator.geolocation.clearWatch(state.geoWatch);
    state.geoWatch = null;
    state.geo = null;
    updateGeoStatus();
  }
}

// ---- camera FOV helpers --------------------------------------------------
function fovTangents() {
  const vw = video.videoWidth || 16;
  const vh = video.videoHeight || 9;
  const effW = state.vidRot % 2 ? vh : vw;
  const effH = state.vidRot % 2 ? vw : vh;
  const tanX = Math.tan((state.hfovDeg * DEG) / 2);
  const tanY = tanX * (effH / effW);
  return { tanX, tanY };
}

// ---- capture reset --------------------------------------------------------
function resetCoverage() {
  state.shots = [];
  state.lastCapQuat = null;
  state.steadyFrames = 0;
  state.R0 = null;
  state._r0Deadline = performance.now() + 1600;
  buildTargets();
  $('coverage').textContent = '0/0';
}

// ---- capture --------------------------------------------------------------
const _capCanvas = document.createElement('canvas');

function grabFrame(long, vw, vh) {
  const s = Math.min(1, long / Math.max(vw, vh));
  const w = Math.max(8, Math.round(vw * s)), h = Math.max(8, Math.round(vh * s));
  _capCanvas.width = w; _capCanvas.height = h;
  const ctx = _capCanvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, w, h);
  return { w, h, data: ctx.getImageData(0, 0, w, h) };
}

// mean absolute gradient of a luma buffer — a cheap blur/sharpness proxy
function sharpness(gray, w, h) {
  let sum = 0, n = 0;
  for (let y = 1; y < h - 1; y += 2) {
    for (let x = 1; x < w - 1; x += 2) {
      const i = y * w + x;
      sum += Math.abs(gray[i] - gray[i + 1]) + Math.abs(gray[i] - gray[i + w]);
      n += 2;
    }
  }
  return n ? sum / n : 0;
}

function stashShot(manual) {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw) return;
  const sm = grabFrame(GRAY_LONG, vw, vh);
  const gray = new Uint8ClampedArray(sm.w * sm.h);
  const d = sm.data.data;
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = (d[p] * 77 + d[p + 1] * 150 + d[p + 2] * 29) >> 8;
  }
  const sharp = sharpness(gray, sm.w, sm.h); // kept only for near-duplicate eviction
  const big = grabFrame(CAP_LONG, vw, vh);
  state.shots.push({
    imgData: big.data, w: big.w, h: big.h,
    gray, gw: sm.w, gh: sm.h, sharp,
    quat: state.quat.slice(), hfovDeg: state.hfovDeg,
  });
  if (state.shots.length > MAX_SHOTS) {
    // drop the weaker of the closest-together pair (blurrier frame loses)
    let bi = 0, bd = Infinity;
    for (let i = 0; i < state.shots.length; i++) {
      for (let j = i + 1; j < state.shots.length; j++) {
        const a = quatAngle(state.shots[i].quat, state.shots[j].quat);
        if (a < bd) { bd = a; bi = state.shots[i].sharp <= state.shots[j].sharp ? i : j; }
      }
    }
    state.shots.splice(bi, 1);
  }
  return true;
}

function doCapture(manual) {
  if (!video.videoWidth) return false;
  updateOrientation(); // capture against the freshest pose
  const { tanX, tanY } = fovTangents();
  const ok = stashShot(manual);                                   // for the real stitch on Done
  if (!ok) return false;
  state.engine.splat(video, state.R, tanX, tanY, state.vidRot);   // live guide preview
  state.lastCapQuat = state.quat;
  const s = $('btn-shutter');
  s.classList.remove('flash');
  void s.offsetWidth;
  s.classList.add('flash');
  if (navigator.vibrate) navigator.vibrate(manual ? 25 : 12);
  return true;
}

// ---- guided capture targets (aim → hold → ring fills → snap) --------------
// A lattice of dots on the sphere, spaced ~60% of the field of view so
// neighbouring shots overlap enough to stitch. Sized to the current FOV.
function buildTargets() {
  const fov = state.hfovDeg || 55;
  // space dots ~80% of the FOV apart -> ~20% overlap between neighbours
  const eqN = clamp(Math.round(360 / (fov * 0.82)), 5, 10);
  const midP = fov * 0.55, highP = Math.min(64, fov * 1.15);
  const rings = [
    { p: 0, n: eqN },
    { p: midP, n: Math.max(5, Math.round(eqN * 0.8)) },
    { p: -midP, n: Math.max(5, Math.round(eqN * 0.8)) },
    { p: highP, n: Math.max(3, Math.round(eqN * 0.42)) },
    { p: -highP, n: Math.max(3, Math.round(eqN * 0.42)) },
    { p: 82, n: 3 },   // near-zenith cap (the last ~8° is filled in software)
    { p: -82, n: 3 },  // near-nadir cap
  ];
  const T = [];
  rings.forEach((r, ri) => {
    const off = ri % 2 ? 180 / r.n : 0; // stagger alternate rings
    for (let i = 0; i < r.n; i++) {
      const y = (i * 360 / r.n + off) * DEG, p = r.p * DEG, cp = Math.cos(p);
      T.push({ dir: [cp * Math.sin(y), Math.sin(p), -cp * Math.cos(y)], done: false, progress: 0 });
    }
  });
  state.targets = T;
  state.activeTarget = -1;
}

// world dir of a ref-frame target: R0 (camera->world, column-major) · v
function refToWorld(R0, v) {
  return [
    R0[0] * v[0] + R0[3] * v[1] + R0[6] * v[2],
    R0[1] * v[0] + R0[4] * v[1] + R0[7] * v[2],
    R0[2] * v[0] + R0[5] * v[1] + R0[8] * v[2],
  ];
}
// world dir -> camera space (Rᵀ·w for a column-major camera->world R)
function worldToCam(R, w) {
  return [
    R[0] * w[0] + R[1] * w[1] + R[2] * w[2],
    R[3] * w[0] + R[4] * w[1] + R[5] * w[2],
    R[6] * w[0] + R[7] * w[1] + R[8] * w[2],
  ];
}

function updateGuidance(now) {
  if (state.speedQuat) {
    const dt = (now - state.speedT) / 1000 || 1 / 60;
    state.speed = quatAngle(state.quat, state.speedQuat) / dt;
  }
  state.speedQuat = state.quat;
  state.speedT = now;
  // being near-still for a few frames IS the anti-blur guard (like other apps)
  state.steadyFrames = state.speed < 0.16 ? (state.steadyFrames || 0) + 1 : 0; // ≈9°/s
  const steady = state.steadyFrames >= 4;

  if (!state.R0 && (state.hasOrientation || state.sim.active || now > state._r0Deadline)) {
    state.R0 = state.R.slice();
  }

  const dt = Math.min(0.05, (now - (state._guideT || now)) / 1000);
  state._guideT = now;
  const hint = $('hint');

  if (!state.R0 || !state.targets.length) { hint.textContent = 'Getting your bearings…'; return; }
  if (!state.autoCap) { hint.textContent = 'Tap the shutter to grab a frame'; return; }

  const aim = forwardDir(state.R);
  let act = -1, actAng = Infinity;
  for (let i = 0; i < state.targets.length; i++) {
    const t = state.targets[i];
    if (t.done) continue;
    t._w = refToWorld(state.R0, t.dir);
    t._ang = Math.acos(clamp(aim[0] * t._w[0] + aim[1] * t._w[1] + aim[2] * t._w[2], -1, 1));
    if (t._ang < actAng) { actAng = t._ang; act = i; }
  }
  // hysteresis: don't hop to a new active dot unless it's clearly closer
  const prev = state.activeTarget;
  if (prev >= 0 && prev < state.targets.length && !state.targets[prev].done &&
      state.targets[prev]._ang < actAng + 4 * DEG) {
    act = prev; actAng = state.targets[prev]._ang;
  }
  state.activeTarget = act;

  const CONE = 8 * DEG, FILL = 0.7;
  for (let i = 0; i < state.targets.length; i++) {
    const t = state.targets[i];
    if (t.done) continue;
    const on = i === act && t._ang < CONE && steady;
    t.progress = clamp(t.progress + (on ? dt / FILL : -dt / 0.3), 0, 1);
    if (t.progress >= 1 && on && doCapture(false)) { t.done = true; t.progress = 1; }
  }

  const done = state.targets.filter((t) => t.done).length;
  $('coverage').textContent = `${done}/${state.targets.length}`;
  if (done === state.targets.length) hint.textContent = 'All dots captured — tap Done';
  else if (act >= 0 && state.targets[act]._ang < CONE) hint.textContent = steady ? 'Hold…' : 'Hold still';
  else hint.textContent = 'Aim at the nearest dot';
}

function drawGuide() {
  const cv = $('guide');
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const cw = Math.round(cv.clientWidth * dpr), ch = Math.round(cv.clientHeight * dpr);
  if (cv.width !== cw || cv.height !== ch) { cv.width = cw; cv.height = ch; }
  const g = cv.getContext('2d');
  g.clearRect(0, 0, cw, ch);
  if (!state.R0 || !state.targets.length) return;

  const { tanX, tanY } = fovTangents();
  const aim = forwardDir(state.R);
  const project = (w) => {
    const c = worldToCam(state.R, w);
    if (c[2] >= -1e-3) return null;
    const sx = (c[0] / -c[2]) / tanX, sy = (c[1] / -c[2]) / tanY;
    return { sx, sy, x: (sx * 0.5 + 0.5) * cw, y: (0.5 - sy * 0.5) * ch };
  };

  for (let i = 0; i < state.targets.length; i++) {
    const t = state.targets[i];
    const w = refToWorld(state.R0, t.dir);
    const active = i === state.activeTarget;
    // only show dots near where you're looking (like other apps)
    const align = aim[0] * w[0] + aim[1] * w[1] + aim[2] * w[2];
    if (!active && align < 0.34) continue; // ~beyond 70° off-axis
    const p = project(w);

    if (t.done) {
      if (p && Math.abs(p.sx) <= 1.15 && Math.abs(p.sy) <= 1.15) {
        g.beginPath(); g.arc(p.x, p.y, 5 * dpr, 0, 7); g.fillStyle = 'rgba(51,210,155,0.5)'; g.fill();
      }
      continue;
    }

    if (!p || Math.abs(p.sx) > 1.15 || Math.abs(p.sy) > 1.15) {
      if (active) drawEdgeArrow(g, cw, ch, dpr, p, w);
      continue;
    }

    if (active) {
      g.beginPath(); g.arc(p.x, p.y, 15 * dpr, 0, 7);
      g.lineWidth = 3 * dpr; g.strokeStyle = 'rgba(255,255,255,0.85)'; g.stroke();
      if (t.progress > 0) {
        g.beginPath();
        g.arc(p.x, p.y, 15 * dpr, -Math.PI / 2, -Math.PI / 2 + t.progress * 2 * Math.PI);
        g.lineWidth = 5 * dpr; g.strokeStyle = '#33d29b'; g.lineCap = 'round'; g.stroke();
        g.lineCap = 'butt';
      }
      g.beginPath(); g.arc(p.x, p.y, 3 * dpr, 0, 7); g.fillStyle = '#fff'; g.fill();
    } else {
      g.beginPath(); g.arc(p.x, p.y, 6 * dpr, 0, 7);
      g.fillStyle = 'rgba(255,255,255,0.28)'; g.fill();
      g.lineWidth = 1.5 * dpr; g.strokeStyle = 'rgba(255,255,255,0.5)'; g.stroke();
    }
  }
}

function drawEdgeArrow(g, cw, ch, dpr, p, w) {
  // direction toward the (off-screen) active target, from screen centre
  let ang;
  if (p) ang = Math.atan2(p.y - ch / 2, p.x - cw / 2);
  else {
    const c = worldToCam(state.R, w);
    ang = Math.atan2(-c[1], c[0]); // fall back to raw bearing; flip Y for screen
  }
  const rad = Math.min(cw, ch) * 0.4;
  const x = cw / 2 + Math.cos(ang) * rad, y = ch / 2 + Math.sin(ang) * rad;
  g.save();
  g.translate(x, y); g.rotate(ang);
  g.beginPath();
  g.moveTo(10 * dpr, 0); g.lineTo(-8 * dpr, 7 * dpr); g.lineTo(-8 * dpr, -7 * dpr); g.closePath();
  g.fillStyle = 'rgba(255,255,255,0.9)'; g.fill();
  g.restore();
}

function loop(now = performance.now()) {
  state.raf = requestAnimationFrame(loop);
  if (glCanvas.width < 8) sizeCanvas();
  updateOrientation();

  if (!state.hasOrientation && !state.sim.active && now > state._simCheckAt) {
    state.sim.active = true;
    $('sim-banner').hidden = false;
  }

  updateGuidance(now);
  drawGuide();

  if (now - state.lastPreview > 60) { state.engine.presentFlat(); state.lastPreview = now; }
}

// ---- start / permissions ----------------------------------------------------
async function start() {
  note('');
  try {
    if (typeof DeviceOrientationEvent !== 'undefined' &&
        typeof DeviceOrientationEvent.requestPermission === 'function') {
      const res = await DeviceOrientationEvent.requestPermission();
      if (res !== 'granted') note('Motion access denied — drag mode enabled instead.', false);
    }
  } catch { /* not iOS, or dismissed */ }
  window.addEventListener('deviceorientation', onOrient, true);

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1920 }, height: { ideal: 1080 },
      },
    });
  } catch (e) {
    note('Camera unavailable: ' + (e.message || e.name) +
      '. The camera needs HTTPS (or localhost).', true);
    return;
  }

  video.srcObject = state.stream;
  try { await video.play(); } catch { /* autoplay quirk */ }

  try {
    if (!state.engine) state.engine = new PanoEngine(glCanvas);
  } catch (e) {
    note(e.message, true);
    return;
  }

  state.engine.clear();
  glCanvas.className = 'gl-mini';
  $('screen-capture').appendChild(glCanvas);
  resetCoverage();

  state._simCheckAt = performance.now() + 1400;
  state.running = true;
  showScreen('capture');   // must be visible before measuring the canvas
  sizeCanvas();
  loop();

  // Ask for location a beat after the camera prompt, not on top of it.
  if ($('geo').checked) setTimeout(() => setGeoEnabled(true), 900);
}

// ---- review --------------------------------------------------------------
async function toReview() {
  cancelAnimationFrame(state.raf);
  state.running = false;
  if (state.shots.length < 2) { toast('Capture at least 2 overlapping frames.'); return resume(); }

  const st = $('export-status');
  st.hidden = false;
  st.textContent = 'Stitching…';
  await new Promise((r) => setTimeout(r, 20));

  let result;
  try {
    result = await stitch(
      state.shots.map((s) => ({ gray: s.gray, w: s.gw, h: s.gh, quat: s.quat, hfovDeg: s.hfovDeg })),
      { onProgress: (stage, f) => { st.textContent = `Stitching · ${stage} ${Math.round(f * 100)}%`; } },
    );
  } catch (e) {
    result = { ok: false, log: ['stitch error: ' + (e && e.message || e)] };
  }
  state._stitchLog = result.log || [];
  console.log('[stitch]', ...(result.log || []));
  const diag = $('stitch-diag');
  diag.textContent = (result.log || []).join('\n');
  diag.classList.toggle('err', !result.ok);
  diag.hidden = result.ok;   // show automatically when it fell back; tap the chip to toggle

  try {
    if (result.ok) {
      st.textContent = 'Stitching · blending';
      await new Promise((r) => setTimeout(r, 20));
      const s0 = state.shots[0];
      const tanX = Math.tan((state.hfovDeg * DEG) / 2) / result.focalScale;
      const tanY = tanX * (s0.h / s0.w);
      // composite only the aligned frames (a mis-posed loose frame is worse
      // than a small gap); fall back to all if too few connected
      const parts = state.shots.map((s, k) => ({
        img: s.imgData, R: result.rotations[k], gain: result.gains[k], conn: result.connected[k],
      }));
      const conn = parts.filter((p) => p.conn);
      state.engine.compositeStitched(conn.length >= 3 ? conn : parts, tanX, tanY);
      const nConn = conn.length;
      if (nConn < state.shots.length) toast(`Aligned ${nConn} of ${state.shots.length} frames`);
    } else {
      state.engine.bake(); // gyro-only fallback (from the live splat accumulation)
      toast('Feature match failed — using gyro alignment');
    }
  } catch (e) {
    console.warn('composite failed', e);
    state.engine.bake();
    toast('Blend failed — using gyro alignment');
  }
  st.hidden = true;

  glCanvas.className = 'gl-full';
  $('viewer').appendChild(glCanvas);
  state.view.yaw = 0; state.view.pitch = 0; state.view.fov = 72 * DEG;
  state.viewFlat = false;
  $('btn-viewmode').textContent = 'Flat view';
  $('review-cov').textContent = `${state.shots.length} frames`;

  if (navigator.canShare) $('btn-share').hidden = false;

  showScreen('review');
  sizeCanvas();
  reviewLoop();
}

function resume() {
  if (state.engine) state.engine._composited = false; // live preview = accum again
  glCanvas.className = 'gl-mini';
  $('screen-capture').appendChild(glCanvas);
  state.running = true;
  showScreen('capture');
  sizeCanvas();
  loop();
}

function reviewLoop() {
  if (state.running) return;
  state._rraf = requestAnimationFrame(reviewLoop);
  sizeCanvas();
  if (state.viewFlat) state.engine.presentFlat();
  else state.engine.sphere(state.view.yaw, state.view.pitch, state.view.fov);
}

function stopReviewLoop() { cancelAnimationFrame(state._rraf); }

// ---- export ------------------------------------------------------------------
function timestamp() {
  const d = new Date(), p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

async function download() {
  const s = $('export-status');
  s.hidden = false; s.textContent = 'Rendering panorama…';
  await new Promise((r) => setTimeout(r, 30));

  let cvs, bounds;
  try {
    cvs = state.engine.exportEquirect();
    bounds = state.engine.coverageBounds();
  } catch (e) { s.textContent = 'Export failed: ' + e.message; setTimeout(() => (s.hidden = true), 2500); return; }

  const W = cvs.width, H = cvs.height;
  const megapixels = (W * H) / 1e6;

  // Honest GPano crop rectangle: tell Street View which rows/cols are real
  // imagery and which are dark padding.
  let crop = {};
  if (bounds && !bounds.full) {
    const horizFull = (bounds.rightUV - bounds.leftUV) > 0.94;
    const top = clamp(Math.round((1 - bounds.topUV) * H), 0, H - 2);
    const bot = clamp(Math.round((1 - bounds.botUV) * H), top + 1, H);
    crop = {
      croppedTop: top,
      croppedH: bot - top,
      croppedLeft: horizFull ? 0 : clamp(Math.round(bounds.leftUV * W), 0, W - 2),
      croppedW: horizFull ? W : Math.max(1, Math.round((bounds.rightUV - bounds.leftUV) * W)),
    };
  }

  cvs.toBlob(async (blob) => {
    try {
      if (!blob) throw new Error('Encoder returned nothing');
      const now = new Date();

      const xmp = buildGPanoXMP({
        width: W, height: H,
        headingDeg: state.headingDeg ?? 0,
        ...crop,
      });
      const exifSegment = buildExifSegment({
        width: W, height: H,
        date: now,
        gps: state.geo,
        headingDeg: state.headingDeg,
      });
      const out = await embedMetadata(blob, { exifSegment, xmpString: xmp });

      const name = `photosphere-${timestamp()}.jpg`;
      state._lastFile = new File([out], name, { type: 'image/jpeg' });

      const url = URL.createObjectURL(out);
      const a = document.createElement('a');
      a.href = url; a.download = name;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 4000);

      let msg = `Saved · ${W}×${H} · ${state.geo ? 'GPS ✓' : 'no GPS'}`;
      if (megapixels < 7.5) msg += ' · ⚠ under Street View’s 7.5 MP minimum';
      s.textContent = msg;
      setTimeout(() => (s.hidden = true), megapixels < 7.5 ? 4200 : 2200);
    } catch (e) {
      s.textContent = 'Failed: ' + e.message;
      setTimeout(() => (s.hidden = true), 2800);
    }
  }, 'image/jpeg', 0.92);
}

async function shareFile() {
  if (!state._lastFile) { await download(); }
  if (state._lastFile && navigator.canShare &&
      navigator.canShare({ files: [state._lastFile] })) {
    try { await navigator.share({ files: [state._lastFile], title: '360° photo' }); }
    catch { /* cancelled */ }
  } else {
    toast('Sharing not supported — use Download.');
  }
}

// ---- canvas sizing --------------------------------------------------------
function sizeCanvas() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const r = glCanvas.getBoundingClientRect();
  const w = Math.max(2, Math.round(r.width * dpr));
  const h = Math.max(2, Math.round(r.height * dpr));
  if (glCanvas.width !== w || glCanvas.height !== h) {
    glCanvas.width = w; glCanvas.height = h;
  }
}

// ---- input wiring --------------------------------------------------------
function wire() {
  $('btn-start').addEventListener('click', start, { once: false });

  $('btn-shutter').addEventListener('click', () => doCapture(true));
  $('btn-done').addEventListener('click', toReview);
  $('btn-cancel').addEventListener('click', () => {
    if (!confirm('Discard this panorama and go back?')) return;
    cancelAnimationFrame(state.raf);
    state.running = false;
    state.stream?.getTracks().forEach((t) => t.stop());
    state.stream = null;
    video.srcObject = null;
    setGeoEnabled(false);
    state.engine?.clear();
    resetCoverage();
    note('');
    showScreen('start');
  });
  $('btn-reset').addEventListener('click', () => { state.engine?.clear(); resetCoverage(); toast('Cleared'); });

  $('btn-gear').addEventListener('click', () => $('panel-fov').classList.toggle('is-open'));
  $('btn-fov-close').addEventListener('click', () => $('panel-fov').classList.remove('is-open'));
  $('fov').addEventListener('input', (e) => {
    state.hfovDeg = +e.target.value;
    $('fov-val').textContent = state.hfovDeg + '°';
    if (state.targets.length) buildTargets(); // re-space the dot lattice to the new FOV
  });
  $('autocap').addEventListener('change', (e) => (state.autoCap = e.target.checked));
  $('vidrot').addEventListener('change', (e) => (state.vidRot = +e.target.value));
  $('geo').addEventListener('change', (e) => setGeoEnabled(e.target.checked));

  // desktop / no-sensor aiming
  const sim = state.sim;
  video.addEventListener('pointerdown', (e) => {
    if (state.hasOrientation) return;
    sim.drag = true; sim.lx = e.clientX; sim.ly = e.clientY;
  });
  window.addEventListener('pointermove', (e) => {
    if (!sim.drag) return;
    sim.yaw -= (e.clientX - sim.lx) * 0.005;
    sim.pitch = clamp(sim.pitch + (e.clientY - sim.ly) * 0.005, -1.45, 1.45);
    sim.lx = e.clientX; sim.ly = e.clientY;
  });
  window.addEventListener('pointerup', () => (sim.drag = false));

  // review: back / flat toggle / download / share
  $('btn-back').addEventListener('click', () => { stopReviewLoop(); state._lastFile = null; resume(); });
  $('review-cov').addEventListener('click', () => { const d = $('stitch-diag'); d.hidden = !d.hidden; });
  $('btn-viewmode').addEventListener('click', () => {
    state.viewFlat = !state.viewFlat;
    $('btn-viewmode').textContent = state.viewFlat ? 'Sphere view' : 'Flat view';
  });
  $('btn-download').addEventListener('click', download);
  $('btn-share').addEventListener('click', shareFile);

  // review: drag to look, wheel / pinch to zoom
  const v = state.view;
  const slot = $('viewer');
  slot.addEventListener('pointerdown', (e) => { v.drag = true; v.lx = e.clientX; v.ly = e.clientY; slot.setPointerCapture?.(e.pointerId); });
  slot.addEventListener('pointermove', (e) => {
    if (!v.drag) return;
    const k = v.fov / (glCanvas.clientHeight || 1);
    v.yaw -= (e.clientX - v.lx) * k;
    v.pitch = clamp(v.pitch + (e.clientY - v.ly) * k, -1.48, 1.48);
    v.lx = e.clientX; v.ly = e.clientY;
  });
  const endDrag = () => (v.drag = false);
  slot.addEventListener('pointerup', endDrag);
  slot.addEventListener('pointercancel', endDrag);
  slot.addEventListener('wheel', (e) => {
    e.preventDefault();
    v.fov = clamp(v.fov * (1 + e.deltaY * 0.0012), 35 * DEG, 105 * DEG);
  }, { passive: false });
  slot.addEventListener('touchmove', (e) => {
    if (e.touches.length !== 2) return;
    e.preventDefault();
    const d = Math.hypot(
      e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY);
    if (v.pinch) v.fov = clamp(v.fov * (v.pinch / d), 35 * DEG, 105 * DEG);
    v.pinch = d;
  }, { passive: false });
  slot.addEventListener('touchend', () => (v.pinch = 0));

  window.addEventListener('resize', sizeCanvas);
  window.addEventListener('orientationchange', () => setTimeout(sizeCanvas, 300));
}

// ---- boot --------------------------------------------------------------------
function boot() {
  wire();
  $('app-version').textContent = 'v' + APP_VERSION;
  console.log('PhotoSphere v' + APP_VERSION);

  const secure = location.protocol === 'https:' ||
    ['localhost', '127.0.0.1'].includes(location.hostname);
  if (!secure) {
    note('Open this over HTTPS or on localhost so the camera can be used.', true);
  }
  if (!('mediaDevices' in navigator) || !navigator.mediaDevices?.getUserMedia) {
    note('This browser has no camera API.', true);
    $('btn-start').disabled = true;
  }

  if ('serviceWorker' in navigator && secure) {
    navigator.serviceWorker.register('sw.js').catch(() => {});
  }

  if (location.search.includes('debug')) {
    window.__ps = { state, buildTargets, updateGuidance, drawGuide, showScreen, loop };
  }
}

boot();
