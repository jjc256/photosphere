import { PanoEngine } from './pano.js';
import { stitch } from './stitch.js';
import { buildGPanoXMP, embedMetadata } from './xmp.js';
import { buildExifSegment } from './exif.js';
import {
  DEG, deviceQuat, quatToMat3, yawPitchToMat3, quatFromAxisAngle,
  multiplyQuat, normalizeQuat, quatAngle, forwardDir, inFrustum,
} from './orientation.js';

const APP_VERSION = '0.4.2';

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

// ---- panorama coverage grid ----------------------------------------------
const NCOL = 24, NROW = 8;
const LAT_MAX = 62, LAT_MIN = -62; // degrees; poles are hard to shoot hand-held

// ---- captured-frame budget --------------------------------------------------
const MAX_SHOTS = 32;   // memory-bounded; ImageData is kept for the final blend
const CAP_LONG = 1024;  // long side kept for compositing
const GRAY_LONG = 512;  // long side used for feature detection
const SHARP_MIN = 4.0;  // mean |gradient| floor — below this a frame is too blurred to match

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
  covered: new Set(),
  shots: [],          // { imgData, w, h, gray, gw, gh, quat, hfovDeg }
  cells: [],
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

// ---- coverage grid ----------------------------------------------------------
function buildGrid() {
  const grid = $('grid');
  grid.style.gridTemplateColumns = `repeat(${NCOL}, 1fr)`;
  grid.style.gridTemplateRows = `repeat(${NROW}, 1fr)`;
  grid.innerHTML = '';
  state.cells = [];
  for (let r = 0; r < NROW; r++) {
    for (let c = 0; c < NCOL; c++) {
      const d = document.createElement('div');
      d.className = 'cell';
      grid.appendChild(d);
      state.cells.push(d);
    }
  }
}

function cellWorld(c, r) {
  const lon = ((c + 0.5) / NCOL) * 2 * Math.PI - Math.PI;
  const lat = (LAT_MAX - ((r + 0.5) / NROW) * (LAT_MAX - LAT_MIN)) * DEG;
  const cl = Math.cos(lat);
  return [cl * Math.sin(lon), Math.sin(lat), -cl * Math.cos(lon)];
}

function resetCoverage() {
  state.covered.clear();
  state.shots = [];
  state.lastCapQuat = null;
  $('coverage').textContent = '0%';
  for (const d of state.cells) d.classList.remove('covered', 'current');
}

function markCovered(R, tanX, tanY) {
  for (let r = 0; r < NROW; r++) {
    for (let c = 0; c < NCOL; c++) {
      const w = cellWorld(c, r);
      if (inFrustum(R, w[0], w[1], w[2], tanX * 0.8, tanY * 0.8)) {
        state.covered.add(r * NCOL + c);
      }
    }
  }
  $('coverage').textContent =
    Math.round((state.covered.size / (NCOL * NROW)) * 100) + '%';
}

function refreshGridClasses() {
  const f = forwardDir(state.R);
  const lon = Math.atan2(f[0], -f[2]);
  const lat = Math.asin(clamp(f[1], -1, 1));
  let cur = -1;
  if (lat <= LAT_MAX * DEG && lat >= LAT_MIN * DEG) {
    const c = ((Math.floor(((lon + Math.PI) / (2 * Math.PI)) * NCOL)) % NCOL + NCOL) % NCOL;
    const r = clamp(
      Math.floor(((LAT_MAX * DEG - lat) / ((LAT_MAX - LAT_MIN) * DEG)) * NROW), 0, NROW - 1);
    cur = r * NCOL + c;
  }
  for (let i = 0; i < state.cells.length; i++) {
    const d = state.cells[i];
    d.classList.toggle('covered', state.covered.has(i));
    d.classList.toggle('current', i === cur);
  }
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
  const sharp = sharpness(gray, sm.w, sm.h);
  // auto-captured but clearly motion-blurred / featureless -> skip it
  if (!manual && sharp < SHARP_MIN) { $('hint').textContent = 'Too blurry — hold still'; return; }

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
}

function doCapture(manual) {
  if (!video.videoWidth) return;
  updateOrientation(); // capture against the freshest pose
  const { tanX, tanY } = fovTangents();
  state.engine.splat(video, state.R, tanX, tanY, state.vidRot); // live guide preview
  stashShot(manual);                                            // for the real stitch on Done
  state.lastCapQuat = state.quat;
  markCovered(state.R, tanX, tanY);
  const s = $('btn-shutter');
  s.classList.remove('flash');
  void s.offsetWidth;
  s.classList.add('flash');
  if (navigator.vibrate) navigator.vibrate(manual ? 25 : 12);
}

function maybeAutoCapture(now) {
  if (state.speedQuat) {
    const dt = (now - state.speedT) / 1000 || 1 / 60;
    state.speed = quatAngle(state.quat, state.speedQuat) / dt;
  }
  state.speedQuat = state.quat;
  state.speedT = now;

  // require several consecutive slow frames — a phone still mid-swing blurs
  state.steadyFrames = state.speed < 0.22 ? (state.steadyFrames || 0) + 1 : 0; // rad/s ≈ 13°/s
  const steady = state.steadyFrames >= 3;

  const hint = $('hint');
  if (!state.autoCap) { hint.textContent = 'Tap the shutter to grab a frame'; return; }

  const moved = state.lastCapQuat ? quatAngle(state.quat, state.lastCapQuat) : Infinity;
  const step = (state.hfovDeg * DEG) * 0.42;

  if (state.engine.frames === 0) {
    hint.textContent = steady ? 'Capturing…' : 'Hold steady to start';
    if (steady) doCapture(false);
    return;
  }
  if (!steady) hint.textContent = 'Pause to capture';
  else if (moved < step) hint.textContent = 'Pan to a new area';
  else { hint.textContent = 'Capturing…'; doCapture(false); }
}

function loop(now = performance.now()) {
  state.raf = requestAnimationFrame(loop);
  if (glCanvas.width < 8) sizeCanvas();
  updateOrientation();

  if (!state.hasOrientation && !state.sim.active && now > state._simCheckAt) {
    state.sim.active = true;
    $('sim-banner').hidden = false;
  }

  maybeAutoCapture(now);

  if (now - state.lastClassPass > 90) { refreshGridClasses(); state.lastClassPass = now; }
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
  buildGrid();
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
      state.engine.compositeStitched(
        state.shots.map((s, k) => ({ img: s.imgData, R: result.rotations[k], gain: result.gains[k] })),
        tanX, tanY,
      );
      const nConn = result.connected.filter(Boolean).length;
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
}

boot();
