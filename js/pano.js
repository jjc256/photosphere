import { seamLabels } from './seams.js';
import { projectionRadiusLimit } from './camera-geometry.js';
import { overlapExposure } from './exposure.js';

// WebGL2 equirectangular panorama engine.
//
//  splat()   – project the current camera frame onto a float accumulation
//              buffer, weighted & feathered, additively blended.
//  present() – normalise (rgb / weight) the accum buffer to a target.
//  sphere()  – interactive perspective view of the finished panorama.
//  exportEquirect() – read the panorama back as a 2D <canvas> for encoding.

const VERT = `#version 300 es
out vec2 vUv;
void main() {
  float x = float((gl_VertexID & 1) << 2) - 1.0;
  float y = float((gl_VertexID & 2) << 1) - 1.0;
  vUv = vec2((x + 1.0) * 0.5, (y + 1.0) * 0.5);
  gl_Position = vec4(x, y, 0.0, 1.0);
}`;

const SPLAT_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform mat3 uRot;        // camera -> world
uniform vec2 uTan;        // tan(hfov/2), tan(vfov/2)
uniform float uFeather;   // edge feather width, 0..0.5
uniform float uGain;      // exposure gain
uniform mat2 uVidRot;     // in-plane frame rotation
uniform sampler2D uVideo;
const float PI = 3.14159265359;
void main() {
  float lon = (vUv.x - 0.5) * 2.0 * PI;
  float lat = (vUv.y - 0.5) * PI;
  float cl = cos(lat);
  vec3 world = vec3(cl * sin(lon), sin(lat), -cl * cos(lon));
  vec3 cam = transpose(uRot) * world;         // world -> camera
  if (cam.z > -1e-4) discard;                  // behind the lens
  float px = (cam.x / -cam.z) / uTan.x;
  float py = (cam.y / -cam.z) / uTan.y;
  float m = max(abs(px), abs(py));
  if (m > 1.0) discard;                        // outside the frame
  vec2 uv = uVidRot * vec2(px, py) * 0.5 + 0.5;
  vec3 c = clamp(texture(uVideo, uv).rgb * uGain, 0.0, 1.0);
  float w = 1.0 - smoothstep(1.0 - uFeather, 1.0, m);
  w = max(w, 2e-3);
  frag = vec4(c * w, w);
}`;

const PRESENT_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uAccum;
uniform float uFlipY;
void main() {
  vec2 uv = vec2(vUv.x, uFlipY > 0.5 ? 1.0 - vUv.y : vUv.y);
  vec4 a = texture(uAccum, uv);
  vec3 c = a.a > 3e-3 ? a.rgb / a.a : vec3(0.015, 0.016, 0.022);
  frag = vec4(c, 1.0);
}`;

const SPHERE_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uPano;
uniform vec2 uRes;
uniform float uYaw;
uniform float uPitch;
uniform float uFovY;
const float PI = 3.14159265359;
void main() {
  float aspect = uRes.x / uRes.y;
  vec2 ndc = vUv * 2.0 - 1.0;
  float t = tan(uFovY * 0.5);
  vec3 dir = normalize(vec3(ndc.x * t * aspect, ndc.y * t, -1.0));
  float cp = cos(uPitch), sp = sin(uPitch);
  vec3 d1 = vec3(dir.x, dir.y * cp - dir.z * sp, dir.y * sp + dir.z * cp);
  float cy = cos(uYaw), sy = sin(uYaw);
  vec3 d = vec3(d1.x * cy + d1.z * sy, d1.y, -d1.x * sy + d1.z * cy);
  float lon = atan(d.x, -d.z);
  float lat = asin(clamp(d.y, -1.0, 1.0));
  vec2 uv = vec2(lon / (2.0 * PI) + 0.5, lat / PI + 0.5);
  frag = vec4(texture(uPano, uv).rgb, 1.0);
}`;

const COVERAGE_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uAccum;
void main() {
  float a = texture(uAccum, vUv).a;
  frag = vec4(vec3(a > 4e-3 ? 1.0 : 0.0), 1.0);
}`;

// ---- multi-band compositor (stitched path) --------------------------------
// Shared warp: equirect texel -> world dir -> camera k -> frame uv, plus a
// border-distance weight used both for feathering and seam selection.
const WARP_HEAD = `#version 300 es
precision highp float;
in vec2 vUv;
uniform mat3 uRot;       // camera -> world (pass transpose of row-major c2w)
uniform vec2 uTan;       // tan(hfov/2), tan(vfov/2)
uniform float uGain;
uniform float uK1;       // radial distortion, solved by the stitcher
uniform float uK2;
uniform float uK3;
uniform float uLinearity;
uniform vec3 uColorGain;
uniform float uMaxRadius; // first monotonic radial branch, bounded by source corners
uniform mat2 uVidRot;    // in-plane frame rotation
uniform vec2 uCenter;    // calibrated principal point in source UVs
uniform sampler2D uFrame;
const float PI = 3.14159265359;
bool warp(out vec3 rgb, out float edge) {
  float lon = (vUv.x - 0.5) * 2.0 * PI;
  float lat = (vUv.y - 0.5) * PI;
  float cl = cos(lat);
  vec3 world = vec3(cl * sin(lon), sin(lat), -cl * cos(lon));
  vec3 cam = transpose(uRot) * world;
  if (cam.z > -1e-4) return false;
  // ideal pinhole coords, then push them back through the solved lens so we
  // sample where the ray actually landed on the sensor
  float xy = length(cam.xy);
  float theta = acos(clamp(-cam.z / length(cam), -1.0, 1.0));
  // Stay on the invertible branch of the calibrated projection. Beyond
  // this limit tan folds back and paints duplicate fragments of the frame.
  if (abs(uLinearity) * theta >= PI * 0.5) return false;
  float radius = abs(uLinearity) < 1e-5 ? theta :
    (uLinearity > 0.0 ? tan(theta * uLinearity) / uLinearity : sin(theta * uLinearity) / uLinearity);
  vec2 ideal = xy < 1e-6 ? vec2(0.0) : cam.xy * (radius / xy);
  float xu = ideal.x;
  float yu = ideal.y;
  float idealRadius = length(ideal);
  if (idealRadius > uMaxRadius) return false;
  float distortedRadius = idealRadius + uK1 * idealRadius * idealRadius + uK2 * idealRadius * idealRadius * idealRadius + uK3 * idealRadius * idealRadius * idealRadius * idealRadius;
  vec2 distorted = idealRadius < 1e-6 ? vec2(0.0) : ideal * (distortedRadius / idealRadius);
  float px = distorted.x / uTan.x;
  float py = distorted.y / uTan.y;
  vec2 rp = uVidRot * vec2(px, py);
  vec2 uv = rp * 0.5 + uCenter;
  if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return false;
  rgb = clamp(texture(uFrame, uv).rgb * uGain * uColorGain, 0.0, 1.0);
  edge = 2.0 * min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));   // distance to nearest frame edge, 0..1
  return true;
}`;

// Plain warp of one frame -> rgb, a = coverage.
const WARPC_FRAG = WARP_HEAD + `
out vec4 frag;
void main() {
  vec3 rgb; float edge;
  if (!warp(rgb, edge)) discard;
  frag = vec4(rgb, 1.0);
}`;

// Feather-weighted accumulation retains actual captured coverage for the
// final reconstruction, independent of the reduced-resolution seam masks.
const FA_FRAG = WARP_HEAD + `
out vec4 frag;
uniform float uEvidence;
void main() {
  vec3 rgb; float edge;
  if (!warp(rgb, edge)) discard;
  float w = (edge * edge + 0.02) * uEvidence;
  frag = vec4(rgb * w, w);
}`;

// Binary ownership mask for one frame (1 where it won the seam label).
const MASK_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uLabel;
uniform float uWant;
void main() {
  float lab = texture(uLabel, vUv).r * 255.0;
  frag = vec4(abs(lab - uWant) < 0.5 ? 1.0 : 0.0);
}`;

// Gaussian pyramid reduction, using the separable [1,4,6,4,1]/16 kernel.
// Bilinear taps at +/-1.2 combine its outer coefficients into nine samples.
const REDUCE_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uSrc;
uniform vec2 uTexel;
void main() {
  vec4 sum = vec4(0.0);
  for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
    float weight = (x == 0 ? 0.375 : 0.3125) * (y == 0 ? 0.375 : 0.3125);
    sum += weight * texture(uSrc, vUv + vec2(x, y) * uTexel * 1.2);
  }
  frag = sum;
}`;

// Normalize each Gaussian level at its texel centres, and extend uncovered
// texels from the next coarser level. This pull/push extension ensures that
// Laplacians telescope exactly even at image borders (division after bilinear
// interpolation would otherwise leave a halo on a single isolated image).
const SOURCE_FILL_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uSource;
uniform sampler2D uCoarse;
uniform bool uBase;
void main() {
  vec4 s = texture(uSource, vUv);
  vec3 value = s.a > 1e-6 ? s.rgb / s.a : (uBase ? vec3(0.0) : texture(uCoarse, vUv).rgb);
  frag = vec4(value, 1.0);
}`;

const LAPLACIAN_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uFine;
uniform sampler2D uCoarse;
uniform sampler2D uMask;
uniform bool uBase;
void main() {
  float weight = texture(uMask, vUv).r;
  if (weight < 1e-6) discard;
  vec3 fine = texture(uFine, vUv).rgb, coarse = texture(uCoarse, vUv).rgb;
  frag = vec4((uBase ? fine : fine - coarse) * weight, weight);
}`;

const RECONSTRUCT_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uBand;
uniform sampler2D uCoarse;
uniform bool uBase;
void main() {
  vec4 band = texture(uBand, vUv);
  vec3 value = band.a > 1e-6 ? band.rgb / band.a : vec3(0.0);
  if (!uBase) value += texture(uCoarse, vUv).rgb;
  frag = vec4(value, 1.0);
}`;

const FINISH_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uResult;
uniform sampler2D uCoverage;
void main() {
  // A coarse Gaussian extends outside actual coverage; clip only at the final
  // level, after reconstruction, instead of cutting each band's support.
  float covered = texture(uCoverage, vUv).a;
  frag = covered > 0.5 ? vec4(clamp(texture(uResult, vUv).rgb, 0.0, 1.0), 1.0) : vec4(0.0);
}`;

// Dense Gaussian kernel at the selected pyramid level. Tap spacing stays
// one texel; widening a blur by spacing samples apart produces a comb.
const BLUR_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uSrc;
uniform vec2 uStep;
uniform float uSigma;
void main() {
  float wsum = 0.0;
  vec4 acc = vec4(0.0);
  for (int i = -6; i <= 6; i++) {
    float w = exp(-float(i * i) / (2.0 * uSigma * uSigma));
    acc += texture(uSrc, vUv + uStep * float(i)) * w;
    wsum += w;
  }
  frag = acc / wsum;
}`;

// Fill only genuinely uncovered output pixels from gyro-positioned frames.
// These frames are deliberately excluded from seam selection: a weak image can
// close a hole, but must never replace feature-aligned imagery.
const HOLEFILL_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uBase;
uniform sampler2D uFill;
void main() {
  vec4 base = texture(uBase, vUv);
  if (base.a > 0.5) { frag = base; return; }
  frag = texture(uFill, vUv);
}`;

// Normalise the accum buffer; alpha carries coverage (1 = real imagery).
const NORMALIZE2_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uAccum;
void main() {
  vec4 a = texture(uAccum, vUv);
  if (a.a > 1e-4) frag = vec4(a.rgb / a.a, 1.0);
  else frag = vec4(0.0, 0.0, 0.0, 0.0);
}`;

// Fill uncovered caps: for a black texel, borrow the nearest covered texel
// along the same meridian (a few marches converge the poles).
const POLEFILL_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uSrc;
uniform vec2 uTexel;
// average a covered row across longitude so the cap fill is a smooth wash,
// not per-column streaks
vec4 rowAvg(float y) {
  vec3 acc = vec3(0.0); float n = 0.0;
  for (int i = -8; i <= 8; i++) {
    vec4 s = texture(uSrc, vec2(vUv.x + float(i) * uTexel.x * 4.0, y));
    if (s.a > 0.5) { acc += s.rgb; n += 1.0; }
  }
  return n > 0.0 ? vec4(acc / n, 1.0) : vec4(0.0);
}
void main() {
  vec4 c = texture(uSrc, vUv);
  if (c.a > 0.5) { frag = c; return; }
  for (int k = 1; k <= 20; k++) {
    float o = float(k) * uTexel.y * 5.0;
    vec4 up = rowAvg(vUv.y + o);
    if (up.a > 0.5) { frag = up; return; }
    vec4 dn = rowAvg(vUv.y - o);
    if (dn.a > 0.5) { frag = dn; return; }
  }
  frag = c;
}`;

const BLIT_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uSrc;
uniform float uFlipY;
void main() {
  frag = texture(uSrc, vec2(vUv.x, uFlipY > 0.5 ? 1.0 - vUv.y : vUv.y));
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error('Shader compile failed: ' + gl.getShaderInfoLog(s));
  }
  return s;
}

function program(gl, fragSrc) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fragSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('Program link failed: ' + gl.getProgramInfoLog(p));
  }
  return p;
}

export class PanoEngine {
  constructor(canvas, { size = 4096 } = {}) {
    const gl = canvas.getContext('webgl2', {
      antialias: false,
      alpha: false,
      preserveDrawingBuffer: false,
      premultipliedAlpha: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is required (needs iOS 15+ / a modern browser).');
    this.gl = gl;
    this.canvas = canvas;

    const floatLinear = gl.getExtension('OES_texture_float_linear');
    const cbf = gl.getExtension('EXT_color_buffer_float');
    const cbhf = gl.getExtension('EXT_color_buffer_half_float');
    if (!cbf && !cbhf) throw new Error('This device cannot render to float textures.');
    this._floatLinear = !!floatLinear;

    this.size = Math.min(size, 4096, gl.getParameter(gl.MAX_TEXTURE_SIZE));
    this.h = this.size / 2;

    this.pSplat = program(gl, SPLAT_FRAG);
    this.pPresent = program(gl, PRESENT_FRAG);
    this.pSphere = program(gl, SPHERE_FRAG);
    this.pCoverage = program(gl, COVERAGE_FRAG);
    this.pWarpC = program(gl, WARPC_FRAG);
    this.pMask = program(gl, MASK_FRAG);
    this.pReduce = program(gl, REDUCE_FRAG);
    this.pSourceFill = program(gl, SOURCE_FILL_FRAG);
    this.pLaplacian = program(gl, LAPLACIAN_FRAG);
    this.pReconstruct = program(gl, RECONSTRUCT_FRAG);
    this.pFinish = program(gl, FINISH_FRAG);
    this.pFA = program(gl, FA_FRAG);
    this.pBlur = program(gl, BLUR_FRAG);
    this.pNorm2 = program(gl, NORMALIZE2_FRAG);
    this.pHole = program(gl, HOLEFILL_FRAG);
    this.pPole = program(gl, POLEFILL_FRAG);
    this.pBlit = program(gl, BLIT_FRAG);
    this._composited = false;

    this.vao = gl.createVertexArray(); // empty; vertices come from gl_VertexID

    // Accumulation target (weighted colour in rgb, weight in a).
    this.accumTex = this._tex(this.size, this.h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT,
      this._floatLinear ? gl.LINEAR : gl.NEAREST, gl.REPEAT);
    this.accumFbo = this._fbo(this.accumTex);

    // Normalised panorama (for the interactive viewer).
    this.panoTex = this._tex(this.size, this.h, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR, gl.REPEAT);
    this.panoFbo = this._fbo(this.panoTex);

    // Current camera frame.
    this.videoTex = this._tex(2, 2, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR, gl.CLAMP_TO_EDGE);

    this.frames = 0;
    this.clear();
  }

  _tex(w, h, internal, format, type, filter, wrap) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  _fbo(tex) {
    const gl = this.gl;
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return f;
  }

  clear() {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFbo);
    gl.viewport(0, 0, this.size, this.h);
    gl.disable(gl.BLEND);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.frames = 0;
    this._composited = false;
  }

  _quad() {
    this.gl.bindVertexArray(this.vao);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 3);
  }

  // Project one camera frame onto the accumulation buffer.
  // R: camera->world mat3. tanX/tanY: tan(fov/2). vidRot: 0..3 quarter turns.
  // source: a <video> or any TexImageSource (ImageData/canvas/ImageBitmap).
  splat(source, R, tanX, tanY, vidRot = 0, feather = 0.16, gain = 1) {
    const gl = this.gl;
    if (source.videoWidth === 0) return;

    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

    const a = (vidRot % 4) * Math.PI / 2;
    const c = Math.cos(a), s = Math.sin(a);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFbo);
    gl.viewport(0, 0, this.size, this.h);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE);

    gl.useProgram(this.pSplat);
    gl.uniformMatrix3fv(gl.getUniformLocation(this.pSplat, 'uRot'), false, R);
    gl.uniform2f(gl.getUniformLocation(this.pSplat, 'uTan'), tanX, tanY);
    gl.uniform1f(gl.getUniformLocation(this.pSplat, 'uFeather'), feather);
    gl.uniform1f(gl.getUniformLocation(this.pSplat, 'uGain'), gain);
    gl.uniformMatrix2fv(gl.getUniformLocation(this.pSplat, 'uVidRot'), false, [c, s, -s, c]);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.uniform1i(gl.getUniformLocation(this.pSplat, 'uVideo'), 0);
    this._quad();

    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.frames++;
  }

  _present(target, vpW, vpH, flipY) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target);
    gl.viewport(0, 0, vpW, vpH);
    gl.disable(gl.BLEND);
    gl.useProgram(this.pPresent);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.accumTex);
    gl.uniform1i(gl.getUniformLocation(this.pPresent, 'uAccum'), 0);
    gl.uniform1f(gl.getUniformLocation(this.pPresent, 'uFlipY'), flipY ? 1 : 0);
    this._quad();
  }

  // Draw the normalised equirect straight to the on-screen canvas (letterboxed 2:1).
  presentFlat() {
    const gl = this.gl;
    const cw = this.canvas.width, chh = this.canvas.height;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, cw, chh);
    gl.disable(gl.BLEND);
    gl.clearColor(0.01, 0.01, 0.015, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    let w = cw, h = cw / 2;
    if (h > chh) { h = chh; w = chh * 2; }
    gl.viewport((cw - w) / 2, (chh - h) / 2, w, h);
    gl.activeTexture(gl.TEXTURE0);
    if (this._composited) {
      gl.useProgram(this.pBlit);
      gl.bindTexture(gl.TEXTURE_2D, this.panoTex);
      gl.uniform1i(gl.getUniformLocation(this.pBlit, 'uSrc'), 0);
      gl.uniform1f(gl.getUniformLocation(this.pBlit, 'uFlipY'), 0);
    } else {
      gl.useProgram(this.pPresent);
      gl.bindTexture(gl.TEXTURE_2D, this.accumTex);
      gl.uniform1i(gl.getUniformLocation(this.pPresent, 'uAccum'), 0);
      gl.uniform1f(gl.getUniformLocation(this.pPresent, 'uFlipY'), 0);
    }
    this._quad();
  }

  // Bake accum -> panoTex, for the interactive viewer.
  bake() {
    this._present(this.panoFbo, this.size, this.h, false);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
  }

  // Which part of the sphere actually got imagery. Returns normalised bounds
  // in [0,1] (u: 0=lon -180 .. 1=lon +180, v: 0=south pole .. 1=north pole)
  // plus `full` when coverage is effectively a whole sphere.
  coverageBounds() {
    const gl = this.gl;
    const W = 128, H = 64;
    if (!this._covFbo) {
      this._covTex = this._tex(W, H, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST, gl.CLAMP_TO_EDGE);
      this._covFbo = this._fbo(this._covTex);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._covFbo);
    gl.viewport(0, 0, W, H);
    gl.disable(gl.BLEND);
    gl.useProgram(this.pCoverage);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._composited ? this.panoTex : this.accumTex);
    gl.uniform1i(gl.getUniformLocation(this.pCoverage, 'uAccum'), 0);
    this._quad();

    const px = new Uint8Array(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    let minX = W, maxX = -1, minY = H, maxY = -1, hit = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (px[(y * W + x) * 4] > 127) {
          hit++;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (hit === 0) return { full: true, leftUV: 0, rightUV: 1, botUV: 0, topUV: 1 };

    const leftUV = minX / W;
    const rightUV = (maxX + 1) / W;
    const botUV = minY / H;          // readPixels row 0 = bottom = south pole
    const topUV = (maxY + 1) / H;
    const frac = hit / (W * H);
    const full = frac > 0.9 && (rightUV - leftUV) > 0.97 && (topUV - botUV) > 0.92;
    return { full, leftUV, rightUV, botUV, topUV };
  }

  // Interactive perspective view of panoTex onto the on-screen canvas.
  sphere(yaw, pitch, fovY) {
    const gl = this.gl;
    const w = this.canvas.width, h = this.canvas.height;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.BLEND);
    gl.useProgram(this.pSphere);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.panoTex);
    gl.uniform1i(gl.getUniformLocation(this.pSphere, 'uPano'), 0);
    gl.uniform2f(gl.getUniformLocation(this.pSphere, 'uRes'), w, h);
    gl.uniform1f(gl.getUniformLocation(this.pSphere, 'uYaw'), yaw);
    gl.uniform1f(gl.getUniformLocation(this.pSphere, 'uPitch'), pitch);
    gl.uniform1f(gl.getUniformLocation(this.pSphere, 'uFovY'), fovY);
    this._quad();
  }

  // Read the panorama back as a top-row-first 2D <canvas>.
  exportEquirect() {
    const gl = this.gl;
    const w = this.size, h = this.h;
    if (!this._expTex) {
      this._expTex = this._tex(w, h, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST, gl.CLAMP_TO_EDGE);
      this._expFbo = this._fbo(this._expTex);
    }
    if (this._composited) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this._expFbo); // left bound for readPixels below
      gl.viewport(0, 0, w, h); gl.disable(gl.BLEND);
      gl.useProgram(this.pBlit);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.panoTex);
      gl.uniform1i(gl.getUniformLocation(this.pBlit, 'uSrc'), 0);
      gl.uniform1f(gl.getUniformLocation(this.pBlit, 'uFlipY'), 1);
      this._quad();
    } else {
      this._present(this._expFbo, w, h, true); // flipY so readPixels comes out top-first
    }
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    const cvs = document.createElement('canvas');
    cvs.width = w; cvs.height = h;
    const ctx = cvs.getContext('2d');
    const img = ctx.createImageData(w, h);
    img.data.set(px);
    ctx.putImageData(img, 0, 0);
    return cvs;
  }

  // ---- multi-band seam compositor --------------------------------------
  // Standard stitcher compositing chain (OpenCV Stitcher / enblend):
  //   coverage mosaic -> graph-cut seam labels -> Burt-Adelson
  //   multi-band blend.
  // Intermediates run at 2048 wide to keep iOS GPU memory sane; the result
  // is upscaled into panoTex.
  _initComposite() {
    if (this._compReady) return;
    const gl = this.gl;
    const cs = Math.min(2048, this.size);
    this.cs = cs; this.csh = cs / 2;
    // Half-float filtering is core in WebGL2; the optional float-linear
    // extension concerns 32-bit floats. Nearest filtering here draws steps.
    const hf = gl.LINEAR;
    const t8 = () => this._tex(cs, this.csh, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR, gl.REPEAT);
    const t16 = () => this._tex(cs, this.csh, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, hf, gl.REPEAT);
    this.frameTex = this._tex(4, 4, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR, gl.CLAMP_TO_EDGE);
    this.accHi = t16(); this.accHiFbo = this._fbo(this.accHi);
    this.accLo = t16(); this.accLoFbo = this._fbo(this.accLo);
    this.labelTex = this._tex(cs, this.csh, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST, gl.REPEAT);
    this.labelFbo = this._fbo(this.labelTex);
    this.avgTex = t8(); this.avgFbo = this._fbo(this.avgTex);
    this.wTex = t8(); this.wFbo = this._fbo(this.wTex);        // warped frame
    this.mTex = t16(); this.mFbo = this._fbo(this.mTex);        // mask / scratch
    this.mHi = t16(); this.mHiFbo = this._fbo(this.mHi);        // mask, narrow blur
    this.pingTex = t16(); this.pingFbo = this._fbo(this.pingTex);
    this.pyramid = [{ w: cs, h: this.csh, source: this.wTex, sourceFbo: this.wFbo,
      mask: this.mHi, maskFbo: this.mHiFbo, accum: this.accHi, accumFbo: this.accHiFbo,
      result: this.accLo, resultFbo: this.accLoFbo }];
    for (let w = cs / 2, h = this.csh / 2; w >= 64; w /= 2, h /= 2) {
      const tex = () => this._tex(w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR, gl.REPEAT);
      const source = tex(), mask = tex(), accum = tex(), result = tex();
      this.pyramid.push({ w, h, source, sourceFbo: this._fbo(source), mask, maskFbo: this._fbo(mask),
        accum, accumFbo: this._fbo(accum), result, resultFbo: this._fbo(result) });
    }
    this.exposureTex = this._tex(512, 256, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST, gl.REPEAT);
    this.exposureFbo = this._fbo(this.exposureTex);
    this._compReady = true;
  }

  _uploadFrame(img) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.frameTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  }

  _warpUniforms(prog, uRot, tanX, tanY, gain, vidRot = 0, camera = null, image = null, colorGain = [1, 1, 1]) {
    const gl = this.gl;
    const a = (vidRot % 4) * Math.PI / 2;
    const c = Math.cos(a), s = Math.sin(a);
    gl.uniformMatrix3fv(gl.getUniformLocation(prog, 'uRot'), false, uRot);
    const focalScale = camera?.focalScale || 1;
    const tx = (camera?.tan?.[0] ?? tanX) / focalScale;
    const ty = (camera?.tan?.[1] ?? tanY) / focalScale;
    const a1 = (this._k1 || 0) * focalScale;
    const a2 = (this._k2 || 0) * focalScale ** 2;
    const a3 = (this._k3 || 0) * focalScale ** 3;
    const center = camera?.center || this._center || [0.5, 0.5];
    gl.uniform2f(gl.getUniformLocation(prog, 'uTan'), tx, ty);
    gl.uniform1f(gl.getUniformLocation(prog, 'uMaxRadius'), projectionRadiusLimit(tx, ty, center, vidRot, a1, a2, a3));
    gl.uniform1f(gl.getUniformLocation(prog, 'uGain'), gain);
    gl.uniform3fv(gl.getUniformLocation(prog, 'uColorGain'), colorGain);
    gl.uniform1f(gl.getUniformLocation(prog, 'uK1'), a1);
    gl.uniform1f(gl.getUniformLocation(prog, 'uK2'), a2);
    gl.uniform1f(gl.getUniformLocation(prog, 'uK3'), a3);
    gl.uniform1f(gl.getUniformLocation(prog, 'uLinearity'), this._linearity ?? 1);
    gl.uniformMatrix2fv(gl.getUniformLocation(prog, 'uVidRot'), false, [c, s, -s, c]);
    gl.uniform2fv(gl.getUniformLocation(prog, 'uCenter'), center);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.frameTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'uFrame'), 0);
  }

  // Blur at a reduced resolution with adjacent samples, then interpolate
  // back. Each 2x reduction averages every input texel, so even thin features
  // contribute continuously. Float intermediates preserve tiny coverage
  // weights at image borders instead of quantizing them into visible lines.
  _blur(srcTex, dstFbo, pingTex, pingFbo, sigma) {
    const gl = this.gl;
    const levels = Math.max(0, Math.min(
      Math.ceil(Math.log2(Math.max(1, sigma / 2))),
      Math.floor(Math.log2(Math.min(this.cs, this.csh))) - 1));
    this._blurLevels ||= [];
    const blit = (src, dst, w, h) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst); gl.viewport(0, 0, w, h);
      gl.useProgram(this.pBlit);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, src);
      gl.uniform1i(gl.getUniformLocation(this.pBlit, 'uSrc'), 0);
      gl.uniform1f(gl.getUniformLocation(this.pBlit, 'uFlipY'), 0);
      this._quad();
    };
    gl.disable(gl.BLEND);
    let input = srcTex, w = this.cs, h = this.csh, target = dstFbo;
    for (let level = 0; level < levels; level++) {
      w = Math.max(1, Math.floor(w / 2)); h = Math.max(1, Math.floor(h / 2));
      if (!this._blurLevels[level]) {
        const tex = this._tex(w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR, gl.REPEAT);
        const ping = this._tex(w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR, gl.REPEAT);
        this._blurLevels[level] = { tex, fbo: this._fbo(tex), ping, pingFbo: this._fbo(ping) };
      }
      const buffer = this._blurLevels[level];
      blit(input, buffer.fbo, w, h);
      input = buffer.tex; target = buffer.fbo;
      pingTex = buffer.ping; pingFbo = buffer.pingFbo;
    }
    gl.useProgram(this.pBlur);
    gl.uniform1i(gl.getUniformLocation(this.pBlur, 'uSrc'), 0);
    gl.uniform1f(gl.getUniformLocation(this.pBlur, 'uSigma'), Math.max(0.5, sigma / 2 ** levels));
    gl.bindFramebuffer(gl.FRAMEBUFFER, pingFbo); gl.viewport(0, 0, w, h);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, input);
    gl.uniform2f(gl.getUniformLocation(this.pBlur, 'uStep'), 1 / w, 0);
    this._quad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, target);
    gl.bindTexture(gl.TEXTURE_2D, pingTex);
    gl.uniform2f(gl.getUniformLocation(this.pBlur, 'uStep'), 0, 1 / h);
    this._quad();
    if (levels) blit(input, dstFbo, this.cs, this.csh);
  }

  _vp() { this.gl.viewport(0, 0, this.cs, this.csh); }

  // frames: [{ img, R (row-major camera->world), gain, weak, connected, vidRot }]. tanX/tanY =
  // tan(fov/2) for the focal-corrected lens. Fills panoTex.
  compositeStitched(frames, tanX, tanY, k1 = 0, k2 = 0, k3 = 0, linearity = 1, center = [0.5, 0.5]) {
    const gl = this.gl;
    this._k1 = k1;
    this._k2 = k2;
    this._k3 = k3;
    this._linearity = linearity;
    this._center = center;
    this._initComposite();
    const w = this.cs, h = this.csh;
    // Keep captured coverage, including sensor-positioned frames. Confidence
    // controls seam ownership, rather than deleting entire captured regions.
    const blendFrames = frames;
    const rots = blendFrames.map((f) => matT3col(f.R));
    // Solve exposure over all geometric overlaps, including textureless
    // images which have no feature-match edge in the alignment graph.
    const exposureWarps = blendFrames.map((frame, k) => {
      this._uploadFrame(frame.img);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.exposureFbo); gl.viewport(0, 0, 512, 256);
      gl.disable(gl.BLEND); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.pWarpC);
      this._warpUniforms(this.pWarpC, rots[k], tanX, tanY, 1, frame.vidRot || 0, frame.camera, frame.img);
      this._quad();
      const pixels = new Uint8Array(512 * 256 * 4);
      gl.readPixels(0, 0, 512, 256, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      return pixels;
    });
    const colorGains = overlapExposure(exposureWarps, 512, 256, blendFrames.map((f) => f.gain || 1));
    this.exposureGains = colorGains;
    const warpTo = (prog, fbo, k) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      this._vp();
      const frame = blendFrames[k];
      this._warpUniforms(prog, rots[k], tanX, tanY, 1, frame.vidRot || 0, frame.camera, frame.img, colorGains[k]);
      gl.uniform1f(gl.getUniformLocation(prog, 'uEvidence'), frame.weak && !frame.connected ? 0.1 : 1);
    };

    // ---- 1. consensus mosaic (feather average) -> avgTex ----------------
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accLoFbo);
    this._vp(); gl.disable(gl.BLEND);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD); gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(this.pFA);
    blendFrames.forEach((fr, k) => {
      this._uploadFrame(fr.img);
      warpTo(this.pFA, this.accLoFbo, k);
      this._quad();
    });
    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.avgFbo);
    this._vp();
    gl.useProgram(this.pNorm2);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.accLo);
    gl.uniform1i(gl.getUniformLocation(this.pNorm2, 'uAccum'), 0);
    this._quad();

    // ---- 2. graph-cut seam labels -------------------------------------
    const labels = seamLabels(exposureWarps, 512, 256, colorGains,
      blendFrames.map((fr) => !fr.weak || fr.connected));
    const labelPixels = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const p = Math.min(255, Math.floor((y + 0.5) * 256 / h)) * 512 + Math.min(511, Math.floor((x + 0.5) * 512 / w));
      labelPixels[(y * w + x) * 4] = labels[p] + 1;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.labelTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, labelPixels);

    // ---- 3. Full Gaussian-mask / Laplacian-image pyramid blend ---------
    const pyramid = this.pyramid;
    const bind = (unit, tex, prog, name) => {
      gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(gl.getUniformLocation(prog, name), unit);
    };
    for (const level of pyramid) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, level.accumFbo); gl.viewport(0, 0, level.w, level.h);
      gl.disable(gl.BLEND); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    }
    const reduce = (source, target, level, previous) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, target); gl.viewport(0, 0, level.w, level.h);
      gl.useProgram(this.pReduce); gl.disable(gl.BLEND);
      bind(0, source, this.pReduce, 'uSrc');
      gl.uniform2f(gl.getUniformLocation(this.pReduce, 'uTexel'), 1 / previous.w, 1 / previous.h);
      this._quad();
    };
    blendFrames.forEach((fr, k) => {
      this._uploadFrame(fr.img);
      gl.useProgram(this.pWarpC); warpTo(this.pWarpC, this.wFbo, k);
      gl.disable(gl.BLEND); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      this._quad();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.mFbo); this._vp(); gl.useProgram(this.pMask);
      bind(0, this.labelTex, this.pMask, 'uLabel');
      gl.uniform1f(gl.getUniformLocation(this.pMask, 'uWant'), k + 1);
      this._quad();
      this._blur(this.mTex, this.mHiFbo, this.pingTex, this.pingFbo, 0.65);
      for (let i = 1; i < pyramid.length; i++) {
        const prev = pyramid[i - 1], level = pyramid[i];
        reduce(prev.source, level.sourceFbo, level, prev);
        reduce(prev.mask, level.maskFbo, level, prev);
      }
      for (let i = pyramid.length - 1; i >= 0; i--) {
        const level = pyramid[i], coarse = pyramid[i + 1];
        gl.bindFramebuffer(gl.FRAMEBUFFER, level.resultFbo); gl.viewport(0, 0, level.w, level.h);
        gl.useProgram(this.pSourceFill);
        bind(0, level.source, this.pSourceFill, 'uSource');
        bind(1, coarse?.result || level.source, this.pSourceFill, 'uCoarse');
        gl.uniform1i(gl.getUniformLocation(this.pSourceFill, 'uBase'), !coarse);
        this._quad();
      }
      for (let i = 0; i < pyramid.length; i++) {
        const level = pyramid[i], coarse = pyramid[i + 1] || level;
        gl.bindFramebuffer(gl.FRAMEBUFFER, level.accumFbo); gl.viewport(0, 0, level.w, level.h);
        gl.useProgram(this.pLaplacian);
        bind(0, level.result, this.pLaplacian, 'uFine');
        bind(1, coarse.result, this.pLaplacian, 'uCoarse');
        bind(2, level.mask, this.pLaplacian, 'uMask');
        gl.uniform1i(gl.getUniformLocation(this.pLaplacian, 'uBase'), i === pyramid.length - 1);
        gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD); gl.blendFunc(gl.ONE, gl.ONE);
        this._quad(); gl.disable(gl.BLEND);
      }
    });
    // ---- 4. Normalize each band and reconstruct from coarse to fine -----
    for (let i = pyramid.length - 1; i >= 0; i--) {
      const level = pyramid[i], coarse = pyramid[i + 1];
      gl.bindFramebuffer(gl.FRAMEBUFFER, level.resultFbo); gl.viewport(0, 0, level.w, level.h);
      gl.useProgram(this.pReconstruct);
      bind(0, level.accum, this.pReconstruct, 'uBand');
      bind(1, coarse?.result || level.accum, this.pReconstruct, 'uCoarse');
      gl.uniform1i(gl.getUniformLocation(this.pReconstruct, 'uBase'), !coarse);
      this._quad();
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.panoFbo); gl.viewport(0, 0, this.size, this.h);
    gl.useProgram(this.pFinish);
    bind(0, pyramid[0].result, this.pFinish, 'uResult');
    bind(1, this.avgTex, this.pFinish, 'uCoverage');
    this._quad();

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.activeTexture(gl.TEXTURE0);
    this._composited = true;
  }
}

// stitch rotations are row-major camera->world. uniformMatrix3fv reads
// column-major, so pass the element-transpose: GLSL `uRot` then equals R and
// `transpose(uRot) * world` = Rᵀ * world = the world dir in camera space.
function matT3col(m) {
  return new Float32Array([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]);
}
