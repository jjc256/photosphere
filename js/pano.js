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
uniform sampler2D uFrame;
const float PI = 3.14159265359;
bool warp(out vec3 rgb, out float edge) {
  float lon = (vUv.x - 0.5) * 2.0 * PI;
  float lat = (vUv.y - 0.5) * PI;
  float cl = cos(lat);
  vec3 world = vec3(cl * sin(lon), sin(lat), -cl * cos(lon));
  vec3 cam = transpose(uRot) * world;
  if (cam.z > -1e-4) return false;
  float px = (cam.x / -cam.z) / uTan.x;
  float py = (cam.y / -cam.z) / uTan.y;
  if (max(abs(px), abs(py)) > 1.0) return false;
  vec2 uv = vec2(px, py) * 0.5 + 0.5;
  rgb = clamp(texture(uFrame, uv).rgb * uGain, 0.0, 1.0);
  edge = min(1.0 - abs(px), 1.0 - abs(py));   // distance to nearest frame edge, 0..1
  return true;
}`;

// Seam labelling: nearest-centre (Voronoi) wins via the depth buffer.
// Output R = (frameIndex+1)/255, G = border distance.
const LABEL_FRAG = WARP_HEAD + `
out vec4 frag;
uniform float uIndex;
uniform float uPriority;               // 1 for aligned frames, low for gyro-only
void main() {
  vec3 rgb; float edge;
  if (!warp(rgb, edge)) discard;
  gl_FragDepth = 1.0 - edge * uPriority * 0.999; // higher (border dist * priority) wins
  frag = vec4((uIndex + 1.0) / 255.0, edge, 0.0, 1.0);
}`;

// HARD mosaic: this frame's warped colour, but only on the pixels it owns
// (label winner). No blending -> single source everywhere -> no ghosting.
const HARD_FRAG = WARP_HEAD + `
out vec4 frag;
uniform sampler2D uLabel;
uniform float uIndex;
void main() {
  vec3 rgb; float edge;
  if (!warp(rgb, edge)) discard;
  float lab = texture(uLabel, vUv).r * 255.0;
  if (abs(lab - (uIndex + 1.0)) > 0.5) discard;
  frag = vec4(rgb, 1.0);
}`;

// Feather-weighted accumulate (for the LOW-frequency blend only). This does
// average overlapping frames, but the result is heavily blurred afterwards so
// a 1-3 px misalignment becomes invisible; it just carries smooth colour /
// exposure across seams.
const FA_FRAG = WARP_HEAD + `
out vec4 frag;
void main() {
  vec3 rgb; float edge;
  if (!warp(rgb, edge)) discard;
  float w = edge * edge + 0.02;
  frag = vec4(rgb * w, w);
}`;

// Separable blur (13-tap gaussian, scaled by uStep).
const BLUR_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uSrc;
uniform vec2 uStep;
void main() {
  float wsum = 0.0;
  vec4 acc = vec4(0.0);
  for (int i = -6; i <= 6; i++) {
    float w = exp(-float(i * i) / 18.0);
    acc += texture(uSrc, vUv + uStep * float(i)) * w;
    wsum += w;
  }
  frag = acc / wsum;
}`;

// 2-band combine: high-frequency detail from the HARD (single-source) mosaic
// + low-frequency colour from the feather-blended, heavily-blurred mosaic.
//   result = hard - blur(hard) + blur(featherBlend)
// Away from seams this is exactly the source frame; across a seam the detail
// stays single-source (an occasional visible cut, never a double image) while
// colour/exposure blends smoothly.
const COMBINE2_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 frag;
uniform sampler2D uHard;
uniform sampler2D uHardLo;
uniform sampler2D uLo;
uniform float uFlipY;
void main() {
  vec2 uv = vec2(vUv.x, uFlipY > 0.5 ? 1.0 - vUv.y : vUv.y);
  vec4 hard = texture(uHard, uv);
  vec3 detail = hard.rgb - texture(uHardLo, uv).rgb;
  vec3 lo = texture(uLo, uv).rgb;
  frag = vec4(clamp(detail + lo, 0.0, 1.0), hard.a);
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
  constructor(canvas) {
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

    this.size = Math.min(4096, gl.getParameter(gl.MAX_TEXTURE_SIZE));
    this.h = this.size / 2;

    this.pSplat = program(gl, SPLAT_FRAG);
    this.pPresent = program(gl, PRESENT_FRAG);
    this.pSphere = program(gl, SPHERE_FRAG);
    this.pCoverage = program(gl, COVERAGE_FRAG);
    this.pLabel = program(gl, LABEL_FRAG);
    this.pHard = program(gl, HARD_FRAG);
    this.pFA = program(gl, FA_FRAG);
    this.pBlur = program(gl, BLUR_FRAG);
    this.pNorm2 = program(gl, NORMALIZE2_FRAG);
    this.pCombine2 = program(gl, COMBINE2_FRAG);
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
    gl.bindTexture(gl.TEXTURE_2D, this.accumTex);
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

  // ---- 2-band seam compositor -----------------------------------------
  // Nearest-centre (Voronoi) seam labels -> a HARD single-source mosaic for
  // high-frequency detail (an occasional visible cut, never a double image)
  // + a heavily blurred feather-blend for low-frequency colour/exposure
  // (a few-px misalignment is invisible once blurred). Then fill the caps.
  // Intermediates run at 2048 wide to keep iOS GPU memory sane; the result
  // is upscaled into panoTex.
  _initComposite() {
    if (this._compReady) return;
    const gl = this.gl;
    const cs = Math.min(2048, this.size);
    this.cs = cs; this.csh = cs / 2;
    const hf = this._floatLinear ? gl.LINEAR : gl.NEAREST;
    const t8 = () => this._tex(cs, this.csh, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR, gl.REPEAT);
    this.frameTex = this._tex(4, 4, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR, gl.CLAMP_TO_EDGE);
    this.faTex = this._tex(cs, this.csh, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, hf, gl.REPEAT);
    this.faFbo = this._fbo(this.faTex);
    this.labelTex = this._tex(cs, this.csh, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.NEAREST, gl.REPEAT);
    this.labelFbo = gl.createFramebuffer();
    this.labelDepth = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, this.labelDepth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, cs, this.csh);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.labelFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.labelTex, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.labelDepth);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.hardTex = t8(); this.hardFbo = this._fbo(this.hardTex);
    this.s0 = t8(); this.s0f = this._fbo(this.s0);
    this.s1 = t8(); this.s1f = this._fbo(this.s1);
    this.s2 = t8(); this.s2f = this._fbo(this.s2);
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

  _warpUniforms(prog, uRot, tanX, tanY, gain) {
    const gl = this.gl;
    gl.uniformMatrix3fv(gl.getUniformLocation(prog, 'uRot'), false, uRot);
    gl.uniform2f(gl.getUniformLocation(prog, 'uTan'), tanX, tanY);
    gl.uniform1f(gl.getUniformLocation(prog, 'uGain'), gain);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.frameTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'uFrame'), 0);
  }

  // separable gaussian: srcTex -> dstFbo, ping-ponging through pingFbo/pingTex
  _blur(srcTex, dstFbo, pingTex, pingFbo, spread) {
    const gl = this.gl, w = this.cs, h = this.csh;
    gl.useProgram(this.pBlur); gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, pingFbo);
    gl.viewport(0, 0, w, h);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(gl.getUniformLocation(this.pBlur, 'uSrc'), 0);
    gl.uniform2f(gl.getUniformLocation(this.pBlur, 'uStep'), spread / w, 0);
    this._quad();
    gl.bindFramebuffer(gl.FRAMEBUFFER, dstFbo);
    gl.bindTexture(gl.TEXTURE_2D, pingTex);
    gl.uniform2f(gl.getUniformLocation(this.pBlur, 'uStep'), 0, spread / h);
    this._quad();
  }

  // frames: [{ img, R (row-major camera->world), gain, weak }]. tanX/tanY =
  // tan(fov/2) for the focal-corrected lens. Fills panoTex.
  compositeStitched(frames, tanX, tanY) {
    const gl = this.gl;
    this._initComposite();
    const w = this.cs, h = this.csh;
    const rots = frames.map((f) => matT3col(f.R));

    // pass 1: seam labels (nearest centre wins via depth)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.labelFbo);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.BLEND);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS); gl.depthMask(true);
    gl.clearColor(0, 0, 0, 0); gl.clearDepth(1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.pLabel);
    frames.forEach((fr, k) => {
      this._uploadFrame(fr.img);
      this._warpUniforms(this.pLabel, rots[k], tanX, tanY, fr.gain || 1);
      gl.uniform1f(gl.getUniformLocation(this.pLabel, 'uIndex'), k);
      gl.uniform1f(gl.getUniformLocation(this.pLabel, 'uPriority'), fr.weak ? 0.18 : 1.0);
      this._quad();
    });
    gl.disable(gl.DEPTH_TEST);

    // pass 2a: HARD mosaic (single source per pixel) -> hardTex
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.hardFbo);
    gl.viewport(0, 0, w, h); gl.disable(gl.BLEND);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.pHard);
    frames.forEach((fr, k) => {
      this._uploadFrame(fr.img);
      this._warpUniforms(this.pHard, rots[k], tanX, tanY, fr.gain || 1);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.labelTex);
      gl.uniform1i(gl.getUniformLocation(this.pHard, 'uLabel'), 1);
      gl.uniform1f(gl.getUniformLocation(this.pHard, 'uIndex'), k);
      this._quad();
    });

    // pass 2b: feather-weighted accumulate -> faTex
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.faFbo);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.BLEND);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD); gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(this.pFA);
    frames.forEach((fr, k) => {
      this._uploadFrame(fr.img);
      this._warpUniforms(this.pFA, rots[k], tanX, tanY, fr.gain || 1);
      this._quad();
    });
    gl.disable(gl.BLEND);

    // normalise feather-blend -> s0
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.s0f);
    gl.viewport(0, 0, w, h); gl.disable(gl.BLEND);
    gl.useProgram(this.pNorm2);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.faTex);
    gl.uniform1i(gl.getUniformLocation(this.pNorm2, 'uAccum'), 0);
    this._quad();

    // band split: hardLo = blur(hard) -> s2 ;  lo = blur(featherBlend) -> s0
    const spread = Math.max(10, w / 42);
    this._blur(this.hardTex, this.s2f, this.s1, this.s1f, spread); // -> s2
    this._blur(this.s0, this.s0f, this.s1, this.s1f, spread);      // s0 -> s0

    // combine  detail(hard) + lo  -> s1
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.s1f);
    gl.viewport(0, 0, w, h); gl.disable(gl.BLEND);
    gl.useProgram(this.pCombine2);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.hardTex);
    gl.uniform1i(gl.getUniformLocation(this.pCombine2, 'uHard'), 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.s2);
    gl.uniform1i(gl.getUniformLocation(this.pCombine2, 'uHardLo'), 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.s0);
    gl.uniform1i(gl.getUniformLocation(this.pCombine2, 'uLo'), 2);
    gl.uniform1f(gl.getUniformLocation(this.pCombine2, 'uFlipY'), 0);
    this._quad();

    // fill uncovered caps: s1 -> s0 -> s1
    gl.useProgram(this.pPole);
    gl.uniform2f(gl.getUniformLocation(this.pPole, 'uTexel'), 1 / w, 1 / h);
    gl.uniform1i(gl.getUniformLocation(this.pPole, 'uSrc'), 0);
    let src = this.s1;
    for (let i = 0; i < 3; i++) {
      const dstFbo = src === this.s1 ? this.s0f : this.s1f;
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstFbo);
      gl.viewport(0, 0, w, h); gl.disable(gl.BLEND);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, src);
      this._quad();
      src = src === this.s1 ? this.s0 : this.s1;
    }

    // upscale the finished 2048 pano into panoTex (this.size)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.panoFbo);
    gl.viewport(0, 0, this.size, this.h); gl.disable(gl.BLEND);
    gl.useProgram(this.pBlit);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, src);
    gl.uniform1i(gl.getUniformLocation(this.pBlit, 'uSrc'), 0);
    gl.uniform1f(gl.getUniformLocation(this.pBlit, 'uFlipY'), 0);
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
