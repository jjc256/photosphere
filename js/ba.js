// Geometry back-end for the stitcher: SO(3) exp/log, a dense linear solver,
// homography RANSAC (for pair verification), IMU-seeded global bundle
// adjustment over camera rotations + one shared focal length, and gain
// compensation. Pure JS / Node-testable.

// ---- small dense linear algebra ----------------------------------------------
export function solveSPD(A, b, n) {
  // Solve A x = b for symmetric A via Gaussian elimination w/ partial pivot.
  const M = A.map((r) => r.slice());
  const x = b.slice();
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
    if (Math.abs(M[p][i]) < 1e-12) { M[i][i] += 1e-9; p = i; }
    [M[i], M[p]] = [M[p], M[i]];
    [x[i], x[p]] = [x[p], x[i]];
    const piv = M[i][i];
    for (let r = i + 1; r < n; r++) {
      const f = M[r][i] / piv;
      if (f === 0) continue;
      for (let c = i; c < n; c++) M[r][c] -= f * M[i][c];
      x[r] -= f * x[i];
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let c = i + 1; c < n; c++) s -= M[i][c] * x[c];
    x[i] = s / M[i][i];
  }
  return x;
}

// ---- SO(3) (all matrices row-major, active right-handed rotations) --------
export function expSO3(w) {
  const th = Math.hypot(w[0], w[1], w[2]);
  if (th < 1e-9) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const x = w[0] / th, y = w[1] / th, z = w[2] / th;
  const c = Math.cos(th), s = Math.sin(th), C = 1 - c;
  return [
    c + x * x * C, x * y * C - z * s, x * z * C + y * s,
    x * y * C + z * s, c + y * y * C, y * z * C - x * s,
    x * z * C - y * s, y * z * C + x * s, c + z * z * C,
  ];
}
export function logSO3(R) {
  const tr = R[0] + R[4] + R[8];
  const cos = Math.min(1, Math.max(-1, (tr - 1) / 2));
  const th = Math.acos(cos);
  if (th < 1e-7) return [0, 0, 0];
  const k = th / (2 * Math.sin(th));
  return [k * (R[7] - R[5]), k * (R[2] - R[6]), k * (R[3] - R[1])];
}
// quaternion (x,y,z,w) -> row-major rotation, consistent with matVec3.
export function qToR(q) {
  const [x, y, z, w] = q;
  const xx = x * x, yy = y * y, zz = z * z;
  return [
    1 - 2 * (yy + zz), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (xx + zz), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (xx + yy),
  ];
}
export function rToQ(R) {
  const tr = R[0] + R[4] + R[8];
  let x, y, z, w;
  if (tr > 0) {
    const s = 0.5 / Math.sqrt(tr + 1);
    w = 0.25 / s; x = (R[7] - R[5]) * s; y = (R[2] - R[6]) * s; z = (R[3] - R[1]) * s;
  } else if (R[0] > R[4] && R[0] > R[8]) {
    const s = 2 * Math.sqrt(1 + R[0] - R[4] - R[8]);
    w = (R[7] - R[5]) / s; x = 0.25 * s; y = (R[1] + R[3]) / s; z = (R[2] + R[6]) / s;
  } else if (R[4] > R[8]) {
    const s = 2 * Math.sqrt(1 + R[4] - R[0] - R[8]);
    w = (R[2] - R[6]) / s; x = (R[1] + R[3]) / s; y = 0.25 * s; z = (R[5] + R[7]) / s;
  } else {
    const s = 2 * Math.sqrt(1 + R[8] - R[0] - R[4]);
    w = (R[3] - R[1]) / s; x = (R[2] + R[6]) / s; y = (R[5] + R[7]) / s; z = 0.25 * s;
  }
  const n = Math.hypot(x, y, z, w) || 1;
  return [x / n, y / n, z / n, w / n];
}
export const matMul3 = (A, B) => {
  const C = new Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
    C[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
  return C;
};
export const matT3 = (A) => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
export const matVec3 = (A, v) => [
  A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
  A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
  A[6] * v[0] + A[7] * v[1] + A[8] * v[2],
];

// ---- homography RANSAC (pair inlier check) --------------------------------
// Smallest-eigenvalue eigenvector of a symmetric n×n matrix via cyclic Jacobi.
function smallestEigenvector(M, n) {
  const a = M.map((r) => r.slice());
  const V = Array.from({ length: n }, (_, i) => { const r = new Array(n).fill(0); r[i] = 1; return r; });
  for (let sweep = 0; sweep < 40; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-18) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-15) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq;
        V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  let best = 0;
  for (let i = 1; i < n; i++) if (a[i][i] < a[best][best]) best = i;
  return V.map((r) => r[best]);
}

function homographyDLT(pts) {
  // pts: [[x1,y1,x2,y2], ...]; normalized already. Solve A h = 0.
  const AtA = Array.from({ length: 9 }, () => new Array(9).fill(0));
  const add = (row) => { for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) AtA[i][j] += row[i] * row[j]; };
  for (const [x, y, u, v] of pts) {
    add([-x, -y, -1, 0, 0, 0, u * x, u * y, u]);
    add([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
  }
  return smallestEigenvector(AtA, 9);
}
export function ransacHomography(ptsA, ptsB, { iters = 400, thresh = 3, seed = 12345 } = {}) {
  const n = ptsA.length;
  if (n < 8) return { H: null, inliers: [] };
  // normalize
  const norm = (pts) => {
    let mx = 0, my = 0;
    for (const p of pts) { mx += p[0]; my += p[1]; }
    mx /= n; my /= n;
    let d = 0;
    for (const p of pts) d += Math.hypot(p[0] - mx, p[1] - my);
    const s = (Math.SQRT2 * n) / (d || 1);
    return { T: [s, 0, -s * mx, 0, s, -s * my, 0, 0, 1], s, mx, my };
  };
  const na = norm(ptsA), nb = norm(ptsB);
  const ap = ptsA.map((p) => [na.s * (p[0] - na.mx), na.s * (p[1] - na.my)]);
  const bp = ptsB.map((p) => [nb.s * (p[0] - nb.mx), nb.s * (p[1] - nb.my)]);

  let rng = seed >>> 0;
  const rand = () => ((rng = (Math.imul(rng, 1664525) + 1013904223) >>> 0) / 4294967296);
  const thN = thresh * na.s;

  let bestIn = [];
  for (let it = 0; it < iters; it++) {
    const idx = new Set();
    while (idx.size < 4) idx.add((rand() * n) | 0);
    const s = [...idx].map((i) => [ap[i][0], ap[i][1], bp[i][0], bp[i][1]]);
    let H;
    try { H = homographyDLT(s); } catch { continue; }
    const inl = [];
    for (let i = 0; i < n; i++) {
      const X = ap[i][0], Y = ap[i][1];
      const w = H[6] * X + H[7] * Y + H[8];
      if (Math.abs(w) < 1e-9) continue;
      const u = (H[0] * X + H[1] * Y + H[2]) / w;
      const v = (H[3] * X + H[4] * Y + H[5]) / w;
      if ((u - bp[i][0]) ** 2 + (v - bp[i][1]) ** 2 < thN * thN) inl.push(i);
    }
    if (inl.length > bestIn.length) { bestIn = inl; if (inl.length > 0.8 * n) break; }
  }

  // refit on all inliers, return H mapping (centered) A -> B in pixel units
  let H = null;
  if (bestIn.length >= 6) {
    const s = bestIn.map((i) => [ap[i][0], ap[i][1], bp[i][0], bp[i][1]]);
    try {
      const Hn = homographyDLT(s);
      const TaI = [na.s, 0, -na.s * na.mx, 0, na.s, -na.s * na.my, 0, 0, 1]; // A_centered -> A_norm
      const TbInv = [1 / nb.s, 0, nb.mx, 0, 1 / nb.s, nb.my, 0, 0, 1];       // B_norm -> B_centered
      H = matMul3(matMul3(TbInv, Hn), TaI);
      if (Math.abs(H[8]) > 1e-12) H = H.map((v) => v / H[8]);
    } catch { H = null; }
  }
  return { H, inliers: bestIn };
}

// Focal length(s) implied by a pure-rotation homography (OpenCV
// focalsFromHomography). H maps centered pixels of image A -> image B.
export function focalFromHomography(H) {
  const h = H;
  const out = [];
  let d1 = h[6] * h[7];
  let d2 = (h[7] - h[6]) * (h[7] + h[6]);
  let v1 = d1 !== 0 ? -(h[0] * h[1] + h[3] * h[4]) / d1 : NaN;
  let v2 = d2 !== 0 ? (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2 : NaN;
  if (v1 < v2) { const t = v1; v1 = v2; v2 = t; }
  let f1 = NaN;
  if (v1 > 0 && v2 > 0) f1 = Math.sqrt(Math.abs(d1) > Math.abs(d2) ? v1 : v2);
  else if (v1 > 0) f1 = Math.sqrt(v1);

  d1 = h[0] * h[3] + h[1] * h[4];
  d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];
  v1 = d1 !== 0 ? -h[2] * h[5] / d1 : NaN;
  v2 = d2 !== 0 ? (h[5] * h[5] - h[2] * h[2]) / d2 : NaN;
  if (v1 < v2) { const t = v1; v1 = v2; v2 = t; }
  let f0 = NaN;
  if (v1 > 0 && v2 > 0) f0 = Math.sqrt(Math.abs(d1) > Math.abs(d2) ? v1 : v2);
  else if (v1 > 0) f0 = Math.sqrt(v1);

  if (isFinite(f0) && f0 > 0) out.push(f0);
  if (isFinite(f1) && f1 > 0) out.push(f1);
  return out;
}

// Relative rotation (cam_i ray -> cam_j ray, both -Z) implied by a
// pure-rotation homography H (centered pixels, image i -> image j), given the
// shared focal f. Returns a proper rotation via Gram-Schmidt.
export function relRotFromHomography(H, f) {
  // M = K⁻¹ H K, K = diag(f, f, 1) on centered coords  ->  M ∝ R⁻¹? see below.
  // H maps ray_i pixels to ray_j pixels, so K⁻¹ H K maps cam_i normalised
  // coords to cam_j, i.e. it is Rrel acting on [x,y,1]-style coords. We built
  // rays with z = -1, but H was fit on (x,y) only, so the ±z ambiguity is
  // resolved by forcing a right-handed rotation and letting refineRelRot polish.
  const M = [
    H[0], H[1], H[2] / f,
    H[3], H[4], H[5] / f,
    H[6] * f, H[7] * f, H[8],
  ];
  const det =
    M[0] * (M[4] * M[8] - M[5] * M[7]) -
    M[1] * (M[3] * M[8] - M[5] * M[6]) +
    M[2] * (M[3] * M[7] - M[4] * M[6]);
  const s = det < 0 ? -1 : 1;
  let c0 = [s * M[0], s * M[3], s * M[6]];
  let c1 = [s * M[1], s * M[4], s * M[7]];
  const nrm = (v) => { const n = Math.hypot(...v) || 1; return [v[0] / n, v[1] / n, v[2] / n]; };
  const u0 = nrm(c0);
  const d = c1[0] * u0[0] + c1[1] * u0[1] + c1[2] * u0[2];
  const u1 = nrm([c1[0] - d * u0[0], c1[1] - d * u0[1], c1[2] - d * u0[2]]);
  const u2 = [
    u0[1] * u1[2] - u0[2] * u1[1],
    u0[2] * u1[0] - u0[0] * u1[2],
    u0[0] * u1[1] - u0[1] * u1[0],
  ];
  // columns u0,u1,u2 -> row-major
  return [u0[0], u1[0], u2[0], u0[1], u1[1], u2[1], u0[2], u1[2], u2[2]];
}

// ---- lens model -----------------------------------------------------------
let generalizedKernel = null;
export function setGeneralizedLensKernel(kernel) { generalizedKernel = kernel; }
const thetaToRadius = (theta, linearity) => generalizedKernel
  ? generalizedKernel.theta_to_radius(theta, linearity)
  : (Math.abs(linearity) < 1e-8 ? theta : Math.tan(theta * linearity) / linearity);
const radiusToTheta = (radius, linearity) => generalizedKernel
  ? generalizedKernel.radius_to_theta(radius, linearity)
  : (Math.abs(linearity) < 1e-8 ? radius : Math.atan(radius * linearity) / linearity);

// Panorama's PTLens polynomial is expressed on radius, not as a scale factor:
// r_d = r + a*r^2 + b*r^3 + c*r^4.
export function distortPt(x, y, a = 0, b = 0, c = 0) {
  const r = Math.hypot(x, y);
  if (r < 1e-12 || (!a && !b && !c)) return [x, y];
  const rd = generalizedKernel ? generalizedKernel.ptlens_distort(r, a, b, c)
    : r + a * r * r + b * r * r * r + c * r * r * r * r;
  return [x * rd / r, y * rd / r];
}

export function undistortPt(x, y, a = 0, b = 0, c = 0) {
  const rd = Math.hypot(x, y);
  if (rd < 1e-12 || (!a && !b && !c)) return [x, y];
  let r = generalizedKernel ? generalizedKernel.ptlens_undistort(rd, a, b, c) : rd;
  if (!generalizedKernel) for (let i = 0; i < 10; i++) {
    const r2 = r * r;
    const value = r + a * r2 + b * r2 * r + c * r2 * r2 - rd;
    const deriv = 1 + 2 * a * r + 3 * b * r2 + 4 * c * r2 * r;
    if (Math.abs(deriv) < 1e-10) break;
    r -= value / deriv;
  }
  return [x * r / rd, y * r / rd];
}

// Panorama applies PTLens in film coordinates, *after* multiplying the
// generalized ray radius by focal length.  Keeping that order matters whenever
// focal is calibrated: applying the polynomial to x/f (the former browser
// implementation) describes a different camera.
function rayFromFilm(x, y, focal, k1, k2, k3, linearity) {
  const u = undistortPt(x, y, k1, k2, k3);
  const r = Math.hypot(u[0], u[1]);
  if (r < 1e-12) return [0, 0, -1];
  const theta = radiusToTheta(r / focal, linearity);
  const s = Math.sin(theta) / r;
  return [u[0] * s, u[1] * s, -Math.cos(theta)];
}

function filmFromRay(d, focal, k1, k2, k3, linearity) {
  const n = Math.hypot(d[0], d[1], d[2]) || 1;
  const theta = Math.acos(Math.max(-1, Math.min(1, -d[2] / n)));
  const xy = Math.hypot(d[0], d[1]);
  if (xy < 1e-12) return [0, 0];
  const r = thetaToRadius(theta, linearity) * focal;
  return distortPt(d[0] * r / xy, d[1] * r / xy, k1, k2, k3);
}

// Gauss-Newton refine of ONE relative rotation Rrel (cam_i ray -> cam_j ray)
// so that x_j ≈ project(Rrel · Kinv x_i). m0: [[xi,yi,xj,yj]] centered pixels.
// Runs LM, drops gross mismatches that slipped past homography RANSAC
// (periodic-texture false positives), then re-runs. Returns rms/inl on the
// surviving inlier set.
export function refineRelRot(Rrel0, m0, f, { iters = 20, k1 = 0, k2 = 0, k3 = 0, linearity = 1 } = {}) {
  // measured pixels are distorted: lift to ideal rays, and push predictions
  // back through the lens before comparing with the measured pixel.
  const rays = new Map();
  const ray = (m) => {
    let r = rays.get(m);
    if (!r) { r = rayFromFilm(m[0], m[1], f, k1, k2, k3, linearity); rays.set(m, r); }
    return r;
  };
  const perErr = (R, m) => {
    const e = new Float64Array(m.length);
    const Ri = matT3(R);
    for (let k = 0; k < m.length; k++) {
      const d = matVec3(R, ray(m[k]));
      const z = Math.min(d[2], -0.05);
      const p = filmFromRay([d[0], d[1], z], f, k1, k2, k3, linearity);
      const back = matVec3(Ri, ray([m[k][2], m[k][3]]));
      const bz = Math.min(back[2], -0.05);
      const q = filmFromRay([back[0], back[1], bz], f, k1, k2, k3, linearity);
      e[k] = Math.hypot(p[0] - m[k][2], p[1] - m[k][3], q[0] - m[k][0], q[1] - m[k][1]);
    }
    return e;
  };
  // Plain L2 Levenberg-Marquardt (no residual clamping — that flattens the
  // cost surface and stalls the solve; outliers are handled by re-filtering).
  const lm = (Rin, m) => {
    let R = Rin.slice();
    const resid = (Rr) => {
      const out = new Float64Array(m.length * 4);
      const Ri = matT3(Rr);
      for (let k = 0; k < m.length; k++) {
        const d = matVec3(Rr, ray(m[k]));
        const z = Math.min(d[2], -0.05);
        const p = filmFromRay([d[0], d[1], z], f, k1, k2, k3, linearity);
        const back = matVec3(Ri, ray([m[k][2], m[k][3]]));
        const bz = Math.min(back[2], -0.05);
        const q = filmFromRay([back[0], back[1], bz], f, k1, k2, k3, linearity);
        out[k * 4] = p[0] - m[k][2];
        out[k * 4 + 1] = p[1] - m[k][3];
        out[k * 4 + 2] = q[0] - m[k][0];
        out[k * 4 + 3] = q[1] - m[k][1];
      }
      return out;
    };
    const sq = (a) => { let s = 0; for (const v of a) s += v * v; return s; };
    let r0 = resid(R), c0 = sq(r0), lambda = 1e-3;
    const EPS = 1e-5, M = r0.length;
    for (let it = 0; it < iters; it++) {
      const J = [new Float64Array(M), new Float64Array(M), new Float64Array(M)];
      for (let p = 0; p < 3; p++) {
        const w = [0, 0, 0]; w[p] = EPS;
        const rp = resid(matMul3(expSO3(w), R));
        for (let k = 0; k < M; k++) J[p][k] = (rp[k] - r0[k]) / EPS;
      }
      const H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], g = [0, 0, 0];
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 3; b++) { let s = 0; for (let k = 0; k < M; k++) s += J[a][k] * J[b][k]; H[a][b] = s; }
        let s = 0; for (let k = 0; k < M; k++) s += J[a][k] * r0[k]; g[a] = -s;
      }
      let ok = false;
      for (let t = 0; t < 7; t++) {
        const Hd = H.map((row, i) => row.map((v, j) => (i === j ? v + lambda * (v + 1) : v)));
        const cand = matMul3(expSO3(solveSPD(Hd, g, 3)), R);
        const cc = sq(resid(cand));
        if (cc < c0 - 1e-9) { R = cand; r0 = resid(R); c0 = cc; lambda = Math.max(lambda * 0.3, 1e-10); ok = true; break; }
        lambda = Math.min(lambda * 5, 1e7);
      }
      if (!ok) break;
    }
    return R;
  };

  // IRLS-style outer loop: solve, drop gross mismatches, resolve.
  let R = lm(Rrel0, m0);
  let m = m0;
  for (let round = 0; round < 3; round++) {
    const e = perErr(R, m);
    const med = [...e].sort((a, b) => a - b)[e.length >> 1] || 1;
    const cut = Math.max(2.5, 3 * med);
    const keep = m.filter((_, k) => e[k] < cut);
    if (keep.length < 12) return { Rrel: R, inl: keep.length, rms: 99 };
    if (keep.length === m.length) break;
    m = keep;
    R = lm(R, m);
  }
  const fe = perErr(R, m);
  return { Rrel: R, inl: m.length, rms: Math.hypot(...fe) / Math.sqrt(m.length) };
}

// RANSAC directly in the calibrated camera model. Homographies cannot describe
// wide-angle rays well enough at the frame edge, so use them only for an
// optional focal vote and make geometric reprojection the acceptance test.
export function ransacGeneralizedRotation(matches, f, seedR, {
  k1 = 0, k2 = 0, k3 = 0, linearity = 1, iters = 220, thresh = 4, seed = 0x9e3779b9,
} = {}) {
  if (matches.length < 40) return { Rrel: seedR, inliers: [] };
  let state = seed >>> 0;
  const rnd = () => { state = (state * 1664525 + 1013904223) >>> 0; return state; };
  const errors = (R) => matches.map((m) => {
    const d = matVec3(R, rayFromFilm(m[0], m[1], f, k1, k2, k3, linearity));
    const p = filmFromRay(d, f, k1, k2, k3, linearity);
    const Ri = matT3(R);
    const back = matVec3(Ri, rayFromFilm(m[2], m[3], f, k1, k2, k3, linearity));
    const q = filmFromRay(back, f, k1, k2, k3, linearity);
    return Math.hypot(p[0] - m[2], p[1] - m[3], q[0] - m[0], q[1] - m[1]);
  });
  let best = { Rrel: seedR, inliers: [] };
  for (let it = 0; it < iters; it++) {
    const ids = new Set();
    while (ids.size < 4) ids.add(rnd() % matches.length);
    const sample = [...ids].map((i) => matches[i]);
    const fit = refineRelRot(seedR, sample, f, { k1, k2, k3, linearity, iters: 7 });
    const inliers = [];
    for (const [i, e] of errors(fit.Rrel).entries()) if (e < thresh) inliers.push(i);
    if (inliers.length > best.inliers.length) best = { Rrel: fit.Rrel, inliers };
  }
  if (best.inliers.length < 25) return best;
  const refined = refineRelRot(best.Rrel, best.inliers.map((i) => matches[i]), f,
    { k1, k2, k3, linearity, iters: 16 });
  const finalErr = errors(refined.Rrel);
  const inliers = [];
  // `thresh` is expressed in the caller's film-coordinate unit. Panorama
  // normalises points by the longer image edge, so a pixel-sized hard floor
  // here would silently admit every outlier after the final RANSAC pass.
  for (const [i, e] of finalErr.entries()) if (e < thresh * 0.72) inliers.push(i);
  return { Rrel: refined.Rrel, inliers };
}

// Iterative L2 rotation averaging. edges: [{ i, j, Rrel, w }] where the model
// is R_j = R_i · Rrelᵀ. priorR: absolute IMU rotation per node. Returns [R_k].
export function rotationAverage(N, edges, priorR, { priorW = 0.25, iters = 40 } = {}) {
  const R = priorR.map((m) => m.slice());
  const adj = Array.from({ length: N }, () => []);
  edges.forEach((e, idx) => { adj[e.i].push(idx); adj[e.j].push(idx); });

  for (let it = 0; it < iters; it++) {
    let maxStep = 0;
    for (let k = 0; k < N; k++) {
      const qs = [[priorW, rToQ(priorR[k])]];
      for (const ei of adj[k]) {
        const e = edges[ei];
        const w = e.w || 1;
        if (e.i === k) qs.push([w, rToQ(matMul3(R[e.j], e.Rrel))]);          // R_i = R_j · Rrel
        else qs.push([w, rToQ(matMul3(R[e.i], matT3(e.Rrel)))]);            // R_j = R_i · Rrelᵀ
      }
      const q = weightedQuatAvg(qs);
      const before = R[k];
      R[k] = qToR(q);
      maxStep = Math.max(maxStep, Math.hypot(...logSO3(matMul3(R[k], matT3(before)))));
    }
    if (maxStep < 1e-5) break;
  }
  return R;
}

function weightedQuatAvg(list) {
  // sign-align to the first, weighted sum, renormalise (valid for tight clusters)
  const ref = list[0][1];
  let x = 0, y = 0, z = 0, w = 0;
  for (const [wt, q] of list) {
    const s = (q[0] * ref[0] + q[1] * ref[1] + q[2] * ref[2] + q[3] * ref[3]) < 0 ? -wt : wt;
    x += s * q[0]; y += s * q[1]; z += s * q[2]; w += s * q[3];
  }
  const n = Math.hypot(x, y, z, w) || 1;
  return [x / n, y / n, z / n, w / n];
}

// ---- IMU-seeded global bundle adjustment ----------------------------------
// frames: [{ w, h }]  (all same size in practice)
// gyroR:  [mat3 camera->world]  one per frame  (initial guess)
// pairs:  [{ i, j, m: [[xi,yi,xj,yj], ...] }]  verified inlier matches
// Returns { R: [mat3], focal, cost }
export function bundleAdjust(frames, gyroR, pairs, {
  focal0, cx, cy, k1 = 0, k2 = 0, k3 = 0, linearity = 1, optimizeFocal = true,
  optimizeLens = false, optimizeDistortion = false, optimizePerFrame = false, optimizePerFrameCenter = false,
  priorW = 0.05, huber = 12, iters = 40, anchor = 0,
  normalSolver = null,
} = {}) {
  const N = frames.length;
  const theta = new Float64Array(6 * N + 7); // rotations + shared lens + per-frame f/cx/cy
  const FI = 3 * N;                           // focal-scale param index
  const LI = FI + 1, K1I = FI + 2, K2I = FI + 3, K3I = FI + 4, CXI = FI + 5, CYI = FI + 6;
  const PFI = FI + 7, PCXI = PFI + N, PCYI = PCXI + N;

  // Active params: every rotation triple except the anchor frame's, + focal.
  // Fixing one frame removes the global-rotation gauge freedom so the normal
  // equations are full rank.
  const active = [];
  for (let k = 0; k < N; k++) if (k !== anchor) active.push(k * 3, k * 3 + 1, k * 3 + 2);
  if (optimizeFocal) active.push(FI);
  // Panorama's final bundle-adjustment pass unlocks generalized linearity and
  // focal length, but leaves PTLens fixed.  Unlocking both from one sparse
  // graph makes them trade off against each other and is what caused the
  // unstable lens jump in the browser replay.
  if (optimizeLens) active.push(LI);
  if (optimizeDistortion) active.push(K1I, K2I, K3I);
  const used = new Set(pairs.flatMap((p) => [p.i, p.j]));
  if (optimizePerFrame) for (let k = 0; k < N; k++) if (used.has(k)) {
    active.push(PFI + k);
    if (optimizePerFrameCenter) active.push(PCXI + k, PCYI + k);
  }
  const P = active.length;

  const curR = (t) => {
    const R = [];
    for (let k = 0; k < N; k++) R.push(matMul3(expSO3([t[k * 3], t[k * 3 + 1], t[k * 3 + 2]]), gyroR[k]));
    return R;
  };

  const residuals = (t) => {
    const R = curR(t);
    const f = focal0 * Math.exp(t[FI]);
    const ll = Math.max(0.2, Math.min(1.8, linearity + t[LI]));
    const kk1 = k1 + t[K1I], kk2 = k2 + t[K2I], kk3 = k3 + t[K3I];
    const ccx = cx + t[CXI], ccy = cy + t[CYI];
    const cam = (k) => ({ f: f * Math.exp(t[PFI + k]), cx: ccx + t[PCXI + k], cy: ccy + t[PCYI + k] });
    const res = [];
    for (const pr of pairs) {
      const Rrel = matMul3(matT3(R[pr.j]), R[pr.i]); // cam i -> world -> cam j
      const Rinv = matT3(Rrel);
      const ci = cam(pr.i), cj = cam(pr.j);
      for (const [xi, yi, xj, yj] of pr.m) {
        const u = rayFromFilm(xi - ci.cx, yi - ci.cy, ci.f, kk1, kk2, kk3, ll);
        const d = matVec3(Rrel, u); // camera looks -Z
        const z = Math.min(d[2], -0.05);              // smooth clamp (no behind-camera spikes)
        const p = filmFromRay([d[0], d[1], z], cj.f, kk1, kk2, kk3, ll);
        let rx = cj.cx + p[0] - xj;
        let ry = cj.cy + p[1] - yj;
        const v = rayFromFilm(xj - cj.cx, yj - cj.cy, cj.f, kk1, kk2, kk3, ll);
        const back = matVec3(Rinv, v);
        const bz = Math.min(back[2], -0.05);
        const q = filmFromRay([back[0], back[1], bz], ci.f, kk1, kk2, kk3, ll);
        let bx = ci.cx + q[0] - xi;
        let by = ci.cy + q[1] - yi;
        const r = Math.hypot(rx, ry, bx, by);
        if (r > huber) { const s = Math.sqrt(huber / r); rx *= s; ry *= s; }
        if (r > huber) { const s = Math.sqrt(huber / r); bx *= s; by *= s; }
        res.push(rx, ry, bx, by);
      }
    }
    for (let k = 0; k < N; k++) {
      res.push(priorW * t[k * 3], priorW * t[k * 3 + 1], priorW * t[k * 3 + 2]);
    }
    if (optimizeFocal) res.push(0.15 * t[FI]); // gentle focal prior
    if (optimizeLens) res.push(0.2 * t[LI]);
    if (optimizeDistortion) res.push(0.15 * t[K1I], 0.15 * t[K2I], 0.15 * t[K3I]);
    if (optimizePerFrame) for (let k = 0; k < N; k++) if (used.has(k)) {
      res.push(0.25 * t[PFI + k]);
      if (optimizePerFrameCenter) res.push(0.04 * t[PCXI + k], 0.04 * t[PCYI + k]);
    }
    return res;
  };

  const cost = (r) => r.reduce((s, v) => s + v * v, 0);
  let r0 = residuals(theta);
  let c0 = cost(r0);
  let lambda = 1e-3;
  const M = r0.length;
  const EPS = 1e-4;

  for (let it = 0; it < iters; it++) {
    // numeric Jacobian over active params only (forward differences)
    const J = Array.from({ length: P }, () => new Float64Array(M));
    for (let p = 0; p < P; p++) {
      const ai = active[p];
      const save = theta[ai];
      theta[ai] = save + EPS;
      const rp = residuals(theta);
      theta[ai] = save;
      const col = J[p];
      for (let m = 0; m < M; m++) col[m] = (rp[m] - r0[m]) / EPS;
    }
    // normal equations
    const H = Array.from({ length: P }, () => new Array(P).fill(0));
    const g = new Array(P).fill(0);
    for (let a = 0; a < P; a++) {
      const Ja = J[a];
      for (let b = a; b < P; b++) {
        const Jb = J[b];
        let s = 0;
        for (let m = 0; m < M; m++) s += Ja[m] * Jb[m];
        H[a][b] = s; H[b][a] = s;
      }
      let s = 0;
      for (let m = 0; m < M; m++) s += Ja[m] * r0[m];
      g[a] = -s;
    }
    let applied = false;
    for (let tries = 0; tries < 7; tries++) {
      const Hd = H.map((row, i) => row.map((v, j) => (i === j ? v + lambda * (v + 1) : v)));
      // The worker supplies a WASM preconditioned-CG implementation. Keep the
      // local JS solver solely for Node replay and WASM-unavailable browsers.
      const dx = normalSolver?.(Hd, g) || solveSPD(Hd, g, P);
      const cand = theta.slice();
      for (let p = 0; p < P; p++) cand[active[p]] += dx[p];
      const rc = residuals(cand);
      const cc = cost(rc);
      if (cc < c0 - 1e-9) {
        theta.set(cand); r0 = rc; c0 = cc; lambda = Math.max(lambda * 0.3, 1e-9);
        applied = true; break;
      }
      lambda = Math.min(lambda * 5, 1e6);
    }
    if (!applied) break;
  }

  const focal = focal0 * Math.exp(theta[FI]), outCx = cx + theta[CXI], outCy = cy + theta[CYI];
  return { R: curR(theta), focal, linearity: linearity + theta[LI],
    k1: k1 + theta[K1I], k2: k2 + theta[K2I], k3: k3 + theta[K3I], cx: outCx, cy: outCy,
    cameras: Array.from({ length: N }, (_, k) => ({ focalScale: Math.exp(theta[PFI + k]), cx: outCx + theta[PCXI + k], cy: outCy + theta[PCYI + k] })), cost: c0 };
}

// ---- gain compensation ---------------------------------------------------------
// ov: [{ i, j, Ii, Ij, n }]  mean intensity of the overlap region seen from each
// side, and overlap pixel count. Solves for per-image gains ~1.
export function gainCompensate(N, ov, { sigmaN = 0.01, sigmaG = 0.1 } = {}) {
  const A = Array.from({ length: N }, () => new Array(N).fill(0));
  const b = new Array(N).fill(0);
  const sn2 = sigmaN * sigmaN, sg2 = sigmaG * sigmaG;
  for (const { i, j, Ii, Ij, n } of ov) {
    A[i][i] += n * (Ii * Ii) / sn2;
    A[j][j] += n * (Ij * Ij) / sn2;
    A[i][j] -= n * (Ii * Ij) / sn2;
    A[j][i] -= n * (Ii * Ij) / sn2;
  }
  for (let i = 0; i < N; i++) { A[i][i] += 1 / sg2; b[i] += 1 / sg2; }
  const g = solveSPD(A, b, N);
  const mean = g.reduce((s, v) => s + v, 0) / N || 1;
  return g.map((v) => v / mean);
}
