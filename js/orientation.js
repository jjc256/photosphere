// Device-orientation → rotation-matrix helpers.
// Camera convention (matches three.js): local axes x=right, y=up, camera looks down -Z.
// All matrices are column-major Float32Array(9), camera->world.

export const DEG = Math.PI / 180;

export function quatFromAxisAngle(x, y, z, angle) {
  const s = Math.sin(angle / 2);
  return [x * s, y * s, z * s, Math.cos(angle / 2)];
}

export function multiplyQuat(a, b) {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

export function normalizeQuat(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

// three.js Euler order 'YXZ'
export function eulerToQuatYXZ(x, y, z) {
  const c1 = Math.cos(x / 2), s1 = Math.sin(x / 2);
  const c2 = Math.cos(y / 2), s2 = Math.sin(y / 2);
  const c3 = Math.cos(z / 2), s3 = Math.sin(z / 2);
  return [
    s1 * c2 * c3 + c1 * s2 * s3,
    c1 * s2 * c3 - s1 * c2 * s3,
    c1 * c2 * s3 - s1 * s2 * c3,
    c1 * c2 * c3 + s1 * s2 * s3,
  ];
}

// alpha/beta/gamma in RADIANS, screenAngle in RADIANS.
// Ported from three.js DeviceOrientationControls.
export function deviceQuat(alpha, beta, gamma, screenAngle) {
  const q = eulerToQuatYXZ(beta, alpha, -gamma);
  const q1 = [-Math.SQRT1_2, 0, 0, Math.SQRT1_2]; // -90° about X: look out the back
  let r = multiplyQuat(q, q1);
  r = multiplyQuat(r, quatFromAxisAngle(0, 0, 1, -screenAngle));
  return normalizeQuat(r);
}

export function quatToMat3(q) {
  const [x, y, z, w] = q;
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  return new Float32Array([
    1 - (yy + zz), xy + wz, xz - wy,
    xy - wz, 1 - (xx + zz), yz + wx,
    xz + wy, yz - wx, 1 - (xx + yy),
  ]);
}

// Sim / desktop mode: build a camera->world matrix from yaw (about Y) then pitch (about X).
export function yawPitchToMat3(yaw, pitch) {
  const qy = quatFromAxisAngle(0, 1, 0, yaw);
  const qx = quatFromAxisAngle(1, 0, 0, pitch);
  return quatToMat3(normalizeQuat(multiplyQuat(qy, qx)));
}

// Smallest angle (radians) between two orientations.
export function quatAngle(a, b) {
  const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  return 2 * Math.acos(Math.min(1, d));
}

// Forward direction (world) for a camera->world matrix: R * (0,0,-1).
export function forwardDir(R) {
  return [-R[6], -R[7], -R[8]];
}

// Is a world-space unit vector inside the camera frustum? (mirrors the splat shader)
export function inFrustum(R, wx, wy, wz, tanX, tanY) {
  const cx = R[0] * wx + R[1] * wy + R[2] * wz;
  const cy = R[3] * wx + R[4] * wy + R[5] * wz;
  const cz = R[6] * wx + R[7] * wy + R[8] * wz;
  if (cz > -1e-4) return false;
  const px = (cx / -cz) / tanX;
  const py = (cy / -cz) / tanY;
  return Math.abs(px) <= 1 && Math.abs(py) <= 1;
}
