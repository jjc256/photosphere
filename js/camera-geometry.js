// Feature pixels are top-left/down; poses and WebGL use right/up/-Z.
// Keep normalized film coordinates in the device camera's axes, including
// the user's quarter-turn correction for a sideways video feed.
export function filmPoint(x, y, shot) {
  const scale = Math.max(shot.w, shot.h);
  const sx = (x - shot.w / 2) / scale;
  const sy = (shot.h / 2 - y) / scale;
  const a = (shot.vidRot || 0) * Math.PI / 2;
  return [Math.cos(a) * sx + Math.sin(a) * sy,
    -Math.sin(a) * sx + Math.cos(a) * sy];
}

export function filmSize(shot) {
  return (shot.vidRot || 0) % 2 ? [shot.h, shot.w] : [shot.w, shot.h];
}

export function renderCamera(shot, camera, focal) {
  const scale = Math.max(shot.w, shot.h);
  const [w, h] = filmSize(shot);
  const a = (shot.vidRot || 0) * Math.PI / 2;
  const sx = Math.cos(a) * camera.cx - Math.sin(a) * camera.cy;
  const sy = Math.sin(a) * camera.cx + Math.cos(a) * camera.cy;
  return {
    focalScale: camera.focalScale,
    // Texture uploads flip Y, so these are bottom-left UV coordinates.
    center: [0.5 + sx * scale / shot.w, 0.5 + sy * scale / shot.h],
    tan: [w / (2 * scale * focal), h / (2 * scale * focal)],
  };
}
