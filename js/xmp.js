// Google Photo Sphere (GPano) XMP metadata + JPEG segment injection.
// A JPEG carrying GPano XMP (plus EXIF GPS from exif.js) is accepted by
// Google Street View / Google Maps, Google Photos, Facebook and VR viewers
// as a true 360° panorama.

export function buildGPanoXMP({
  width, height, headingDeg = 0, pitchDeg = 0, rollDeg = 0,
  croppedLeft = 0, croppedTop = 0, croppedW, croppedH,
}) {
  const fw = Math.round(width);
  const fh = Math.round(height);
  const cw = Math.round(croppedW ?? width);
  const ch = Math.round(croppedH ?? height);
  const deg = (v) => (((v % 360) + 360) % 360).toFixed(1);
  const sdeg = (v) => Math.max(-90, Math.min(90, v)).toFixed(1);
  return `<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="PhotoSphere Web">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:GPano="http://ns.google.com/photos/1.0/panorama/"
    GPano:ProjectionType="equirectangular"
    GPano:UsePanoramaViewer="True"
    GPano:CaptureSoftware="PhotoSphere Web"
    GPano:StitchingSoftware="PhotoSphere Web"
    GPano:CroppedAreaImageWidthPixels="${cw}"
    GPano:CroppedAreaImageHeightPixels="${ch}"
    GPano:FullPanoWidthPixels="${fw}"
    GPano:FullPanoHeightPixels="${fh}"
    GPano:CroppedAreaLeftPixels="${Math.round(croppedLeft)}"
    GPano:CroppedAreaTopPixels="${Math.round(croppedTop)}"
    GPano:PoseHeadingDegrees="${deg(headingDeg)}"
    GPano:PosePitchDegrees="${sdeg(pitchDeg)}"
    GPano:PoseRollDegrees="${sdeg(rollDeg)}"/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>`;
}

const XMP_NS = 'http://ns.adobe.com/xap/1.0/\0';

function makeXmpSegment(xmpString) {
  const enc = new TextEncoder();
  const ns = enc.encode(XMP_NS);
  const xmp = enc.encode(xmpString);
  const payload = ns.length + xmp.length;
  if (payload + 2 > 0xffff) throw new Error('XMP packet too large for one APP1 segment');
  const seg = new Uint8Array(4 + payload);
  seg[0] = 0xff; seg[1] = 0xe1;
  seg[2] = ((payload + 2) >> 8) & 0xff;
  seg[3] = (payload + 2) & 0xff;
  seg.set(ns, 4);
  seg.set(xmp, 4 + ns.length);
  return seg;
}

// Splice metadata segments into a JPEG, after SOI + APP0 and any existing
// APP1s. EXIF is written before XMP, as readers expect.
export async function embedMetadata(jpegBlob, { exifSegment = null, xmpString = null }) {
  const buf = new Uint8Array(await jpegBlob.arrayBuffer());
  if (buf[0] !== 0xff || buf[1] !== 0xd8) throw new Error('Not a JPEG file');

  let offset = 2;
  while (buf[offset] === 0xff && (buf[offset + 1] === 0xe0 || buf[offset + 1] === 0xe1)) {
    offset += 2 + ((buf[offset + 2] << 8) | buf[offset + 3]);
  }

  const segs = [];
  if (exifSegment) segs.push(exifSegment);
  if (xmpString) segs.push(makeXmpSegment(xmpString));
  let add = 0;
  for (const s of segs) add += s.length;

  const out = new Uint8Array(buf.length + add);
  out.set(buf.subarray(0, offset), 0);
  let o = offset;
  for (const s of segs) { out.set(s, o); o += s.length; }
  out.set(buf.subarray(offset), o);
  return new Blob([out], { type: 'image/jpeg' });
}

// Back-compat helper (XMP only).
export const injectXMP = (blob, xmpString) => embedMetadata(blob, { xmpString });
