// Minimal big-endian ("MM") EXIF APP1 builder for 360° photos headed to
// Google Street View / Google Maps.
//
// canvas.toBlob('image/jpeg') writes no EXIF at all, so Google has no way to
// place the sphere on the map. This adds the tags it actually reads:
//   - GPS position (lat / lon / altitude)      -> where the photo goes
//   - DateTimeOriginal + GPS date/time (UTC)   -> when it was taken
//   - GPSImgDirection                          -> which way is "forward"
//   - pixel dimensions, colour space, software
//
// Returns the complete APP1 segment (FF E1 <len> "Exif\0\0" <TIFF>).

const ASCII = 2, SHORT = 3, LONG = 4, RATIONAL = 5, BYTE = 1, UNDEFINED = 7;
const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 10: 8 };

const pad = (n) => String(n).padStart(2, '0');
const fmtDateTime = (d) =>
  `${d.getFullYear()}:${pad(d.getMonth() + 1)}:${pad(d.getDate())} ` +
  `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
const fmtDateUTC = (d) =>
  `${d.getUTCFullYear()}:${pad(d.getUTCMonth() + 1)}:${pad(d.getUTCDate())}`;

function toDMS(deg) {
  const a = Math.abs(deg);
  const d = Math.floor(a);
  const mf = (a - d) * 60;
  const m = Math.floor(mf);
  const s = (mf - m) * 60;
  return [[d, 1], [m, 1], [Math.round(s * 10000), 10000]];
}

// ---- entry constructors ---------------------------------------------------
const te = new TextEncoder();
const asciiEntry = (tag, str) => {
  const b = te.encode(str.replace(/\0+$/, '') + '\0');
  return { tag, type: ASCII, count: b.length, bytes: b };
};
const shortEntry = (tag, ...vals) => ({ tag, type: SHORT, count: vals.length, vals });
const longEntry = (tag, ...vals) => ({ tag, type: LONG, count: vals.length, vals });
const rationalEntry = (tag, pairs) => ({ tag, type: RATIONAL, count: pairs.length, pairs });
const byteEntry = (tag, ...vals) => ({ tag, type: BYTE, count: vals.length, bytes: new Uint8Array(vals) });
const undefinedEntry = (tag, arr) => ({ tag, type: UNDEFINED, count: arr.length, bytes: new Uint8Array(arr) });

function valueBytes(e) {
  if (e.bytes) return e.bytes;
  const out = new Uint8Array(e.count * TYPE_SIZE[e.type]);
  const dv = new DataView(out.buffer);
  if (e.type === SHORT) e.vals.forEach((v, i) => dv.setUint16(i * 2, v & 0xffff));
  else if (e.type === LONG) e.vals.forEach((v, i) => dv.setUint32(i * 4, v >>> 0));
  else if (e.type === RATIONAL) e.pairs.forEach(([n, d], i) => {
    dv.setUint32(i * 8, n >>> 0); dv.setUint32(i * 8 + 4, d >>> 0);
  });
  return out;
}

function concatBytes(list) {
  let n = 0;
  for (const b of list) n += b.length;
  const out = new Uint8Array(n);
  let o = 0;
  for (const b of list) { out.set(b, o); o += b.length; }
  return out;
}

// Size of a full IFD block (directory + its overflow data), order-independent.
function ifdBlockSize(entries) {
  let size = 2 + 12 * entries.length + 4;
  for (const e of entries) {
    const len = valueBytes(e).length;
    if (len > 4) size += len + (len & 1);
  }
  return size;
}

// Serialise one IFD at absolute TIFF offset `ifdOffset`.
function buildIFD(entries, ifdOffset, nextOffset = 0) {
  entries = entries.slice().sort((a, b) => a.tag - b.tag);
  const n = entries.length;
  const dirSize = 2 + 12 * n + 4;
  const dir = new Uint8Array(dirSize);
  const dv = new DataView(dir.buffer);
  dv.setUint16(0, n);

  let dataCursor = ifdOffset + dirSize;
  const tail = [];
  entries.forEach((e, i) => {
    const off = 2 + i * 12;
    dv.setUint16(off, e.tag);
    dv.setUint16(off + 2, e.type);
    dv.setUint32(off + 4, e.count);
    const vb = valueBytes(e);
    if (vb.length <= 4) {
      dir.set(vb, off + 8); // left-justified in the 4-byte value field
    } else {
      dv.setUint32(off + 8, dataCursor);
      tail.push(vb);
      dataCursor += vb.length;
      if (vb.length & 1) { tail.push(new Uint8Array(1)); dataCursor += 1; }
    }
  });
  dv.setUint32(2 + 12 * n, nextOffset);
  return concatBytes([dir, ...tail]);
}

export function buildExifSegment({
  width, height, date = new Date(), gps = null,
  headingDeg = null, software = 'PhotoSphere Web',
}) {
  const dt = fmtDateTime(date);

  // ---- Exif Sub-IFD ----
  const exif = [
    undefinedEntry(0x9000, [0x30, 0x32, 0x33, 0x30]), // ExifVersion "0230"
    asciiEntry(0x9003, dt),                            // DateTimeOriginal (local)
    asciiEntry(0x9004, dt),                            // DateTimeDigitized (local)
    shortEntry(0xa001, 1),                             // ColorSpace = sRGB
    longEntry(0xa002, width),                          // PixelXDimension
    longEntry(0xa003, height),                         // PixelYDimension
  ];

  // ---- GPS IFD ----
  const hasGps = gps && Number.isFinite(gps.lat) && Number.isFinite(gps.lon);
  const gpsEntries = [];
  if (hasGps) {
    gpsEntries.push(byteEntry(0x0000, 2, 3, 0, 0));                       // GPSVersionID
    gpsEntries.push(asciiEntry(0x0001, gps.lat >= 0 ? 'N' : 'S'));
    gpsEntries.push(rationalEntry(0x0002, toDMS(gps.lat)));
    gpsEntries.push(asciiEntry(0x0003, gps.lon >= 0 ? 'E' : 'W'));
    gpsEntries.push(rationalEntry(0x0004, toDMS(gps.lon)));
    if (gps.alt != null && Number.isFinite(gps.alt)) {
      gpsEntries.push(byteEntry(0x0005, gps.alt < 0 ? 1 : 0));           // GPSAltitudeRef
      gpsEntries.push(rationalEntry(0x0006, [[Math.round(Math.abs(gps.alt) * 100), 100]]));
    }
    gpsEntries.push(rationalEntry(0x0007, [                              // GPSTimeStamp (UTC)
      [date.getUTCHours(), 1], [date.getUTCMinutes(), 1], [date.getUTCSeconds(), 1],
    ]));
    if (headingDeg != null && Number.isFinite(headingDeg)) {
      const h = (((headingDeg % 360) + 360) % 360);
      gpsEntries.push(asciiEntry(0x0010, 'M'));                          // GPSImgDirectionRef (magnetic)
      gpsEntries.push(rationalEntry(0x0011, [[Math.round(h * 100), 100]]));
    }
    gpsEntries.push(asciiEntry(0x0012, 'WGS-84'));                       // GPSMapDatum
    gpsEntries.push(asciiEntry(0x001d, fmtDateUTC(date)));               // GPSDateStamp (UTC)
  }

  // ---- IFD0 ----
  const ifd0Base = [
    shortEntry(0x0112, 1),        // Orientation = normal
    asciiEntry(0x0131, software), // Software
    asciiEntry(0x0132, dt),       // DateTime (local)
  ];

  // Lay out: [TIFF header 8] [IFD0 block] [Exif block] [GPS block]
  const TIFF_BASE = 8;
  const sizingIfd0 = ifd0Base.concat([longEntry(0x8769, 0)]);
  if (hasGps) sizingIfd0.push(longEntry(0x8825, 0));

  const exifOffset = TIFF_BASE + ifdBlockSize(sizingIfd0);
  const gpsOffset = exifOffset + ifdBlockSize(exif);

  const finalIfd0 = ifd0Base.concat([longEntry(0x8769, exifOffset)]);
  if (hasGps) finalIfd0.push(longEntry(0x8825, gpsOffset));

  const tiffBody = concatBytes([
    buildIFD(finalIfd0, TIFF_BASE, 0),
    buildIFD(exif, exifOffset, 0),
    hasGps ? buildIFD(gpsEntries, gpsOffset, 0) : new Uint8Array(0),
  ]);

  const header = new Uint8Array(8);
  const hv = new DataView(header.buffer);
  hv.setUint16(0, 0x4d4d); // "MM" big-endian
  hv.setUint16(2, 0x002a);
  hv.setUint32(4, 8);      // IFD0 begins at offset 8

  const payload = concatBytes([te.encode('Exif\0\0'), header, tiffBody]);
  const segLen = payload.length + 2;
  if (segLen > 0xffff) throw new Error('EXIF segment too large');

  const seg = new Uint8Array(4 + payload.length);
  seg[0] = 0xff; seg[1] = 0xe1;
  seg[2] = (segLen >> 8) & 0xff;
  seg[3] = segLen & 0xff;
  seg.set(payload, 4);
  return seg;
}
