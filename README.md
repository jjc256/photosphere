# PhotoSphere — a 360° camera web app

Capture a full **360° × 180°** panorama by sweeping your phone around, then
download it as an equirectangular JPEG with **Google Photo Sphere (GPano)**
metadata — so Google Photos, Facebook, VR headsets and photo-sphere viewers
recognise it as a real 360° photo.

It's a static, dependency-free Progressive Web App. Add it to your Home Screen
and it runs full-screen like a native camera app.

---

## Run it

### On a desktop browser (quick test)

```bash
node serve.mjs
```

Open <http://localhost:8080>. With no motion sensors you get **drag mode**:
drag the video to aim a virtual camera and test the stitching with your webcam.

### On your iPhone / iPad (the real thing)

iOS Safari only allows the camera on `https://` or `localhost`, so you need a
trusted (or accepted) certificate:

```bash
node serve.mjs --cert
```

This makes a self-signed cert (requires `openssl`, preinstalled on macOS) and
prints an HTTPS LAN URL like `https://192.168.1.23:8443`. Open that on your
iPhone (same Wi-Fi), tap through the one-time "Not Private" warning, then
**Share → Add to Home Screen**.

Alternatives that avoid the cert warning: run any static host
(`npx serve`, GitHub Pages, Netlify, Cloudflare Pages) or a tunnel
(`cloudflared tunnel --url http://localhost:8080`).

---

## How to shoot a good sphere

1. Tap **Start capture** and allow **Motion & Orientation** + **Camera**.
2. A lattice of **dots** appears around you. Line the centre crosshair up with a
   dot and **hold still** — a ring fills and the frame is grabbed automatically
   (blurred grabs are rejected). The dot turns green; move to the next one.
3. Pivot the phone around *your own body*, keeping it close to you — don't walk;
   parallax is what breaks hand-held stitching. Keep textured things in frame.
4. Work outward through the dots (the counter shows how many are done). You
   don't have to get them all — tap **Done** once you've covered what you want.
5. Tap **Done**. The frames are stitched (a few seconds — "Stitching · …"),
   then you land in the review viewer: drag to look around, pinch to zoom,
   toggle the flat equirectangular view.
6. Tap **Download 360° photo**. The file is saved via the browser's download /
   share sheet (Files, or share to Photos).

**If straight edges don't line up between frames**, open **FOV** and adjust the
*Horizontal field of view* slider to match your lens (iPhone main camera held
in portrait is roughly 50–58°). The **Frame rotation** control is a fallback for
the rare browser that hands the camera feed sideways.

---

## How it works

| Piece | What it does |
|---|---|
| `js/orientation.js` | Turns iOS `deviceorientation` (`alpha/beta/gamma` + screen angle) into a camera→world rotation matrix (three.js `DeviceOrientationControls` math). |
| `js/cv-features.js` + `js/vendor/opencv.js` | On-device **OpenCV WebAssembly** vision backend. It uses SIFT automatically when supplied by a SIFT-enabled OpenCV build; the vendored compatible build uses OpenCV ORB + BF cross-checking. The prior compact JavaScript ORB implementation remains as an offline/runtime fallback. |
| `js/ba.js` | Geometry back-end: SO(3) exp/log, homography RANSAC, **focal length from the homographies** (OpenCV `focalsFromHomography`), a **radial lens model** (`x_d = x_u(1 + k₁r_u²)`, Brown) with both directions, per-pair relative-rotation refinement (L2 Levenberg–Marquardt + IRLS outlier re-filter), **global L2 rotation averaging** anchored to the IMU, and **gain compensation**. |
| `js/stitch.js` | Runs once on **Done**: OpenCV-WASM features → overlap-graph matching → RANSAC-verify → median focal → **calibrate the lens** (solve k₁ and polish the focal by minimising total pairwise reprojection error) → refine each pair's rotation → rotation-average → component bundle adjustment → gain-compensate. Frames without a coherent support group are kept out of the seam blend. |
| `js/pano.js` | WebGL2 engine. `splat()` drives the live capture preview. `compositeStitched()` runs the standard stitcher compositing chain: warp every frame to the sphere → build a consensus mosaic → **content-aware seam labels** (border distance minus a blurred photometric-disagreement term, a cheap stand-in for Kwatra graph-cut seams, resolved with the depth buffer) → **Burt–Adelson multi-band blend** (each source's detail band weighted by its mask blurred narrowly, its base band by the same mask blurred widely) → pole fill → upscale to `panoTex`. Also the interactive sphere view and equirect read-back for export. |
| `js/xmp.js` | Builds the GPano XMP packet and splices metadata segments into the JPEG (EXIF then XMP, after `APP0`). |
| `js/exif.js` | Hand-rolled big-endian **EXIF `APP1`** writer — GPS position, capture time, view direction — the tags Google Maps / Street View needs. |
| `js/app.js` | Camera + permissions, the auto-capture heuristic (angular step ≈ 0.42 × FOV, only while steady), the coverage grid, per-frame stash (full-res `ImageData` + a downscaled luma copy for features), the location watch, the review viewer, and export/download. |
| `sw.js` + `manifest.webmanifest` | Offline app shell and Home-Screen install. |

Stitching is **feature-based, seeded by the IMU**: the gyroscope gives a global
first guess for every frame at once (no incremental drift), then ORB matches +
rotation averaging pull the frames into photometric agreement and recover the
lens — both focal length and radial distortion. Solving distortion matters more
than it sounds: assuming a pinhole not only leaves residual error at the frame
edges (exactly where seams land) but also biases the focal estimate badly — on a
synthetic lens with k₁ = −0.18 the homography focal comes out 39% wrong, and
recovers to 0.3% once k₁ is solved. The device gyro is the fallback at every level — an
unmatched frame, or a capture with too little overlap, still lands roughly
right. Shoot from **one spot** (pivot around yourself); parallax from close
objects is the residual limit that a pure-rotation model can't remove.

`selftest.mjs` (`node selftest.mjs`) renders 12 synthetic views at known poses,
feeds the solver noisy gyro seeds, and checks recovery — at 2.5–5° IMU noise all
frames connect, per-frame rotation error drops from ~1.4° to ~0.35°, focal is
recovered to <0.5%, and per-frame gains are recovered.

### Output

Equirectangular JPEG, `2:1`, up to 4096 × 2048 (device-dependent), quality 0.92.

**GPano XMP** (photo-sphere viewers):

```
GPano:ProjectionType        = equirectangular
GPano:UsePanoramaViewer      = True
GPano:FullPanoWidthPixels / FullPanoHeightPixels = full sphere
GPano:CroppedArea*           = the rows/cols actually captured (dark padding is
                               reported honestly, measured from the accum buffer)
GPano:PoseHeadingDegrees     = compass heading at capture start
GPano:Pose{Pitch,Roll}Degrees = 0 (assumed level)
```

**EXIF** — what puts it on the map for **Google Maps / Street View**:

```
GPS: GPSLatitude/Ref, GPSLongitude/Ref, GPSAltitude/Ref, GPSMapDatum=WGS-84,
     GPSTimeStamp + GPSDateStamp (UTC), GPSImgDirection/Ref (= heading, magnetic)
     GPSVersionID 2.3.0.0
Exif SubIFD: DateTimeOriginal / DateTimeDigitized (local), ExifVersion 0230,
     ColorSpace = sRGB, PixelXDimension / PixelYDimension
IFD0: Orientation = 1, Software, DateTime
```

The **"Tag GPS location"** toggle in the FOV panel (on by default) starts a
`watchPosition`, shows the current fix, and embeds it **only in the file you
download** — nothing is uploaded. Turn it off and the EXIF GPS block is omitted.

Uncovered parts of the sphere are left dark.

### Publishing to Google Maps

1. Shoot with **Tag GPS location** on and get a good fix (shown in the panel).
2. Download the JPEG. It already satisfies Street View's requirements: 2:1
   equirectangular, JPEG, ≥ 7.5 MP (4096 × 2048 = 8.4 MP — the app warns if a
   device's WebGL limit forces it lower), < 75 MB, with GPS + heading.
3. Upload via the **Google Maps app → Contribute → Add → 360 photos**, the
   **Street View** app, or the Street View Publish API. Google reads the
   embedded position and heading to place and orient it.

---

## Browser support

- **iOS 15+ Safari** (WebGL2 + `EXT_color_buffer_float` / `half_float`).
- Any modern Chrome / Firefox / Edge on Android or desktop.
- Needs a secure context (`https:` or `localhost`) for the camera.

## Layout

```
index.html
css/style.css
js/{app,pano,orientation,orb,ba,stitch,xmp,exif}.js
selftest.mjs       # node selftest.mjs — synthetic stitcher check
sw.js
manifest.webmanifest
icons/
serve.mjs          # dev server (+ --cert)
```
