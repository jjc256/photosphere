// OpenCV WebAssembly feature backend. The vendor build is intentionally loaded
// only when stitching starts: capture remains instant and all image data stays
// on the device. A SIFT-enabled custom build is preferred when present.
import { detectAndDescribe, matchDescriptors } from './orb.js';

let backendPromise;

function loadOpenCV() {
  if (window.__photoSphereCV) return Promise.resolve(window.__photoSphereCV);
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'js/vendor/opencv.js';
    script.async = true;
    script.onload = async () => {
      try {
        const cv = await window.cv;
        window.__photoSphereCV = cv;
        resolve(cv);
      } catch (e) { reject(e); }
    };
    script.onerror = () => reject(new Error('OpenCV WASM could not be loaded'));
    document.head.appendChild(script);
  });
}

function cvData(mat) {
  if (mat.type() === 5) return Float32Array.from(mat.data32F); // CV_32FC1
  return Uint8Array.from(mat.data);
}

function makeMat(cv, f) {
  const m = new cv.Mat(f.rows, f.cols, f.type);
  if (f.desc instanceof Float32Array) m.data32F.set(f.desc);
  else m.data.set(f.desc);
  return m;
}

function openCVBackend(cv) {
  const hasSift = typeof cv.SIFT === 'function' || typeof cv.SIFT?.create === 'function' || typeof cv.SIFT_create === 'function';
  const kind = hasSift ? 'OpenCV SIFT' : 'OpenCV ORB';
  return {
    kind,
    detect(gray, w, h) {
      const src = cv.matFromArray(h, w, cv.CV_8UC1, gray);
      const kps = new cv.KeyPointVector();
      const desc = new cv.Mat();
      const mask = new cv.Mat();
      const detector = hasSift
        // Panorama's OpenCV SIFT backend keeps up to 2,750 features at the
        // default 0.04 contrast threshold.  The previous 900-keypoint cap
        // made low-texture transitions look disconnected before RANSAC.
        ? (typeof cv.SIFT === 'function' ? new cv.SIFT(2750)
          : (cv.SIFT?.create ? cv.SIFT.create(2750) : cv.SIFT_create(2750)))
        : new cv.ORB(1200, 1.2, 8);
      try {
        detector.detectAndCompute(src, mask, kps, desc);
        const points = [];
        for (let i = 0; i < kps.size(); i++) {
          const p = kps.get(i).pt;
          points.push({ x: p.x, y: p.y });
        }
        return { kps: points, desc: cvData(desc), rows: desc.rows, cols: desc.cols, type: desc.type() };
      } finally {
        detector.delete(); src.delete(); kps.delete(); desc.delete(); mask.delete();
      }
    },
    match(a, b) {
      if (!a.rows || !b.rows || a.type !== b.type || a.cols !== b.cols) return [];
      const da = makeMat(cv, a), db = makeMat(cv, b);
      // Panorama uses Lowe's directional best/second-best test for SIFT. It
      // rejects repeated texture before geometry sees it, unlike a fixed
      // distance cutoff or mutual-best-only match.
      if (a.desc instanceof Float32Array && typeof cv.DMatchVectorVector === 'function') {
        const knn = new cv.DMatchVectorVector();
        const matcher = new cv.BFMatcher(cv.NORM_L2, false);
        try {
          matcher.knnMatch(da, db, knn, 2);
          const out = [];
          for (let i = 0; i < knn.size(); i++) {
            const pair = knn.get(i);
            if (pair.size() >= 2) {
              const best = pair.get(0), second = pair.get(1);
              if (best.distance < second.distance * 0.8) out.push([best.queryIdx, best.trainIdx]);
            }
            pair.delete();
          }
          return out;
        } finally {
          matcher.delete(); knn.delete(); da.delete(); db.delete();
        }
      }
      const matches = new cv.DMatchVector();
      const norm = a.desc instanceof Float32Array ? cv.NORM_L2 : cv.NORM_HAMMING;
      const matcher = new cv.BFMatcher(norm, true);
      try {
        matcher.match(da, db, matches);
        const limit = a.desc instanceof Float32Array ? 260 : 68;
        const out = [];
        for (let i = 0; i < matches.size(); i++) {
          const m = matches.get(i);
          if (m.distance <= limit) out.push([m.queryIdx, m.trainIdx]);
        }
        return out;
      } finally {
        matcher.delete(); matches.delete(); da.delete(); db.delete();
      }
    },
  };
}

export function featureBackend() {
  if (backendPromise) return backendPromise;
  if (typeof window === 'undefined') {
    return Promise.resolve({ kind: 'JS ORB', detect: (g, w, h) => detectAndDescribe(g, w, h,
      { fastThresh: 18, maxFeatures: 900, minKeypoints: 120 }), match: (a, b) => matchDescriptors(a.desc, b.desc, 0.82) });
  }
  backendPromise = loadOpenCV().then(openCVBackend).catch(() => ({
    kind: 'JS ORB fallback', detect: (g, w, h) => detectAndDescribe(g, w, h,
      { fastThresh: 18, maxFeatures: 900, minKeypoints: 120 }), match: (a, b) => matchDescriptors(a.desc, b.desc, 0.82),
  }));
  return backendPromise;
}
