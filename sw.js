// App-shell service worker. Network-first for everything (so a redeploy always
// ships), cache only as an offline fallback.
const CACHE = 'photosphere-0.5.5';
const ASSETS = [
  './',
  './index.html',
  './css/style.css',
  './js/app.js',
  './js/pano.js',
  './js/orientation.js',
  './js/orb.js',
  './js/ba.js',
  './js/stitch.js',
  './js/xmp.js',
  './js/exif.js',
  './manifest.webmanifest',
  './icons/icon.svg',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const { request } = e;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== location.origin) return;

  // Network-first: always try the network, cache the result, fall back to the
  // cache (or the shell for navigations) only when offline.
  e.respondWith(
    fetch(request)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(request, copy)).catch(() => {});
        return res;
      })
      .catch(() => caches.match(request).then((hit) =>
        hit || (request.mode === 'navigate' ? caches.match('./index.html') : Response.error())))
  );
});
