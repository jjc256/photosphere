#!/usr/bin/env node
// Tiny zero-dependency static server for PhotoSphere.
//
//   node serve.mjs            → http://localhost:8080  (+ https://…:8443 if a cert exists)
//   node serve.mjs --cert     → generate a self-signed cert (needs `openssl`) then serve
//
// iOS Safari only grants camera access on https:// or localhost. To test on your
// iPhone over Wi-Fi, run `node serve.mjs --cert`, open the https:// LAN URL shown,
// and accept the certificate warning once.

import http from 'node:http';
import https from 'node:https';
import { readFile, stat, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import os from 'node:os';

const ROOT = fileURLToPath(new URL('.', import.meta.url));
const HTTP_PORT = process.env.PORT ? +process.env.PORT : 8080;
const HTTPS_PORT = process.env.HTTPS_PORT ? +process.env.HTTPS_PORT : 8443;
const CERT = join(ROOT, 'cert', 'cert.pem');
const KEY = join(ROOT, 'cert', 'key.pem');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.webmanifest': 'application/manifest+json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.ico': 'image/x-icon',
};

function lanIPs() {
  const out = [];
  for (const list of Object.values(os.networkInterfaces())) {
    for (const ni of list || []) {
      if (ni.family === 'IPv4' && !ni.internal) out.push(ni.address);
    }
  }
  return out;
}

async function handler(req, res) {
  try {
    let path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (path === '/') path = '/index.html';
    const full = normalize(join(ROOT, path));
    if (!full.startsWith(ROOT)) { res.writeHead(403).end('Forbidden'); return; }

    let s;
    try { s = await stat(full); } catch { res.writeHead(404).end('Not found'); return; }
    if (s.isDirectory()) { res.writeHead(404).end('Not found'); return; }

    const body = await readFile(full);
    res.writeHead(200, {
      'Content-Type': MIME[extname(full).toLowerCase()] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
      // getUserMedia + DeviceOrientation need these in some embed contexts
      'Permissions-Policy': 'camera=(self), gyroscope=(self), accelerometer=(self), magnetometer=(self)',
    });
    res.end(body);
  } catch (e) {
    res.writeHead(500).end('Server error: ' + e.message);
  }
}

function makeCert() {
  const ips = lanIPs();
  const san = ['DNS:localhost', 'IP:127.0.0.1', ...ips.map((ip) => `IP:${ip}`)].join(',');
  console.log('Generating self-signed cert with SAN:', san);
  execFileSync('openssl', [
    'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
    '-keyout', KEY, '-out', CERT,
    '-days', '825', '-subj', '/CN=PhotoSphere Dev',
    '-addext', `subjectAltName=${san}`,
  ], { stdio: 'inherit' });
}

async function main() {
  if (process.argv.includes('--cert')) {
    if (!existsSync(join(ROOT, 'cert'))) await mkdir(join(ROOT, 'cert'));
    makeCert();
  }

  http.createServer(handler).listen(HTTP_PORT, () => {
    console.log(`\n  HTTP   http://localhost:${HTTP_PORT}`);
  });

  if (existsSync(CERT) && existsSync(KEY)) {
    const opts = { cert: await readFile(CERT), key: await readFile(KEY) };
    https.createServer(opts, handler).listen(HTTPS_PORT, () => {
      console.log(`  HTTPS  https://localhost:${HTTPS_PORT}`);
      for (const ip of lanIPs()) console.log(`         https://${ip}:${HTTPS_PORT}   ← open this on your iPhone`);
    });
  } else {
    console.log('  (no cert/ — run `node serve.mjs --cert` for an on-device HTTPS URL)');
  }
  console.log('');
}

main();
