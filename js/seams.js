// Sequential graph-cut seams: minimize disagreement along the actual boundary,
// rather than selecting the best-looking source independently at every pixel.
class CutGraph {
  constructor(n, capacity) {
    this.n = n; this.head = new Int32Array(n).fill(-1);
    this.to = new Int32Array(capacity); this.next = new Int32Array(capacity);
    this.cap = new Float64Array(capacity); this.count = 0;
  }
  edge(a, b, forward, reverse = 0) {
    for (const [u, v, c] of [[a, b, forward], [b, a, reverse]]) {
      const e = this.count++; this.to[e] = v; this.cap[e] = c;
      this.next[e] = this.head[u]; this.head[u] = e;
    }
  }
  cut(source, sink) {
    const level = new Int32Array(this.n), queue = new Int32Array(this.n), ptr = new Int32Array(this.n);
    const path = new Int32Array(this.n), nodes = new Int32Array(this.n), flow = new Float64Array(this.n);
    const send = (source, amount) => {
      let depth = 0; nodes[0] = source; flow[0] = amount;
      while (depth >= 0) {
        const u = nodes[depth];
        if (u === sink) {
          const sent = flow[depth];
          for (let i = 0; i < depth; i++) { this.cap[path[i]] -= sent; this.cap[path[i] ^ 1] += sent; }
          return sent;
        }
        let e = ptr[u];
        while (e !== -1 && (this.cap[e] < 1e-8 || level[this.to[e]] !== level[u] + 1)) e = this.next[e];
        ptr[u] = e;
        if (e === -1) {
          level[u] = -1; depth--;
          if (depth >= 0) ptr[nodes[depth]] = this.next[path[depth]];
        } else {
          path[depth] = e; nodes[depth + 1] = this.to[e];
          flow[depth + 1] = Math.min(flow[depth], this.cap[e]); depth++;
        }
      }
      return 0;
    };
    for (;;) {
      level.fill(-1); level[source] = 0;
      let start = 0, end = 1; queue[0] = source;
      while (start < end) {
        const u = queue[start++];
        for (let e = this.head[u]; e !== -1; e = this.next[e]) if (this.cap[e] > 1e-8 && level[this.to[e]] === -1) {
          level[this.to[e]] = level[u] + 1; queue[end++] = this.to[e];
        }
      }
      if (level[sink] < 0) return level; // source-side nodes have nonnegative levels
      ptr.set(this.head);
      while (send(source, 1e9) > 1e-8) { /* augment the level graph */ }
    }
  }
}

// Distance to a source boundary, respecting the longitude wrap.
function interior(rgba, w, h) {
  const d = new Float32Array(w * h), q = new Int32Array(w * h);
  let start = 0, end = 0;
  for (let p = 0; p < d.length; p++) {
    d[p] = rgba[4 * p + 3] ? w : 0;
    if (!d[p]) q[end++] = p;
  }
  while (start < end) {
    const p = q[start++], x = p % w, y = Math.floor(p / w);
    for (const v of [p - x + (x + w - 1) % w, p - x + (x + 1) % w, y > 0 ? p - w : -1, y + 1 < h ? p + w : -1]) {
      if (v >= 0 && d[v] > d[p] + 1) { d[v] = d[p] + 1; q[end++] = v; }
    }
  }
  return d;
}

export function seamLabels(warps, w, h, gains, verified) {
  const n = w * h, labels = new Int16Array(n).fill(-1);
  const distances = warps.map((pixels) => interior(pixels, w, h));
  const order = warps.map((_, i) => i).sort((a, b) => Number(verified[b]) - Number(verified[a]));
  for (const k of order) {
    const frame = warps[k], ids = new Int32Array(n).fill(-1);
    const fixed = new Int8Array(n); // 0=empty, 1=old, 2=new
    const difference = new Float32Array(n);
    let count = 0;
    for (let p = 0; p < n; p++) {
      const old = labels[p], has = frame[p * 4 + 3] > 0;
      if (old < 0) { fixed[p] = has ? 2 : 0; continue; }
      if (!has || (verified[old] && !verified[k])) { fixed[p] = 1; continue; }
      ids[p] = count++;
      for (let c = 0; c < 3; c++) difference[p] += Math.abs(Math.min(255, frame[p * 4 + c] * gains[k][c]) - Math.min(255, warps[old][p * 4 + c] * gains[old][c])) / 765;
    }
    // Only overlap pixels are unknowns. Eliminating fixed regions keeps
    // memory and max-flow work proportional to overlap, not the whole sphere.
    const graph = new CutGraph(count + 2, count * 16 + 8);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const p = y * w + x, id = ids[p];
      if (id < 0) continue;
      const delta = 0.002 * 256 * (distances[k][p] - distances[labels[p]][p]) / (w * w);
      let oldCost = Math.max(0, delta), newCost = Math.max(0, -delta);
      for (const q of [y * w + (x + w - 1) % w, y * w + (x + 1) % w, y > 0 ? p - w : -1, y + 1 < h ? p + w : -1]) {
        if (q < 0) continue;
        const cost = 0.001 + difference[p] + difference[q];
        if (ids[q] >= 0) { if (p < q) graph.edge(id, ids[q], cost, cost); }
        else if (fixed[q] === 1) newCost += cost;
        else if (fixed[q] === 2) oldCost += cost;
      }
      // Source-side = old mosaic; sink-side = new image.
      graph.edge(count, id, newCost); graph.edge(id, count + 1, oldCost);
    }
    const side = graph.cut(count, count + 1);
    for (let p = 0; p < n; p++) if (fixed[p] === 2 || (ids[p] >= 0 && side[ids[p]] < 0)) labels[p] = k;
  }
  return labels;
}
