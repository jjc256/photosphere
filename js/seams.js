// Graph-cut seams followed by conservative alpha-expansion refinement.
// Preserve coverage while removing old boundaries without increasing jumps.
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

export function seamLabels(warps, w, h, gains, verified, { refine = true, onDiagnostics = () => {} } = {}) {
  const n = w * h;
  const distances = warps.map((pixels) => interior(pixels, w, h));
  const labels = initialSeamLabels(warps, w, h, gains, verified, distances);
  if (!refine || warps.length < 2) return labels;
  const order = warps.map((_, i) => i).sort((a, b) => Number(verified[b]) - Number(verified[a]));
  const colourDistance = (a, b, p) => {
    if (!warps[a][p * 4 + 3] || !warps[b][p * 4 + 3]) return 0;
    let cost = 0;
    for (let c = 0; c < 3; c++) cost += Math.abs(Math.min(255, warps[a][p * 4 + c] * gains[a][c]) - Math.min(255, warps[b][p * 4 + c] * gains[b][c]));
    return cost / 765;
  };
  const boundaryCost = (a, b, p, q) => a === b ? 0 : 0.002 + colourDistance(a, b, p) + colourDistance(a, b, q);
  const energy = () => {
    let total = 0, cuts = 0, mismatch = 0;
    for (let p = 0; p < n; p++) {
      const k = labels[p];
      if (k < 0) continue;
      total -= 0.002 * 256 * distances[k][p] / (w * w);
      for (const q of [Math.floor(p / w) * w + (p % w + 1) % w, p + w]) {
        if (q < n && labels[q] >= 0 && labels[q] !== k) {
          total += boundaryCost(k, labels[q], p, q); cuts++;
          const j = labels[q];
          if (warps[k][p * 4 + 3] && warps[j][p * 4 + 3] && warps[k][q * 4 + 3] && warps[j][q * 4 + 3]) {
            const error = colourDistance(k, j, p) + colourDistance(k, j, q);
            mismatch += error * error;
          }
        }
      }
    }
    return { total, cuts, mismatch };
  };
  let bestEnergy = energy();
  const before = { ...bestEnergy };
  let acceptedMoves = 0;
  // Revisit sources after the initial covering mosaic exists. Alpha expansion
  // can then remove seams introduced earlier instead of only adding new ones.
  for (const k of order) {
    const frame = warps[k], ids = new Int32Array(n).fill(-1);
    const fixed = new Int8Array(n); // 0=empty, 1=old, 2=new
    let count = 0;
    for (let p = 0; p < n; p++) {
      const old = labels[p], has = frame[p * 4 + 3] > 0;
      if (old < 0) { fixed[p] = has ? 2 : 0; continue; }
      if (!has || (verified[old] && !verified[k])) { fixed[p] = 1; continue; }
      ids[p] = count++;
    }
    // Only overlap pixels are unknowns. Eliminating fixed regions keeps
    // memory and max-flow work proportional to overlap, not the whole sphere.
    const graph = new CutGraph(count + 2, count * 16 + 8);
    const delta = new Float64Array(count);
    for (let p = 0; p < n; p++) if (ids[p] >= 0) {
      delta[ids[p]] = 0.002 * 256 * (distances[labels[p]][p] - distances[k][p]) / (w * w);
    }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const p = y * w + x, id = ids[p];
      if (id < 0) continue;
      for (const q of [y * w + (x + w - 1) % w, y * w + (x + 1) % w, y > 0 ? p - w : -1, y + 1 < h ? p + w : -1]) {
        if (q < 0) continue;
        if (ids[q] >= 0) {
          if (p > q) continue;
          const e00 = boundaryCost(labels[p], labels[q], p, q);
          const e01 = boundaryCost(labels[p], k, p, q);
          const e10 = boundaryCost(k, labels[q], p, q);
          // Include the cost of keeping an existing old/old seam. Missing
          // overlap can violate the metric inequality: clamp that proposal
          // and accept it only if the measured full-image energy improves.
          const weight = Math.max(0, (e01 + e10 - e00) / 2);
          graph.edge(id, ids[q], weight, weight);
          delta[id] += e10 - e00 - weight;
          delta[ids[q]] += e01 - e00 - weight;
        } else if (fixed[q] === 1) {
          delta[id] += boundaryCost(k, labels[q], p, q) - boundaryCost(labels[p], labels[q], p, q);
        } else if (fixed[q] === 2) {
          delta[id] -= boundaryCost(labels[p], k, p, q);
        }
      }
    }
    for (let id = 0; id < count; id++) {
      // Source-side = old mosaic; sink-side = new image.
      graph.edge(count, id, Math.max(0, delta[id]));
      graph.edge(id, count + 1, Math.max(0, -delta[id]));
    }
    const side = graph.cut(count, count + 1);
    const previous = labels.slice();
    for (let p = 0; p < n; p++) if (fixed[p] === 2 || (ids[p] >= 0 && side[ids[p]] < 0)) labels[p] = k;
    const nextEnergy = energy();
    // Shorter boundaries must not trade many gentle joins for a few large
    // jumps. Keep the original labels unless neither quality measure gets worse.
    if (nextEnergy.total < bestEnergy.total - 1e-8 && nextEnergy.cuts <= bestEnergy.cuts && nextEnergy.mismatch <= bestEnergy.mismatch + 1e-8) {
      bestEnergy = nextEnergy; acceptedMoves++;
    } else labels.set(previous);
  }
  onDiagnostics({ before, after: bestEnergy, acceptedMoves, sourceFrames: new Set(labels.filter((v) => v >= 0)).size });
  return labels;
}

function initialSeamLabels(warps, w, h, gains, verified, distances) {
  const n = w * h, labels = new Int16Array(n).fill(-1);
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
