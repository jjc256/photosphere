let solverPromise;

export function generalizedLensSolver() {
  if (!solverPromise) {
    const url = new URL('./vendor/photosphere_solver.wasm', import.meta.url);
    solverPromise = fetch(url).then(async (response) => {
      if (!response.ok) throw new Error(`Lens solver fetch failed: ${response.status}`);
      let instance;
      try { instance = (await WebAssembly.instantiateStreaming(response, {})).instance; }
      catch { instance = (await WebAssembly.instantiate(await response.arrayBuffer(), {})).instance; }
      const exports = instance.exports;
      // Keep the lens API as direct exports, and add a typed, ownership-safe
      // normal-system entry point for the background bundle-adjustment worker.
      const solveDenseNormal = (normal, rhs) => {
        const n = rhs.length;
        const packed = normal.flat ? normal.flat() : normal;
        if (!n || packed.length !== n * n) return null;
        const np = exports.alloc_f64(packed.length);
        const bp = exports.alloc_f64(n);
        const xp = exports.alloc_f64(n);
        try {
          new Float64Array(exports.memory.buffer, np, packed.length).set(packed);
          new Float64Array(exports.memory.buffer, bp, n).set(rhs);
          const ok = exports.pcg_solve(np, bp, xp, n, Math.max(32, n * 3), 1e-9);
          if (!ok) return null;
          return Array.from(new Float64Array(exports.memory.buffer, xp, n));
        } finally {
          exports.free_f64(np, packed.length);
          exports.free_f64(bp, n);
          exports.free_f64(xp, n);
        }
      };
      return { ...exports, solveDenseNormal };
    });
  }
  return solverPromise;
}
