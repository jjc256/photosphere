let solverPromise;

export function generalizedLensSolver() {
  if (!solverPromise) {
    const url = new URL('./vendor/photosphere_solver.wasm', import.meta.url);
    solverPromise = fetch(url).then(async (response) => {
      if (!response.ok) throw new Error(`Lens solver fetch failed: ${response.status}`);
      try { return (await WebAssembly.instantiateStreaming(response, {})).instance.exports; }
      catch { return (await WebAssembly.instantiate(await response.arrayBuffer(), {})).instance.exports; }
    });
  }
  return solverPromise;
}
