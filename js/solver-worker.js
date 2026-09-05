import { bundleAdjust } from './ba.js';
import { generalizedLensSolver } from './solver-wasm.js';

self.onmessage = async ({ data }) => {
  try {
    const solver = await generalizedLensSolver();
    self.postMessage({ id: data.id, result: bundleAdjust(data.frames, data.rotations, data.pairs, {
      ...data.options,
      normalSolver: solver.solveDenseNormal,
    }) });
  } catch (error) {
    self.postMessage({ id: data.id, error: error?.message || String(error) });
  }
};
