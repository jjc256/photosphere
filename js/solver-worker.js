import { bundleAdjust } from './ba.js';

self.onmessage = ({ data }) => {
  try {
    self.postMessage({ id: data.id, result: bundleAdjust(data.frames, data.rotations, data.pairs, data.options) });
  } catch (error) {
    self.postMessage({ id: data.id, error: error?.message || String(error) });
  }
};
