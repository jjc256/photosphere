let worker;
let sequence = 0;
const pending = new Map();

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL('./solver-worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = ({ data }) => {
      const request = pending.get(data.id);
      if (!request) return;
      pending.delete(data.id);
      data.error ? request.reject(new Error(data.error)) : request.resolve(data.result);
    };
    worker.onerror = (event) => {
      for (const request of pending.values()) request.reject(event.error || new Error(event.message));
      pending.clear(); worker?.terminate(); worker = null;
    };
  }
  return worker;
}

export function globalBundleAdjust(frames, rotations, pairs, options) {
  if (typeof Worker === 'undefined') return Promise.resolve(bundleAdjustFallback(frames, rotations, pairs, options));
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, frames, rotations, pairs, options });
  });
}

// Node replay has no Worker global; import lazily only on that path.
async function bundleAdjustFallback(frames, rotations, pairs, options) {
  const { bundleAdjust } = await import('./ba.js');
  return bundleAdjust(frames, rotations, pairs, options);
}
