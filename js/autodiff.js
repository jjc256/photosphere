// Forward-mode dual scalar used for Panorama-style pair-local Jacobian blocks.
export class Dual {
  constructor(value, gradient) { this.value = value; this.gradient = gradient; }
  static constant(value, n) { return new Dual(value, new Float64Array(n)); }
  static variable(value, n, i) { const g = new Float64Array(n); g[i] = 1; return new Dual(value, g); }
}

const lift = (x, n) => x instanceof Dual ? x : Dual.constant(x, n);
const binary = (a, b, value, da, db) => {
  const n = a.gradient.length; b = lift(b, n); const g = new Float64Array(n);
  for (let i = 0; i < n; i++) g[i] = da * a.gradient[i] + db * b.gradient[i];
  return new Dual(value, g);
};
export const add = (a, b) => { b = lift(b, a.gradient.length); return binary(a, b, a.value + b.value, 1, 1); };
export const sub = (a, b) => { b = lift(b, a.gradient.length); return binary(a, b, a.value - b.value, 1, -1); };
export const mul = (a, b) => { b = lift(b, a.gradient.length); return binary(a, b, a.value * b.value, b.value, a.value); };
export const div = (a, b) => { b = lift(b, a.gradient.length); return binary(a, b, a.value / b.value, 1 / b.value, -a.value / (b.value * b.value)); };
export const sin = (a) => new Dual(Math.sin(a.value), a.gradient.map((v) => v * Math.cos(a.value)));
export const cos = (a) => new Dual(Math.cos(a.value), a.gradient.map((v) => -v * Math.sin(a.value)));
export const sqrt = (a) => { const v = Math.sqrt(a.value); return new Dual(v, a.gradient.map((x) => v ? x / (2 * v) : 0)); };
export const atan = (a) => new Dual(Math.atan(a.value), a.gradient.map((v) => v / (1 + a.value * a.value)));
export const exp = (a) => { const v = Math.exp(a.value); return new Dual(v, a.gradient.map((x) => x * v)); };
export const acos = (a) => new Dual(Math.acos(a.value), a.gradient.map((x) => x / -Math.sqrt(Math.max(1e-24, 1 - a.value * a.value))));
export const tan = (a) => div(sin(a), cos(a));
