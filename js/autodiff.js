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
const pair = (a, b) => {
  if (a instanceof Dual) return [a, lift(b, a.gradient.length)];
  if (b instanceof Dual) return [lift(a, b.gradient.length), b];
  throw new TypeError('autodiff operation needs at least one Dual operand');
};
export const add = (a, b) => { [a, b] = pair(a, b); return binary(a, b, a.value + b.value, 1, 1); };
export const sub = (a, b) => { [a, b] = pair(a, b); return binary(a, b, a.value - b.value, 1, -1); };
export const mul = (a, b) => { [a, b] = pair(a, b); return binary(a, b, a.value * b.value, b.value, a.value); };
export const div = (a, b) => { [a, b] = pair(a, b); return binary(a, b, a.value / b.value, 1 / b.value, -a.value / (b.value * b.value)); };
export const sin = (a) => new Dual(Math.sin(a.value), a.gradient.map((v) => v * Math.cos(a.value)));
export const cos = (a) => new Dual(Math.cos(a.value), a.gradient.map((v) => -v * Math.sin(a.value)));
export const sqrt = (a) => { const v = Math.sqrt(a.value); return new Dual(v, a.gradient.map((x) => v ? x / (2 * v) : 0)); };
export const atan = (a) => new Dual(Math.atan(a.value), a.gradient.map((v) => v / (1 + a.value * a.value)));
export const asin = (a) => new Dual(Math.asin(a.value), a.gradient.map((x) => x / Math.sqrt(Math.max(1e-24, 1 - a.value * a.value))));
export const exp = (a) => { const v = Math.exp(a.value); return new Dual(v, a.gradient.map((x) => x * v)); };
export const acos = (a) => new Dual(Math.acos(a.value), a.gradient.map((x) => x / -Math.sqrt(Math.max(1e-24, 1 - a.value * a.value))));
export const tan = (a) => div(sin(a), cos(a));

// Accept either a scalar or a Dual. These small adapters let camera code use
// the same equations while a pair block is evaluated with derivatives.
export const valueOf = (x) => x instanceof Dual ? x.value : x;
export const asDual = (x, n) => x instanceof Dual ? x : Dual.constant(x, n);
