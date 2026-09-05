//! Browser-safe numerical kernel for PhotoSphere's generalized camera model.
//! The equations mirror Panorama's Gennery projection and PTLens polynomial.

/// Allocate an f64 buffer in linear memory for the small JS/WASM ABI.  The
/// caller must pair this with `free_f64`; keeping ownership explicit avoids a
/// JavaScript-side allocator dependency in workers and on Vercel's static host.
#[unsafe(no_mangle)]
pub extern "C" fn alloc_f64(len: usize) -> *mut f64 {
    let mut values = Vec::<f64>::with_capacity(len);
    let pointer = values.as_mut_ptr();
    core::mem::forget(values);
    pointer
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_f64(pointer: *mut f64, len: usize) {
    if !pointer.is_null() {
        unsafe { drop(Vec::from_raw_parts(pointer, 0, len)); }
    }
}

/// Solve a symmetric positive-definite normal system using diagonal-preconditioned
/// conjugate gradients. `normal` is row-major n*n and `out` is n elements.
/// Unlike the old browser-only Gaussian-elimination path, this is portable to
/// every WASM host and never materialises an augmented dense elimination matrix.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pcg_solve(
    normal: *const f64,
    rhs: *const f64,
    out: *mut f64,
    n: usize,
    max_iterations: usize,
    tolerance: f64,
) -> i32 {
    if normal.is_null() || rhs.is_null() || out.is_null() || n == 0 { return 0; }
    let a = unsafe { core::slice::from_raw_parts(normal, n * n) };
    let b = unsafe { core::slice::from_raw_parts(rhs, n) };
    let x = unsafe { core::slice::from_raw_parts_mut(out, n) };
    x.fill(0.0);

    let mut r = b.to_vec();
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];
    for i in 0..n {
        let diagonal = a[i * n + i].abs().max(1.0e-12);
        z[i] = r[i] / diagonal;
        p[i] = z[i];
    }
    let mut rz = dot(&r, &z);
    let target = tolerance * tolerance * dot(b, b).max(1.0);
    for _ in 0..max_iterations.max(n) {
        for row in 0..n { ap[row] = dot(&a[row * n..(row + 1) * n], &p); }
        let denom = dot(&p, &ap);
        if !denom.is_finite() || denom.abs() < 1.0e-20 { return 0; }
        let alpha = rz / denom;
        for i in 0..n { x[i] += alpha * p[i]; r[i] -= alpha * ap[i]; }
        if dot(&r, &r) <= target { return 1; }
        for i in 0..n { z[i] = r[i] / a[i * n + i].abs().max(1.0e-12); }
        let next_rz = dot(&r, &z);
        if !next_rz.is_finite() || rz.abs() < 1.0e-30 { return 0; }
        let beta = next_rz / rz;
        for i in 0..n { p[i] = z[i] + beta * p[i]; }
        rz = next_rz;
    }
    1
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[unsafe(no_mangle)]
pub extern "C" fn theta_to_radius(theta: f64, linearity: f64) -> f64 {
    if linearity.abs() < 1.0e-8 { theta } else { (theta * linearity).tan() / linearity }
}

#[unsafe(no_mangle)]
pub extern "C" fn radius_to_theta(radius: f64, linearity: f64) -> f64 {
    if linearity.abs() < 1.0e-8 { radius } else { (radius * linearity).atan() / linearity }
}

#[unsafe(no_mangle)]
pub extern "C" fn ptlens_distort(radius: f64, a: f64, b: f64, c: f64) -> f64 {
    radius + a * radius * radius + b * radius * radius * radius + c * radius * radius * radius * radius
}

#[unsafe(no_mangle)]
pub extern "C" fn ptlens_undistort(radius: f64, a: f64, b: f64, c: f64) -> f64 {
    let mut x = radius;
    for _ in 0..10 {
        let x2 = x * x;
        let value = x + a * x2 + b * x2 * x + c * x2 * x2 - radius;
        let derivative = 1.0 + 2.0 * a * x + 3.0 * b * x2 + 4.0 * c * x2 * x;
        if derivative.abs() < 1.0e-10 { break; }
        x -= value / derivative;
    }
    x
}
