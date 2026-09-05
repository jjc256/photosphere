//! Browser-safe numerical kernel for PhotoSphere's generalized camera model.
//! The equations mirror Panorama's Gennery projection and PTLens polynomial.

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
