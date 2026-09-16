//! Exact rational Hermite-Coons setback patch for a convex orthogonal corner.
//!
//! Two boundaries are the radius-one quarter-circle strip ends. The other two
//! are cubic curves on the planar supports, tangent to the retained sharp edge
//! at their common endpoint. Prescribing cross-derivatives in the four support
//! tangent planes gives G1 joins; a bilinear Coons fill does not impose them.
//! The shared sharp-edge endpoint is deliberately singular (two tangent planes
//! meet there), not an interior fold or a missing boundary.
//!
//! Multiply the rational Hermite formula by q(u)q(v), the two circular-arc
//! denominators. Its numerator is biquintic. Power-to-Bernstein conversion
//! therefore produces the exact rational surface, without fitting or sampling.

use brepkit_math::MathError;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};

const ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);
const H: [[f64; 4]; 4] = [
    [1.0, 0.0, -3.0, 2.0],
    [0.0, 0.0, 3.0, -2.0],
    [0.0, 1.0, -2.0, 1.0],
    [0.0, 0.0, -1.0, 1.0],
];

pub(super) fn planar_controls() -> [[Point3; 4]; 2] {
    let p = Point3::new(0.0, 2.0, 0.0);
    let tangent = Point3::new(0.0, 5.0 / 3.0, 0.0);
    [
        [
            p,
            tangent,
            Point3::new(2.0 / 3.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ],
        [
            p,
            tangent,
            Point3::new(0.0, 1.0, 2.0 / 3.0),
            Point3::new(0.0, 1.0, 1.0),
        ],
    ]
}

pub(super) fn surface() -> Result<NurbsSurface, MathError> {
    let w = std::f64::consts::FRAC_1_SQRT_2;
    let q = [1.0, 2.0 * (w - 1.0), 2.0 * (1.0 - w)];
    let hq = H.map(|basis| multiply(&basis, &q));
    let [c0, d0] = planar_controls().map(|points| curve_power(&points, &[1.0; 4]));
    let c1 = curve_power(
        &[
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
        ],
        &[1.0, w, 1.0],
    );
    let d1 = curve_power(
        &[
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 1.0),
        ],
        &[1.0, w, 1.0],
    );
    // Smooth endpoint interpolation: derivative magnitudes 1 and sqrt(2)
    // match cubic planar and rational quadratic circular boundary derivatives.
    // Vanishing endpoint slopes make all four mixed derivatives compatible.
    let delta = std::f64::consts::SQRT_2 - 1.0;
    let speed = [1.0, 0.0, 3.0 * delta, -2.0 * delta];
    let y_speed = speed.map(|value| Vec3::new(0.0, -value, 0.0));
    let x_speed = speed.map(|value| Vec3::new(value, 0.0, 0.0));
    let z_speed = speed.map(|value| Vec3::new(0.0, 0.0, value));
    let mut numerator = [[ZERO; 6]; 6];
    for (poly, basis) in [
        (multiply_vec(&c0, &q), hq[0]),
        (c1, hq[1]),
        (multiply_vec(&y_speed, &q), hq[2]),
        (multiply_vec(&z_speed, &q), hq[3]),
    ] {
        for i in 0..6 {
            for j in 0..6 {
                numerator[i][j] += poly[i] * basis[j];
            }
        }
    }
    for (basis, poly) in [
        (hq[0], multiply_vec(&d0, &q)),
        (hq[1], d1),
        (hq[2], multiply_vec(&y_speed, &q)),
        (hq[3], multiply_vec(&x_speed, &q)),
    ] {
        for i in 0..6 {
            for j in 0..6 {
                numerator[i][j] += poly[j] * basis[i];
            }
        }
    }
    let root_two = std::f64::consts::SQRT_2;
    let corners = [
        [
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ],
        [
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, -root_two, 0.0),
            Vec3::new(0.0, 0.0, root_two),
        ],
        [
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, -root_two, 0.0),
            ZERO,
            ZERO,
        ],
        [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(root_two, 0.0, 0.0),
            ZERO,
            ZERO,
        ],
    ];
    for a in 0..4 {
        for b in 0..4 {
            for i in 0..6 {
                for j in 0..6 {
                    numerator[i][j] -= corners[a][b] * (hq[a][i] * hq[b][j]);
                }
            }
        }
    }
    let denominator: [f64; 6] = std::array::from_fn(|i| {
        (0..=i.min(2))
            .map(|k| q[k] * choose(i, k) / choose(5, k))
            .sum()
    });
    let mut points = vec![vec![Point3::new(0.0, 0.0, 0.0); 6]; 6];
    let mut weights = vec![vec![0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            let mut homogeneous = ZERO;
            for k in 0..=i {
                for l in 0..=j {
                    homogeneous += numerator[k][l]
                        * (choose(i, k) / choose(5, k) * choose(j, l) / choose(5, l));
                }
            }
            weights[i][j] = denominator[i] * denominator[j];
            let p = homogeneous * weights[i][j].recip();
            points[i][j] = Point3::new(p.x(), p.y(), p.z());
        }
    }
    let knots: Vec<_> = [0.0; 6].into_iter().chain([1.0; 6]).collect();
    NurbsSurface::new(5, 5, knots.clone(), knots, points, weights)
}

fn choose(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |value, i| value * (n - i) as f64 / (i + 1) as f64)
}

fn multiply(a: &[f64], b: &[f64]) -> [f64; 6] {
    let mut out = [0.0; 6];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            if i + j < 6 {
                out[i + j] += x * y;
            }
        }
    }
    out
}

fn multiply_vec(a: &[Vec3], b: &[f64]) -> [Vec3; 6] {
    let mut out = [ZERO; 6];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            if i + j < 6 {
                out[i + j] += x * y;
            }
        }
    }
    out
}

fn curve_power(points: &[Point3], weights: &[f64]) -> [Vec3; 6] {
    let mut out = [ZERO; 6];
    let n = points.len() - 1;
    for (i, p) in points.iter().enumerate() {
        for (k, coefficient) in out.iter_mut().enumerate().take(n + 1).skip(i) {
            let sign = if (k - i) % 2 == 0 { 1.0 } else { -1.0 };
            *coefficient += Vec3::new(p.x(), p.y(), p.z())
                * (sign * choose(n, i) * choose(n - i, k - i) * weights[i]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn setback_patch_has_four_tangent_boundaries_and_no_interior_fold() {
        let s = surface().unwrap();
        let tol = brepkit_math::tolerance::Tolerance::new();
        for i in 1..100 {
            let t = f64::from(i) / 100.0;
            let a = s.evaluate(t, 1.0);
            let b = s.evaluate(1.0, t);
            let na = Vec3::new(a.x() - 1.0, a.y() - 1.0, 0.0)
                .normalize()
                .unwrap();
            let nb = Vec3::new(0.0, b.y() - 1.0, b.z() - 1.0)
                .normalize()
                .unwrap();
            assert!(
                (s.normal(t, 1.0).unwrap() - na).length() < tol.angular,
                "cylinder v=1 t={t}"
            );
            assert!(
                (s.normal(1.0, t).unwrap() - nb).length() < tol.angular,
                "cylinder u=1 t={t}"
            );
            assert!(
                (s.normal(t, 0.0).unwrap() - Vec3::new(0.0, 0.0, -1.0)).length() < tol.angular,
                "plane v=0 t={t}"
            );
            assert!(
                (s.normal(0.0, t).unwrap() - Vec3::new(-1.0, 0.0, 0.0)).length() < tol.angular,
                "plane u=0 t={t}"
            );
            assert!(((a - Point3::new(1.0, 1.0, 1.0)).length() - 1.0).abs() < tol.linear);
            assert!(((b - Point3::new(1.0, 1.0, 1.0)).length() - 1.0).abs() < tol.linear);
            for j in 1..100 {
                let v = f64::from(j) / 100.0;
                let normal = s.normal(t, v).unwrap();
                let p = s.evaluate(t, v);
                assert!(
                    p.x() >= 0.0
                        && p.x() <= 1.0
                        && p.z() >= 0.0
                        && p.z() <= 1.0
                        && p.y() >= 0.0
                        && p.y() <= 2.0
                );
                // The patch removes material from both independently rounded
                // supports; it does not add a bulge outside either fillet.
                assert!(p.y() >= 1.0 - (1.0 - (p.x() - 1.0).powi(2)).sqrt());
                assert!(p.y() >= 1.0 - (1.0 - (p.z() - 1.0).powi(2)).sqrt());
                assert!(
                    normal.dot(Vec3::new(-1.0, -1.0, -1.0)) > 0.0,
                    "fold {t},{v}: {normal:?}"
                );
            }
        }
    }
}
