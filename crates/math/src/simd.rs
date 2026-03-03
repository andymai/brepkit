//! SIMD-friendly batch math operations.
//!
//! When the `simd` feature is enabled, these functions structure computations
//! to encourage auto-vectorization by the compiler. On `wasm32` with
//! `-C target-feature=+simd128`, LLVM will use SIMD instructions.

use crate::mat::Mat4;
use crate::vec::{Point3, Vec3};

/// Batch transform: apply a [`Mat4`] to an array of points.
///
/// Processing points in batches enables auto-vectorization.
pub fn batch_transform_points(mat: &Mat4, points: &[Point3], out: &mut Vec<Point3>) {
    out.clear();
    out.reserve(points.len());
    for p in points {
        out.push(mat.mul_point(*p));
    }
}

/// Batch dot products: compute `dot(a[i], b[i])` for parallel arrays.
///
/// # Panics
///
/// Debug-asserts that `a` and `b` have equal length.
#[must_use]
pub fn batch_dot(a: &[Vec3], b: &[Vec3]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(ai, bi)| ai.dot(*bi)).collect()
}

/// Batch cross products: compute `cross(a[i], b[i])` for parallel arrays.
///
/// # Panics
///
/// Debug-asserts that `a` and `b` have equal length.
#[must_use]
pub fn batch_cross(a: &[Vec3], b: &[Vec3]) -> Vec<Vec3> {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| ai.cross(*bi))
        .collect()
}

/// Batch normalize: normalize an array of vectors.
///
/// Zero-length vectors are left as-is.
#[must_use]
pub fn batch_normalize(vecs: &[Vec3]) -> Vec<Vec3> {
    vecs.iter().map(|v| v.normalize().unwrap_or(*v)).collect()
}

/// Batch squared distances between corresponding point pairs.
///
/// # Panics
///
/// Debug-asserts that `a` and `b` have equal length.
#[must_use]
pub fn batch_distance_sq(a: &[Point3], b: &[Point3]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| {
            let d = *ai - *bi;
            d.length_squared()
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn batch_transform_matches_individual() {
        let mat = Mat4::translation(1.0, 2.0, 3.0) * Mat4::rotation_z(std::f64::consts::FRAC_PI_4);
        let points = vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(3.0, 4.0, 5.0),
        ];

        let mut batch_out = Vec::new();
        batch_transform_points(&mat, &points, &mut batch_out);

        for (i, p) in points.iter().enumerate() {
            let expected = mat.mul_point(*p);
            assert!(
                (batch_out[i].x() - expected.x()).abs() < 1e-14,
                "x mismatch at {i}"
            );
            assert!(
                (batch_out[i].y() - expected.y()).abs() < 1e-14,
                "y mismatch at {i}"
            );
            assert!(
                (batch_out[i].z() - expected.z()).abs() < 1e-14,
                "z mismatch at {i}"
            );
        }
    }

    #[test]
    fn batch_dot_matches_individual() {
        let a = vec![
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(-1.0, 0.0, 1.0),
        ];
        let b = vec![
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(2.0, 3.0, 4.0),
        ];

        let results = batch_dot(&a, &b);

        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            let expected = ai.dot(*bi);
            assert!((results[i] - expected).abs() < 1e-14, "dot mismatch at {i}");
        }
    }

    #[test]
    fn batch_distance_matches_individual() {
        let a = vec![
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(-1.0, -2.0, -3.0),
        ];
        let b = vec![
            Point3::new(4.0, 6.0, 3.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(1.0, 2.0, 3.0),
        ];

        let results = batch_distance_sq(&a, &b);

        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            let d = *ai - *bi;
            let expected = d.length_squared();
            assert!(
                (results[i] - expected).abs() < 1e-14,
                "distance_sq mismatch at {i}"
            );
        }
    }
}
