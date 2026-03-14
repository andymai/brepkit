//! Orthonormal reference frame in 3D space.
//!
//! [`Frame3`] bundles an origin with three mutually perpendicular unit axes.
//! It replaces the 14+ hand-rolled "pick a candidate, cross twice" patterns
//! scattered across surfaces, curves, and intersection code.

use crate::MathError;
use crate::vec::{Point3, Vec3};

/// An orthonormal reference frame: origin + three mutually perpendicular unit
/// axes (`x`, `y`, `z`).
///
/// The `z` axis is the *primary* direction (surface normal, curve axis, etc.).
/// `x` and `y` span the plane perpendicular to `z`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame3 {
    /// Frame origin.
    pub origin: Point3,
    /// First axis in the perpendicular plane.
    pub x: Vec3,
    /// Second axis in the perpendicular plane.
    pub y: Vec3,
    /// Primary axis / normal.
    pub z: Vec3,
}

impl Frame3 {
    /// Build an orthonormal frame from an origin and a primary axis (normal).
    ///
    /// `x` and `y` are chosen arbitrarily in the plane perpendicular to `z`.
    /// The input `normal` is normalized internally.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::ZeroVector`] if `normal` is zero-length.
    pub fn from_normal(origin: Point3, normal: Vec3) -> Result<Self, MathError> {
        let z = normal.normalize()?;
        let (x, y) = perpendicular_pair(z)?;
        Ok(Self { origin, x, y, z })
    }

    /// Build an orthonormal frame from an origin, a primary axis, and a
    /// preferred reference direction for `x`.
    ///
    /// `ref_dir` is projected onto the plane perpendicular to `z` to produce
    /// `x`. If `ref_dir` is (nearly) parallel to `z`, the frame falls back to
    /// an arbitrary perpendicular choice.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::ZeroVector`] if `normal` is zero-length.
    pub fn from_normal_and_ref(
        origin: Point3,
        normal: Vec3,
        ref_dir: Vec3,
    ) -> Result<Self, MathError> {
        let z = normal.normalize()?;
        let ref_proj = ref_dir - z * ref_dir.dot(z);
        let x = if let Ok(v) = ref_proj.normalize() {
            v
        } else {
            // ref_dir is parallel to z — fall back to arbitrary choice.
            let (arb_x, _) = perpendicular_pair(z)?;
            arb_x
        };
        let y = z.cross(x);
        Ok(Self { origin, x, y, z })
    }
}

/// Given a unit vector `z`, return two unit vectors `(x, y)` forming an
/// orthonormal basis where `x = z × candidate` and `y = z × x`.
fn perpendicular_pair(z: Vec3) -> Result<(Vec3, Vec3), MathError> {
    let candidate = if z.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let x = z.cross(candidate).normalize()?;
    let y = z.cross(x);
    Ok((x, y))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn frame_from_z_axis() {
        let f = Frame3::from_normal(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 5.0))
            .expect("non-zero");
        assert!((f.z.z() - 1.0).abs() < 1e-14);
        assert!(f.x.dot(f.y).abs() < 1e-14);
        assert!(f.x.dot(f.z).abs() < 1e-14);
        assert!(f.y.dot(f.z).abs() < 1e-14);
    }

    #[test]
    fn frame_from_x_axis() {
        // Tests the branch where x_component > 0.9
        let f = Frame3::from_normal(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0))
            .expect("non-zero");
        assert!((f.z.x() - 1.0).abs() < 1e-14);
        assert!(f.x.dot(f.z).abs() < 1e-14);
    }

    #[test]
    fn frame_with_ref_dir() {
        let f = Frame3::from_normal_and_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .expect("non-zero");
        // x should align with the reference direction
        assert!((f.x.x() - 1.0).abs() < 1e-14);
        assert!(f.x.y().abs() < 1e-14);
    }

    #[test]
    fn frame_with_parallel_ref_falls_back() {
        // ref_dir parallel to normal — should still produce valid frame
        let f = Frame3::from_normal_and_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        )
        .expect("non-zero");
        assert!(f.x.dot(f.z).abs() < 1e-14);
        assert!((f.x.length() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn zero_normal_errors() {
        let r = Frame3::from_normal(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
        assert!(r.is_err());
    }

    proptest! {
        #[test]
        fn prop_frame_orthonormal(
            nx in -10.0f64..10.0, ny in -10.0f64..10.0, nz in -10.0f64..10.0,
        ) {
            let n = Vec3::new(nx, ny, nz);
            if let Ok(f) = Frame3::from_normal(Point3::new(0.0, 0.0, 0.0), n) {
                // All axes are unit length
                prop_assert!((f.x.length() - 1.0).abs() < 1e-12);
                prop_assert!((f.y.length() - 1.0).abs() < 1e-12);
                prop_assert!((f.z.length() - 1.0).abs() < 1e-12);
                // Mutually perpendicular
                prop_assert!(f.x.dot(f.y).abs() < 1e-12);
                prop_assert!(f.x.dot(f.z).abs() < 1e-12);
                prop_assert!(f.y.dot(f.z).abs() < 1e-12);
                // Right-handed: x × y ≈ z
                let cross = f.x.cross(f.y);
                prop_assert!((cross.x() - f.z.x()).abs() < 1e-12);
                prop_assert!((cross.y() - f.z.y()).abs() < 1e-12);
                prop_assert!((cross.z() - f.z.z()).abs() < 1e-12);
            }
        }
    }
}
