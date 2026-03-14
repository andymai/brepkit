//! Matrix types for geometric transforms.
//!
//! [`Mat3`] is a 3x3 matrix and [`Mat4`] is a 4x4 affine transform matrix.

use std::ops::Mul;

use crate::MathError;
use crate::vec::Point3;

// ---------------------------------------------------------------------------
// Mat3
// ---------------------------------------------------------------------------

/// A 3x3 matrix stored in row-major order.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mat3(pub [[f64; 3]; 3]);

impl Mat3 {
    /// The 3x3 identity matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    /// Transpose the matrix.
    #[must_use]
    pub const fn transpose(self) -> Self {
        let m = &self.0;
        Self([
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ])
    }

    /// Compute the determinant of the matrix.
    #[must_use]
    pub fn determinant(self) -> f64 {
        let m = &self.0;
        m[0][0].mul_add(
            m[1][1].mul_add(m[2][2], -(m[1][2] * m[2][1])),
            m[0][1].mul_add(
                m[1][2].mul_add(m[2][0], -(m[1][0] * m[2][2])),
                m[0][2] * m[1][0].mul_add(m[2][1], -(m[1][1] * m[2][0])),
            ),
        )
    }
}

// ---------------------------------------------------------------------------
// Mat4
// ---------------------------------------------------------------------------

/// A 4x4 matrix stored in row-major order, typically used for affine transforms.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mat4(pub [[f64; 4]; 4]);

impl Mat4 {
    /// The 4x4 identity matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Create a translation matrix.
    #[must_use]
    pub const fn translation(tx: f64, ty: f64, tz: f64) -> Self {
        Self([
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Create a uniform or non-uniform scale matrix.
    #[must_use]
    pub const fn scale(sx: f64, sy: f64, sz: f64) -> Self {
        Self([
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Create a rotation matrix around the X axis by `angle` radians.
    #[must_use]
    pub fn rotation_x(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Create a rotation matrix around the Y axis by `angle` radians.
    #[must_use]
    pub fn rotation_y(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self([
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Create a rotation matrix around the Z axis by `angle` radians.
    #[must_use]
    pub fn rotation_z(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self([
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Transform a 3D point by this matrix (assumes w = 1).
    #[must_use]
    pub fn mul_point(self, p: Point3) -> Point3 {
        let m = &self.0;
        Point3::new(
            m[0][0].mul_add(
                p.x(),
                m[0][1].mul_add(p.y(), m[0][2].mul_add(p.z(), m[0][3])),
            ),
            m[1][0].mul_add(
                p.x(),
                m[1][1].mul_add(p.y(), m[1][2].mul_add(p.z(), m[1][3])),
            ),
            m[2][0].mul_add(
                p.x(),
                m[2][1].mul_add(p.y(), m[2][2].mul_add(p.z(), m[2][3])),
            ),
        )
    }

    /// Transpose the matrix.
    #[must_use]
    pub const fn transpose(self) -> Self {
        let m = &self.0;
        Self([
            [m[0][0], m[1][0], m[2][0], m[3][0]],
            [m[0][1], m[1][1], m[2][1], m[3][1]],
            [m[0][2], m[1][2], m[2][2], m[3][2]],
            [m[0][3], m[1][3], m[2][3], m[3][3]],
        ])
    }

    /// Compute the determinant using cofactor expansion along the first row.
    #[must_use]
    #[allow(clippy::similar_names)]
    pub fn determinant(self) -> f64 {
        let m = &self.0;

        let s0 = m[0][0].mul_add(m[1][1], -(m[1][0] * m[0][1]));
        let s1 = m[0][0].mul_add(m[1][2], -(m[1][0] * m[0][2]));
        let s2 = m[0][0].mul_add(m[1][3], -(m[1][0] * m[0][3]));
        let s3 = m[0][1].mul_add(m[1][2], -(m[1][1] * m[0][2]));
        let s4 = m[0][1].mul_add(m[1][3], -(m[1][1] * m[0][3]));
        let s5 = m[0][2].mul_add(m[1][3], -(m[1][2] * m[0][3]));

        let c5 = m[2][2].mul_add(m[3][3], -(m[3][2] * m[2][3]));
        let c4 = m[2][1].mul_add(m[3][3], -(m[3][1] * m[2][3]));
        let c3 = m[2][1].mul_add(m[3][2], -(m[3][1] * m[2][2]));
        let c2 = m[2][0].mul_add(m[3][3], -(m[3][0] * m[2][3]));
        let c1 = m[2][0].mul_add(m[3][2], -(m[3][0] * m[2][2]));
        let c0 = m[2][0].mul_add(m[3][1], -(m[3][0] * m[2][1]));

        s0.mul_add(
            c5,
            (-s1).mul_add(
                c4,
                s2.mul_add(c3, s3.mul_add(c2, (-s4).mul_add(c1, s5 * c0))),
            ),
        )
    }

    /// Compute the inverse of the matrix using the adjugate method.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::SingularMatrix`] if the determinant is approximately zero.
    #[allow(clippy::similar_names)]
    pub fn inverse(self) -> Result<Self, MathError> {
        let m = &self.0;

        // Reuse the 2x2 minor pattern from determinant().
        let s0 = m[0][0].mul_add(m[1][1], -(m[1][0] * m[0][1]));
        let s1 = m[0][0].mul_add(m[1][2], -(m[1][0] * m[0][2]));
        let s2 = m[0][0].mul_add(m[1][3], -(m[1][0] * m[0][3]));
        let s3 = m[0][1].mul_add(m[1][2], -(m[1][1] * m[0][2]));
        let s4 = m[0][1].mul_add(m[1][3], -(m[1][1] * m[0][3]));
        let s5 = m[0][2].mul_add(m[1][3], -(m[1][2] * m[0][3]));

        let c5 = m[2][2].mul_add(m[3][3], -(m[3][2] * m[2][3]));
        let c4 = m[2][1].mul_add(m[3][3], -(m[3][1] * m[2][3]));
        let c3 = m[2][1].mul_add(m[3][2], -(m[3][1] * m[2][2]));
        let c2 = m[2][0].mul_add(m[3][3], -(m[3][0] * m[2][3]));
        let c1 = m[2][0].mul_add(m[3][2], -(m[3][0] * m[2][2]));
        let c0 = m[2][0].mul_add(m[3][1], -(m[3][0] * m[2][1]));

        let det = s0.mul_add(
            c5,
            (-s1).mul_add(
                c4,
                s2.mul_add(c3, s3.mul_add(c2, (-s4).mul_add(c1, s5 * c0))),
            ),
        );

        // Scale-relative singularity check.  The determinant is computed as a
        // sum of products s_i * c_j of 2x2 minors, so its magnitude scales as
        // max_minor^2.  Comparing against that avoids the false-singular
        // problem that a hardcoded threshold (1e-15) causes when matrix entries
        // are very small or very large.
        let max_minor = s0
            .abs()
            .max(s1.abs())
            .max(s2.abs())
            .max(s3.abs())
            .max(s4.abs())
            .max(s5.abs())
            .max(c0.abs())
            .max(c1.abs())
            .max(c2.abs())
            .max(c3.abs())
            .max(c4.abs())
            .max(c5.abs());
        if max_minor == 0.0 {
            return Err(MathError::SingularMatrix);
        }
        if det.abs() < f64::EPSILON * max_minor * max_minor {
            return Err(MathError::SingularMatrix);
        }

        let inv_det = 1.0 / det;

        Ok(Self([
            [
                m[1][1].mul_add(c5, m[1][3].mul_add(c3, -(m[1][2] * c4))) * inv_det,
                (-m[0][1]).mul_add(c5, m[0][2].mul_add(c4, -(m[0][3] * c3))) * inv_det,
                m[3][1].mul_add(s5, m[3][3].mul_add(s3, -(m[3][2] * s4))) * inv_det,
                (-m[2][1]).mul_add(s5, m[2][2].mul_add(s4, -(m[2][3] * s3))) * inv_det,
            ],
            [
                (-m[1][0]).mul_add(c5, m[1][2].mul_add(c2, -(m[1][3] * c1))) * inv_det,
                m[0][0].mul_add(c5, m[0][3].mul_add(c1, -(m[0][2] * c2))) * inv_det,
                (-m[3][0]).mul_add(s5, m[3][2].mul_add(s2, -(m[3][3] * s1))) * inv_det,
                m[2][0].mul_add(s5, m[2][3].mul_add(s1, -(m[2][2] * s2))) * inv_det,
            ],
            [
                m[1][0].mul_add(c4, m[1][3].mul_add(c0, -(m[1][1] * c2))) * inv_det,
                (-m[0][0]).mul_add(c4, m[0][1].mul_add(c2, -(m[0][3] * c0))) * inv_det,
                m[3][0].mul_add(s4, m[3][3].mul_add(s0, -(m[3][1] * s2))) * inv_det,
                (-m[2][0]).mul_add(s4, m[2][1].mul_add(s2, -(m[2][3] * s0))) * inv_det,
            ],
            [
                (-m[1][0]).mul_add(c3, m[1][1].mul_add(c1, -(m[1][2] * c0))) * inv_det,
                m[0][0].mul_add(c3, m[0][2].mul_add(c0, -(m[0][1] * c1))) * inv_det,
                (-m[3][0]).mul_add(s3, m[3][1].mul_add(s1, -(m[3][2] * s0))) * inv_det,
                m[2][0].mul_add(s3, m[2][2].mul_add(s0, -(m[2][1] * s1))) * inv_det,
            ],
        ]))
    }
}

impl Mul for Mat4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let a = &self.0;
        let b = &rhs.0;
        let mut out = [[0.0_f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                out[i][j] = a[i][0].mul_add(
                    b[0][j],
                    a[i][1].mul_add(b[1][j], a[i][2].mul_add(b[2][j], a[i][3] * b[3][j])),
                );
            }
        }
        Self(out)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn approx_eq_mat4(a: &Mat4, b: &Mat4, tol: f64) -> bool {
        for i in 0..4 {
            for j in 0..4 {
                if (a.0[i][j] - b.0[i][j]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn identity_inverse() {
        let inv = Mat4::identity().inverse().expect("invertible");
        assert!(approx_eq_mat4(&inv, &Mat4::identity(), 1e-14));
    }

    #[test]
    fn translation_inverse() {
        let m = Mat4::translation(1.0, 2.0, 3.0);
        let inv = m.inverse().expect("invertible");
        let product = m * inv;
        assert!(approx_eq_mat4(&product, &Mat4::identity(), 1e-12));
    }

    #[test]
    fn rotation_inverse() {
        let m = Mat4::rotation_x(0.7) * Mat4::rotation_y(1.2) * Mat4::rotation_z(0.3);
        let inv = m.inverse().expect("invertible");
        let product = m * inv;
        assert!(approx_eq_mat4(&product, &Mat4::identity(), 1e-12));
    }

    #[test]
    fn scale_inverse() {
        let m = Mat4::scale(2.0, 3.0, 4.0);
        let inv = m.inverse().expect("invertible");
        let product = m * inv;
        assert!(approx_eq_mat4(&product, &Mat4::identity(), 1e-12));
    }

    #[test]
    fn singular_matrix() {
        let m = Mat4([[1.0, 0.0, 0.0, 0.0]; 4]);
        assert!(m.inverse().is_err());
    }

    #[test]
    fn combined_transform_inverse() {
        let m = Mat4::translation(5.0, -3.0, 2.0)
            * Mat4::rotation_z(std::f64::consts::FRAC_PI_4)
            * Mat4::scale(2.0, 0.5, 1.0);
        let inv = m.inverse().expect("invertible");
        let product = m * inv;
        assert!(approx_eq_mat4(&product, &Mat4::identity(), 1e-10));
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_inverse_roundtrip(
            tx in -10.0f64..10.0,
            ty in -10.0f64..10.0,
            tz in -10.0f64..10.0,
            angle in 0.0f64..std::f64::consts::TAU,
        ) {
            let m = Mat4::translation(tx, ty, tz) * Mat4::rotation_z(angle);
            let inv = m.inverse().expect("invertible");
            let product = m * inv;
            prop_assert!(approx_eq_mat4(&product, &Mat4::identity(), 1e-10));
        }

        /// Verify that inverse works for matrices with small and large entry
        /// magnitudes (the old hardcoded 1e-15 threshold would reject these).
        #[test]
        fn prop_inverse_scaled(
            tx in -10.0f64..10.0,
            ty in -10.0f64..10.0,
            tz in -10.0f64..10.0,
            angle in 0.0f64..std::f64::consts::TAU,
            scale_exp in prop::sample::select(&[-8_i32, -6, -4, -2, 2, 4, 6][..]),
        ) {
            let scale = 10.0_f64.powi(scale_exp);
            let m = Mat4::translation(tx * scale, ty * scale, tz * scale)
                * Mat4::rotation_z(angle)
                * Mat4::scale(scale, scale, scale);
            let inv = m.inverse().expect("invertible");
            let product = m * inv;
            // Tolerance scales with condition number; 1e-6 is generous enough
            // for the range of scales we test.
            prop_assert!(approx_eq_mat4(&product, &Mat4::identity(), 1e-6));
        }
    }
}
