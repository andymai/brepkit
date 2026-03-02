//! Vector and point types for geometric computation.
//!
//! Vectors represent directions and displacements; points represent positions.
//! Both are newtypes over fixed-size `f64` arrays.

use std::ops::{Add, Mul, Neg, Sub};

use crate::MathError;

// ---------------------------------------------------------------------------
// Vec2
// ---------------------------------------------------------------------------

/// A 2D vector representing a direction or displacement.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vec2(pub [f64; 2]);

impl Vec2 {
    /// Create a new 2D vector.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self([x, y])
    }

    /// X component.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.0[0]
    }

    /// Y component.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.0[1]
    }

    /// Dot product of two 2D vectors.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f64 {
        self.0[0].mul_add(rhs.0[0], self.0[1] * rhs.0[1])
    }

    /// Squared length (avoids a sqrt).
    #[must_use]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Return a unit-length vector in the same direction.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::ZeroVector`] if the length is zero.
    pub fn normalize(self) -> Result<Self, MathError> {
        let len = self.length();
        if len == 0.0 {
            return Err(MathError::ZeroVector);
        }
        Ok(Self([self.0[0] / len, self.0[1] / len]))
    }
}

impl Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl Mul<f64> for Vec2 {
    type Output = Self;

    fn mul(self, s: f64) -> Self {
        Self([self.0[0] * s, self.0[1] * s])
    }
}

impl Neg for Vec2 {
    type Output = Self;

    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1]])
    }
}

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

/// A 3D vector representing a direction or displacement.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vec3(pub [f64; 3]);

impl Vec3 {
    /// Create a new 3D vector.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self([x, y, z])
    }

    /// X component.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.0[0]
    }

    /// Y component.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.0[1]
    }

    /// Z component.
    #[must_use]
    pub const fn z(self) -> f64 {
        self.0[2]
    }

    /// Dot product of two 3D vectors.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f64 {
        self.0[0].mul_add(rhs.0[0], self.0[1].mul_add(rhs.0[1], self.0[2] * rhs.0[2]))
    }

    /// Cross product of two 3D vectors.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self([
            self.0[1].mul_add(rhs.0[2], -(self.0[2] * rhs.0[1])),
            self.0[2].mul_add(rhs.0[0], -(self.0[0] * rhs.0[2])),
            self.0[0].mul_add(rhs.0[1], -(self.0[1] * rhs.0[0])),
        ])
    }

    /// Squared length (avoids a sqrt).
    #[must_use]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Return a unit-length vector in the same direction.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::ZeroVector`] if the length is zero.
    pub fn normalize(self) -> Result<Self, MathError> {
        let len = self.length();
        if len == 0.0 {
            return Err(MathError::ZeroVector);
        }
        Ok(Self([self.0[0] / len, self.0[1] / len, self.0[2] / len]))
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, s: f64) -> Self {
        Self([self.0[0] * s, self.0[1] * s, self.0[2] * s])
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2]])
    }
}

// ---------------------------------------------------------------------------
// Point2
// ---------------------------------------------------------------------------

/// A 2D point representing a position in the plane.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Point2(pub [f64; 2]);

impl Point2 {
    /// Create a new 2D point.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self([x, y])
    }

    /// X coordinate.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.0[0]
    }

    /// Y coordinate.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.0[1]
    }
}

/// Translate a point by a vector.
impl Add<Vec2> for Point2 {
    type Output = Self;

    fn add(self, rhs: Vec2) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

/// Displacement from one point to another.
impl Sub for Point2 {
    type Output = Vec2;

    fn sub(self, rhs: Self) -> Vec2 {
        Vec2([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

// ---------------------------------------------------------------------------
// Point3
// ---------------------------------------------------------------------------

/// A 3D point representing a position in space.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Point3(pub [f64; 3]);

impl Point3 {
    /// Create a new 3D point.
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self([x, y, z])
    }

    /// X coordinate.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.0[0]
    }

    /// Y coordinate.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.0[1]
    }

    /// Z coordinate.
    #[must_use]
    pub const fn z(self) -> f64 {
        self.0[2]
    }
}

/// Translate a point by a vector.
impl Add<Vec3> for Point3 {
    type Output = Self;

    fn add(self, rhs: Vec3) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

/// Displacement from one point to another.
impl Sub for Point3 {
    type Output = Vec3;

    fn sub(self, rhs: Self) -> Vec3 {
        Vec3([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!((a.dot(b) - 32.0).abs() < 1e-14);
    }

    #[test]
    fn vec3_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        assert!((z.x()).abs() < 1e-14);
        assert!((z.y()).abs() < 1e-14);
        assert!((z.z() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let n = v.normalize().expect("non-zero");
        assert!((n.length() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn vec3_zero_normalize_fails() {
        let v = Vec3::new(0.0, 0.0, 0.0);
        assert!(v.normalize().is_err());
    }

    #[test]
    fn point3_sub_gives_vec3() {
        let a = Point3::new(3.0, 4.0, 5.0);
        let b = Point3::new(1.0, 1.0, 1.0);
        let v = a - b;
        assert!((v.x() - 2.0).abs() < 1e-14);
        assert!((v.y() - 3.0).abs() < 1e-14);
        assert!((v.z() - 4.0).abs() < 1e-14);
    }

    #[test]
    fn point3_add_vec3() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let v = Vec3::new(1.0, 1.0, 1.0);
        let q = p + v;
        assert!((q.x() - 2.0).abs() < 1e-14);
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_normalize_unit_length(x in -10.0f64..10.0, y in -10.0f64..10.0, z in -10.0f64..10.0) {
            let v = Vec3::new(x, y, z);
            if let Ok(n) = v.normalize() {
                prop_assert!((n.length() - 1.0).abs() < 1e-12, "length = {}", n.length());
            }
        }

        #[test]
        fn prop_cross_anticommutative(
            ax in -10.0f64..10.0, ay in -10.0f64..10.0, az in -10.0f64..10.0,
            bx in -10.0f64..10.0, by in -10.0f64..10.0, bz in -10.0f64..10.0,
        ) {
            let a = Vec3::new(ax, ay, az);
            let b = Vec3::new(bx, by, bz);
            let ab = a.cross(b);
            let ba = b.cross(a);
            // a×b = -(b×a)
            prop_assert!((ab.x() + ba.x()).abs() < 1e-10);
            prop_assert!((ab.y() + ba.y()).abs() < 1e-10);
            prop_assert!((ab.z() + ba.z()).abs() < 1e-10);
        }
    }
}
