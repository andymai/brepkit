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
