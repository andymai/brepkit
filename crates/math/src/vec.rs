//! Vector and point types for geometric computation.
//!
//! [`Vector`] represents directions and displacements; [`Position`] represents
//! positions. Both are newtypes over fixed-size `f64` arrays, parameterized by
//! dimension via const generics.
//!
//! Ergonomic type aliases are provided for common dimensions:
//!
//! | Alias    | Underlying type |
//! |----------|-----------------|
//! | [`Vec2`] | `Vector<2>` |
//! | [`Vec3`] | `Vector<3>` |
//! | [`Point2`] | `Position<2>` |
//! | [`Point3`] | `Position<3>` |

use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use crate::MathError;

// ===========================================================================
// Vector<N>
// ===========================================================================

/// An N-dimensional vector representing a direction or displacement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<const N: usize>(pub [f64; N]);

/// A 2D vector.
pub type Vec2 = Vector<2>;

/// A 3D vector.
pub type Vec3 = Vector<3>;

// ---------------------------------------------------------------------------
// Shared methods (all dimensions)
// ---------------------------------------------------------------------------

impl<const N: usize> Vector<N> {
    /// Squared Euclidean length (avoids a sqrt).
    #[must_use]
    pub fn length_squared(self) -> f64 {
        let mut sum = 0.0;
        let mut i = 0;
        while i < N {
            sum = self.0[i].mul_add(self.0[i], sum);
            i += 1;
        }
        sum
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
    /// Returns [`MathError::ZeroVector`] if the vector is zero, has
    /// denormalized length (< [`f64::MIN_POSITIVE`]), or has overflowed
    /// length (non-finite).
    pub fn normalize(self) -> Result<Self, MathError> {
        let len = self.length();
        if !len.is_finite() || len < f64::MIN_POSITIVE {
            return Err(MathError::ZeroVector);
        }
        let inv = 1.0 / len;
        Ok(Self(std::array::from_fn(|i| self.0[i] * inv)))
    }
}

// ---------------------------------------------------------------------------
// 2D-specific methods
// ---------------------------------------------------------------------------

impl Vector<2> {
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
}

// ---------------------------------------------------------------------------
// 3D-specific methods
// ---------------------------------------------------------------------------

impl Vector<3> {
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
}

// ---------------------------------------------------------------------------
// Vector operators (all dimensions)
// ---------------------------------------------------------------------------

impl<const N: usize> Add for Vector<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<const N: usize> AddAssign for Vector<N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.0[i] += rhs.0[i];
        }
    }
}

impl<const N: usize> Sub for Vector<N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl<const N: usize> SubAssign for Vector<N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.0[i] -= rhs.0[i];
        }
    }
}

/// `vector * scalar`
impl<const N: usize> Mul<f64> for Vector<N> {
    type Output = Self;

    fn mul(self, s: f64) -> Self {
        Self(std::array::from_fn(|i| self.0[i] * s))
    }
}

/// `scalar * vector`
impl<const N: usize> Mul<Vector<N>> for f64 {
    type Output = Vector<N>;

    fn mul(self, rhs: Vector<N>) -> Vector<N> {
        Vector(std::array::from_fn(|i| self * rhs.0[i]))
    }
}

impl<const N: usize> Neg for Vector<N> {
    type Output = Self;

    fn neg(self) -> Self {
        Self(std::array::from_fn(|i| -self.0[i]))
    }
}

// ===========================================================================
// Position<N>
// ===========================================================================

/// An N-dimensional point representing a position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position<const N: usize>(pub [f64; N]);

/// A 2D point.
pub type Point2 = Position<2>;

/// A 3D point.
pub type Point3 = Position<3>;

// ---------------------------------------------------------------------------
// 2D-specific methods
// ---------------------------------------------------------------------------

impl Position<2> {
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

// ---------------------------------------------------------------------------
// 3D-specific methods
// ---------------------------------------------------------------------------

impl Position<3> {
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

// ---------------------------------------------------------------------------
// Position operators (all dimensions)
// ---------------------------------------------------------------------------

/// Translate a point by a vector: `point + vector → point`.
impl<const N: usize> Add<Vector<N>> for Position<N> {
    type Output = Self;

    fn add(self, rhs: Vector<N>) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

/// Translate a point by a negative vector: `point - vector → point`.
impl<const N: usize> Sub<Vector<N>> for Position<N> {
    type Output = Self;

    fn sub(self, rhs: Vector<N>) -> Self {
        Self(std::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

/// Displacement from one point to another: `point - point → vector`.
impl<const N: usize> Sub for Position<N> {
    type Output = Vector<N>;

    fn sub(self, rhs: Self) -> Vector<N> {
        Vector(std::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

// ===========================================================================
// Serde support
// ===========================================================================

/// Implements `Serialize` and `Deserialize` for a const-generic newtype
/// wrapping `[f64; N]`. Serde's derive macro cannot handle `[T; N]` for
/// arbitrary `N`, so we provide manual impls that serialize as a tuple.
macro_rules! impl_serde_for_array_newtype {
    ($ty:ident, $name:expr) => {
        #[cfg(feature = "serde")]
        impl<const N: usize> serde::Serialize for $ty<N> {
            fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
                use serde::ser::SerializeTuple;
                let mut tup = ser.serialize_tuple(N)?;
                for &val in &self.0 {
                    tup.serialize_element(&val)?;
                }
                tup.end()
            }
        }

        #[cfg(feature = "serde")]
        impl<'de, const N: usize> serde::Deserialize<'de> for $ty<N> {
            fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
                struct ArrayVisitor<const M: usize>;

                impl<'de, const M: usize> serde::de::Visitor<'de> for ArrayVisitor<M> {
                    type Value = [f64; M];

                    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(f, "an array of {} floats", M)
                    }

                    fn visit_seq<A: serde::de::SeqAccess<'de>>(
                        self,
                        mut seq: A,
                    ) -> Result<Self::Value, A::Error> {
                        let mut arr = [0.0; M];
                        for (i, slot) in arr.iter_mut().enumerate() {
                            *slot = seq
                                .next_element()?
                                .ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                        }
                        Ok(arr)
                    }
                }

                de.deserialize_tuple(N, ArrayVisitor::<N>).map($ty)
            }
        }
    };
}

impl_serde_for_array_newtype!(Vector, "Vector");
impl_serde_for_array_newtype!(Position, "Position");

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
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
    fn vec3_denormal_normalize_fails() {
        // Denormalized-length vector should be rejected (A1 fix)
        let v = Vec3::new(1e-320, 0.0, 0.0);
        assert!(v.normalize().is_err());
    }

    #[test]
    fn vec2_dot() {
        let a = Vec2::new(3.0, 4.0);
        let b = Vec2::new(1.0, 2.0);
        assert!((a.dot(b) - 11.0).abs() < 1e-14);
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

    #[test]
    fn point3_sub_vec3() {
        let p = Point3::new(3.0, 4.0, 5.0);
        let v = Vec3::new(1.0, 1.0, 1.0);
        let q = p - v;
        assert!((q.x() - 2.0).abs() < 1e-14);
        assert!((q.y() - 3.0).abs() < 1e-14);
        assert!((q.z() - 4.0).abs() < 1e-14);
    }

    #[test]
    fn scalar_times_vec3() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let scaled = 2.0 * v;
        assert!((scaled.x() - 2.0).abs() < 1e-14);
        assert!((scaled.y() - 4.0).abs() < 1e-14);
        assert!((scaled.z() - 6.0).abs() < 1e-14);
    }

    #[test]
    fn vec3_add_assign() {
        let mut a = Vec3::new(1.0, 2.0, 3.0);
        a += Vec3::new(4.0, 5.0, 6.0);
        assert!((a.x() - 5.0).abs() < 1e-14);
        assert!((a.y() - 7.0).abs() < 1e-14);
        assert!((a.z() - 9.0).abs() < 1e-14);
    }

    #[test]
    fn vec3_sub_assign() {
        let mut a = Vec3::new(5.0, 7.0, 9.0);
        a -= Vec3::new(1.0, 2.0, 3.0);
        assert!((a.x() - 4.0).abs() < 1e-14);
        assert!((a.y() - 5.0).abs() < 1e-14);
        assert!((a.z() - 6.0).abs() < 1e-14);
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

        #[test]
        fn prop_point_sub_vec_inverse_of_add(
            px in -100.0f64..100.0, py in -100.0f64..100.0, pz in -100.0f64..100.0,
            vx in -100.0f64..100.0, vy in -100.0f64..100.0, vz in -100.0f64..100.0,
        ) {
            let p = Point3::new(px, py, pz);
            let v = Vec3::new(vx, vy, vz);
            // (p + v) - v == p
            let roundtrip = (p + v) - v;
            prop_assert!((roundtrip.x() - p.x()).abs() < 1e-10);
            prop_assert!((roundtrip.y() - p.y()).abs() < 1e-10);
            prop_assert!((roundtrip.z() - p.z()).abs() < 1e-10);
        }
    }
}
