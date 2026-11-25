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

use crate::madd::Madd;
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

/// The least sum of squares that [`Vector::length`] takes as it comes,
/// 2^-1000: each larger square is normal, and the root keeps its digits.
const SQUARES_LOW: f64 = f64::from_bits(23 << 52);

/// 2^600: a vector whose squares underflow is scaled up by it, which
/// changes no digit.
const SCALE_UP: f64 = f64::from_bits(1623 << 52);

/// 2^-600: a vector whose squares overflow is scaled down by it, which
/// changes no digit of a component large enough to count in its length.
const SCALE_DOWN: f64 = f64::from_bits(423 << 52);

/// The least part of a vector square to an axis, over the vector's length,
/// that [`Vector::square_part`] takes for a direction. A vector along the
/// axis leaves a part of a few units of rounding, far under it.
const SQUARE_PART_LOW: f64 = 1e-12;

/// A power of two by which a vector whose largest component is `size` comes
/// near one in size.
///
/// The scaled size is from 2^-74 to 2^24, so that a product of two of its
/// components neither overflows nor underflows. The scale is exact. It is
/// one for a size that is zero or not finite.
#[must_use]
pub fn unit_scale(size: f64) -> f64 {
    if !(size.is_finite() && size > 0.0) {
        return 1.0;
    }
    #[allow(clippy::cast_possible_truncation)]
    let exponent = size.log2().floor().clamp(-1000.0, 1000.0) as i32;
    2.0_f64.powi(-exponent)
}

impl<const N: usize> Vector<N> {
    /// Squared Euclidean length (avoids a sqrt).
    #[must_use]
    pub fn length_squared(self) -> f64 {
        let mut sum = 0.0;
        let mut i = 0;
        while i < N {
            sum = self.0[i].madd(self.0[i], sum);
            i += 1;
        }
        sum
    }

    /// Euclidean length, at every finite size: a vector whose squares
    /// overflow or underflow is measured at a power of two of its size. It
    /// is infinite only for a length past the largest float.
    #[must_use]
    pub fn length(self) -> f64 {
        let sum = self.length_squared();
        if sum.is_finite() && sum >= SQUARES_LOW {
            return sum.sqrt();
        }
        self.rescaled()
            .map_or_else(|| sum.sqrt(), |(v, back)| v.length_squared().sqrt() * back)
    }

    /// The vector times a power of two that brings its largest component
    /// near one, and the power of two that undoes it; `None` for a zero
    /// vector or a component that is not finite.
    fn rescaled(self) -> Option<(Self, f64)> {
        if !self.0.iter().all(|c| c.is_finite()) {
            return None;
        }
        let most = self.0.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
        if most <= 0.0 {
            return None;
        }
        let (by, back) = if most > 1.0 {
            (SCALE_DOWN, SCALE_UP)
        } else {
            (SCALE_UP, SCALE_DOWN)
        };
        Some((Self(std::array::from_fn(|i| self.0[i] * by)), back))
    }

    /// The largest of the absolute values of the components; not a number
    /// if a component is not.
    #[must_use]
    pub fn largest_component(self) -> f64 {
        self.0.iter().fold(0.0_f64, |m, c| {
            if m.is_nan() || c.is_nan() {
                f64::NAN
            } else {
                m.max(c.abs())
            }
        })
    }

    /// The vector scaled by [`unit_scale`] of its largest component: the
    /// same direction, with no product of two components out of range.
    /// Two vectors so scaled give the sign of their dot or cross product,
    /// and the angle between them, as they are.
    #[must_use]
    pub fn at_unit_size(self) -> Self {
        let by = unit_scale(self.largest_component());
        Self(std::array::from_fn(|i| self.0[i] * by))
    }

    /// Return a unit-length vector in the same direction. A vector of any
    /// finite, non-zero size has one: one whose squares overflow or
    /// underflow is scaled by a power of two first.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::ZeroVector`] if the vector is zero or a
    /// component is not finite.
    pub fn normalize(self) -> Result<Self, MathError> {
        let sum = self.length_squared();
        let v = if sum.is_finite() && sum >= SQUARES_LOW {
            self
        } else {
            self.rescaled().ok_or(MathError::ZeroVector)?.0
        };
        let len = v.length_squared().sqrt();
        if !len.is_finite() || len < f64::MIN_POSITIVE {
            return Err(MathError::ZeroVector);
        }
        let inv = 1.0 / len;
        Ok(Self(std::array::from_fn(|i| v.0[i] * inv)))
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
        self.0[0].madd(rhs.0[0], self.0[1] * rhs.0[1])
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
        self.0[0].madd(rhs.0[0], self.0[1].madd(rhs.0[1], self.0[2] * rhs.0[2]))
    }

    /// Cross product of two 3D vectors.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self([
            self.0[1].madd(rhs.0[2], -(self.0[2] * rhs.0[1])),
            self.0[2].madd(rhs.0[0], -(self.0[0] * rhs.0[2])),
            self.0[0].madd(rhs.0[1], -(self.0[1] * rhs.0[0])),
        ])
    }

    /// The direction of the part of the vector square to the unit vector
    /// `axis`.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::ZeroVector`] if the vector is zero or not
    /// finite, or lies so nearly along `axis` that its square part, under
    /// 1e-12 of its length, is rounding. A test of the part's length alone
    /// takes the rounding of a vector along the axis for a direction. The
    /// vector is taken at a unit size first, so that neither the projection
    /// nor the test overflows or loses digits at either end of the range.
    pub fn square_part(self, axis: Self) -> Result<Self, MathError> {
        let v = self.at_unit_size();
        let part = v - axis * v.dot(axis);
        if part.length() >= SQUARE_PART_LOW * v.length() {
            part.normalize()
        } else {
            Err(MathError::ZeroVector)
        }
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

    /// The square part of a vector at either end of the float range: a
    /// reference of 1.7e308 square to z, and one of the least subnormal
    /// against (1, 1, 1). Each gives a unit direction square to the axis to
    /// a few units of rounding. A vector along the axis is refused at any
    /// size, and so is one that is not finite.
    #[test]
    fn a_square_part_holds_at_the_ends_of_the_float_range() {
        let ulps = 8.0 * f64::EPSILON;
        let z = Vec3::new(0.0, 0.0, 1.0);
        let diagonal = Vec3::new(1.0, 1.0, 1.0).normalize().unwrap();
        let least = f64::from_bits(1);
        for (v, axis, want) in [
            (
                Vec3::new(1.7e308, 1.7e308, 0.0),
                z,
                Vec3::new(1.0, 1.0, 0.0).normalize().unwrap(),
            ),
            (
                Vec3::new(least, 0.0, 0.0),
                diagonal,
                Vec3::new(2.0, -1.0, -1.0).normalize().unwrap(),
            ),
            (
                Vec3::new(f64::MAX, -f64::MAX, f64::MAX),
                diagonal,
                Vec3::new(1.0, -2.0, 1.0).normalize().unwrap(),
            ),
        ] {
            let part = v.square_part(axis).unwrap();
            assert!(part.dot(axis).abs() <= ulps, "{v:?}: {part:?}");
            assert!((part.length() - 1.0).abs() <= ulps, "{v:?}: {part:?}");
            assert!((part - want).length() <= ulps, "{v:?}: {part:?}");
        }
        for s in [1.0, 1e-300, 1e300, least, f64::MAX] {
            assert!(
                Vec3::new(s, s, s).square_part(diagonal).is_err(),
                "size {s}"
            );
        }
        for bad in [
            Vec3::new(f64::NAN, 1.0, 0.0),
            Vec3::new(f64::INFINITY, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        ] {
            assert!(bad.square_part(z).is_err(), "{bad:?}");
        }
    }

    /// A vector at a unit size at the ends of the float range: components
    /// of `f64::MAX`, of the least subnormal, the two in one vector, and
    /// zero. Each scaled vector is finite, its largest component is from
    /// 2^-74 to 2^24, and each component keeps its sign. A vector with a
    /// component that is not a number has no largest component.
    #[test]
    fn the_unit_size_holds_at_the_ends_of_the_float_range() {
        let least = f64::from_bits(1);
        for v in [
            Vec3::new(f64::MAX, f64::MAX, -f64::MAX),
            Vec3::new(least, 0.0, -least),
            Vec3::new(f64::MAX, least, 0.0),
            Vec3::new(-least, f64::MAX, 1.0),
        ] {
            let u = v.at_unit_size();
            let most = u.largest_component();
            assert!(u.0.iter().all(|c| c.is_finite()), "{v:?}: {u:?}");
            assert!(
                (2.0_f64.powi(-74)..=2.0_f64.powi(24)).contains(&most),
                "{v:?}: {u:?}"
            );
            for (a, b) in v.0.iter().zip(u.0) {
                assert!(
                    b.to_bits() == 0 || a.signum().to_bits() == b.signum().to_bits(),
                    "{v:?}: {u:?}"
                );
            }
        }
        let zero = Vec3::new(0.0, 0.0, 0.0);
        assert_eq!(zero.at_unit_size(), zero);
        for size in [0.0, f64::NAN, f64::INFINITY, -1.0] {
            assert_eq!(unit_scale(size).to_bits(), 1.0_f64.to_bits(), "{size}");
        }
        assert_eq!(unit_scale(least).to_bits(), 2.0_f64.powi(1000).to_bits());
        assert_eq!(
            unit_scale(f64::MAX).to_bits(),
            2.0_f64.powi(-1000).to_bits()
        );
        assert!(Vec3::new(f64::NAN, 1.0, 0.0).largest_component().is_nan());
        assert!(Vec3::new(1.0, f64::NAN, 0.0).at_unit_size().y().is_nan());
    }

    /// A vector's length and direction hold at every finite size: squared,
    /// a component of 1e200 overflows and one of 1e-200 underflows.
    #[test]
    fn a_vector_of_any_finite_size_has_a_length_and_a_direction() {
        let sizes = [
            1e200,
            1e-200,
            1e-310,
            f64::MIN_POSITIVE,
            1.7e308,
            1e155,
            1e-160,
        ];
        for s in sizes {
            for (v, length) in [
                (Vec3::new(s, 0.0, 0.0), s),
                (Vec3::new(0.0, -s, 0.0), s),
                (Vec3::new(0.0, 0.0, s), s),
            ] {
                assert_eq!(v.length().to_bits(), length.to_bits(), "{v:?}");
                let unit = v.normalize().unwrap();
                assert!((unit.length() - 1.0).abs() < 1e-15, "{v:?} gives {unit:?}");
                assert!(
                    (unit.dot(v) - v.length()).abs() <= 1e-15 * v.length(),
                    "{v:?}"
                );
            }
            let tilt = Vec3::new(3.0 * s / 4.0, 0.0, s);
            if (3.0 * s / 4.0).is_normal() {
                let unit = tilt.normalize().unwrap();
                assert!(
                    (unit.x() - 0.6).abs() < 1e-15 && (unit.z() - 0.8).abs() < 1e-15,
                    "{s}: {unit:?}"
                );
                assert!(
                    (tilt.length() / (1.25 * s) - 1.0).abs() < 1e-15,
                    "{s}: {}",
                    tilt.length()
                );
            }
            let flat = Vec2::new(0.0, s);
            assert_eq!(flat.length().to_bits(), s.to_bits());
            assert!((flat.normalize().unwrap().y() - 1.0).abs() < 1e-15);
        }
        let past = Vec3::new(1.7e308, 1.7e308, 0.0);
        assert!(past.length().is_infinite());
        let unit = past.normalize().unwrap();
        assert!(
            (unit.x() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15,
            "{unit:?}"
        );
        for bad in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(f64::NAN, 1.0, 0.0),
            Vec3::new(f64::INFINITY, 1.0, 0.0),
        ] {
            assert!(bad.normalize().is_err(), "{bad:?}");
        }
    }

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

    /// A vector with a subnormal length still has a direction, and no
    /// component of it is infinite.
    #[test]
    fn vec3_denormal_normalize_gives_a_unit_vector() {
        let v = Vec3::new(1e-320, 0.0, 0.0);
        let unit = v.normalize().unwrap();
        assert!(
            (unit.x() - 1.0).abs() < 1e-15 && unit.y().abs() < 1e-15,
            "{unit:?}"
        );
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
