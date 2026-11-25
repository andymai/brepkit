//! Axis-aligned bounding boxes for spatial queries.

use crate::madd::Madd;
use crate::vec::{Point2, Point3, Vec2, Vec3};

/// The lower of `a` and `b`, or not a number if either is not.
const fn lower(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// The higher of `a` and `b`, or not a number if either is not.
const fn upper(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// The lowest and highest of each coordinate of `points`, each not a number
/// where a point's is not, and every one not a number for no point.
fn bounds<const N: usize>(points: impl IntoIterator<Item = [f64; N]>) -> ([f64; N], [f64; N]) {
    let mut iter = points.into_iter();
    let Some(first) = iter.next() else {
        return ([f64::NAN; N], [f64::NAN; N]);
    };
    let (mut min, mut max) = (first, first);
    for p in iter {
        for i in 0..N {
            min[i] = lower(min[i], p[i]);
            max[i] = upper(max[i], p[i]);
        }
    }
    (min, max)
}

/// A 2D axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Aabb2 {
    /// Minimum corner.
    pub min: Point2,
    /// Maximum corner.
    pub max: Point2,
}

impl Aabb2 {
    /// The box of `points`, for points known to be finite and at least one.
    ///
    /// The box hides no point. A coordinate that is not a number, in any
    /// point, makes that coordinate of both corners not a number, and no
    /// point at all makes every coordinate not a number: no test of such a
    /// box passes, and [`Self::is_finite`] is false for it. Use
    /// [`Self::try_from_points`] where other points can arrive.
    #[must_use]
    pub fn from_points(points: impl IntoIterator<Item = Point2>) -> Self {
        let (min, max) = bounds(points.into_iter().map(|p| p.0));
        Self {
            min: Point2::new(min[0], min[1]),
            max: Point2::new(max[0], max[1]),
        }
    }

    /// The box of `points`, or `None` if there is no point or a coordinate
    /// of one is not finite.
    #[must_use]
    pub fn try_from_points(points: impl IntoIterator<Item = Point2>) -> Option<Self> {
        Some(Self::from_points(points)).filter(|b| b.is_finite())
    }

    /// Whether each coordinate of both corners is finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        [self.min.x(), self.min.y(), self.max.x(), self.max.y()]
            .iter()
            .all(|c| c.is_finite())
    }

    /// Whether this box intersects another.
    #[inline]
    #[must_use]
    pub fn intersects(self, other: Self) -> bool {
        self.min.x() <= other.max.x()
            && self.max.x() >= other.min.x()
            && self.min.y() <= other.max.y()
            && self.max.y() >= other.min.y()
    }

    /// Whether this box contains a point.
    #[must_use]
    pub fn contains_point(self, p: Point2) -> bool {
        p.x() >= self.min.x()
            && p.x() <= self.max.x()
            && p.y() >= self.min.y()
            && p.y() <= self.max.y()
    }

    /// Compute the union of two bounding boxes.
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            min: Point2::new(
                lower(self.min.x(), other.min.x()),
                lower(self.min.y(), other.min.y()),
            ),
            max: Point2::new(
                upper(self.max.x(), other.max.x()),
                upper(self.max.y(), other.max.y()),
            ),
        }
    }

    /// Return a new box expanded by `margin` on each side.
    #[inline]
    #[must_use]
    pub fn expanded(self, margin: f64) -> Self {
        Self {
            min: self.min + Vec2::new(-margin, -margin),
            max: self.max + Vec2::new(margin, margin),
        }
    }
}

/// A 3D axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Aabb3 {
    /// Minimum corner.
    pub min: Point3,
    /// Maximum corner.
    pub max: Point3,
}

impl Aabb3 {
    /// The box that holds all of space: a box that must hold a point no
    /// coordinate of which is known.
    pub const WHOLE: Self = Self {
        min: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        max: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
    };

    /// The box of `points`, for points known to be finite and at least one.
    ///
    /// The box hides no point. A coordinate that is not a number, in any
    /// point, makes that coordinate of both corners not a number, and no
    /// point at all makes every coordinate not a number: no test of such a
    /// box passes, and [`Self::is_finite`] is false for it. Use
    /// [`Self::try_from_points`] where other points can arrive.
    #[must_use]
    pub fn from_points(points: impl IntoIterator<Item = Point3>) -> Self {
        let (min, max) = bounds(points.into_iter().map(|p| p.0));
        Self {
            min: Point3::new(min[0], min[1], min[2]),
            max: Point3::new(max[0], max[1], max[2]),
        }
    }

    /// The box of `points`, or `None` if there is no point or a coordinate
    /// of one is not finite.
    #[must_use]
    pub fn try_from_points(points: impl IntoIterator<Item = Point3>) -> Option<Self> {
        Some(Self::from_points(points)).filter(|b| b.is_finite())
    }

    /// Whether each coordinate of both corners is finite.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.min.0.iter().chain(&self.max.0).all(|c| c.is_finite())
    }

    /// Whether no coordinate of either corner is not a number. A box with
    /// an infinite side is a number; one that hides where a point lies is
    /// not.
    #[must_use]
    pub fn is_number(self) -> bool {
        !self.min.0.iter().chain(&self.max.0).any(|c| c.is_nan())
    }

    /// Whether this box intersects another.
    #[inline]
    #[must_use]
    pub fn intersects(self, other: Self) -> bool {
        self.min.x() <= other.max.x()
            && self.max.x() >= other.min.x()
            && self.min.y() <= other.max.y()
            && self.max.y() >= other.min.y()
            && self.min.z() <= other.max.z()
            && self.max.z() >= other.min.z()
    }

    /// Whether this box contains a point.
    #[inline]
    #[must_use]
    pub fn contains_point(self, p: Point3) -> bool {
        p.x() >= self.min.x()
            && p.x() <= self.max.x()
            && p.y() >= self.min.y()
            && p.y() <= self.max.y()
            && p.z() >= self.min.z()
            && p.z() <= self.max.z()
    }

    /// Compute the union of two bounding boxes.
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            min: Point3::new(
                lower(self.min.x(), other.min.x()),
                lower(self.min.y(), other.min.y()),
                lower(self.min.z(), other.min.z()),
            ),
            max: Point3::new(
                upper(self.max.x(), other.max.x()),
                upper(self.max.y(), other.max.y()),
                upper(self.max.z(), other.max.z()),
            ),
        }
    }

    /// Return a new box expanded by `margin` on each side.
    #[inline]
    #[must_use]
    pub fn expanded(self, margin: f64) -> Self {
        Self {
            min: self.min + Vec3::new(-margin, -margin, -margin),
            max: self.max + Vec3::new(margin, margin, margin),
        }
    }

    /// Surface area of the box (used for SAH cost in BVH).
    #[must_use]
    pub fn surface_area(self) -> f64 {
        let d = self.max - self.min;
        2.0 * d.x().madd(d.y(), d.y().madd(d.z(), d.z() * d.x()))
    }

    /// Center point of the bounding box.
    #[must_use]
    pub fn center(self) -> Point3 {
        Point3::new(
            self.min.x().madd(0.5, self.max.x() * 0.5),
            self.min.y().madd(0.5, self.max.y() * 0.5),
            self.min.z().madd(0.5, self.max.z() * 0.5),
        )
    }

    /// Whether the whole line through `origin` along `dir`, both ways,
    /// meets this box (touching counts).
    ///
    /// A zero direction component is tested by the origin's coordinate
    /// alone, so an origin lying on a face of the box with the line in that
    /// face still meets it.
    #[must_use]
    pub fn line_intersects(self, origin: Point3, dir: Vec3) -> bool {
        if !self.is_number() {
            return false;
        }
        let mut lo = f64::NEG_INFINITY;
        let mut hi = f64::INFINITY;
        for (o, d, min, max) in [
            (origin.x(), dir.x(), self.min.x(), self.max.x()),
            (origin.y(), dir.y(), self.min.y(), self.max.y()),
            (origin.z(), dir.z(), self.min.z(), self.max.z()),
        ] {
            if d == 0.0 {
                if o < min || o > max {
                    return false;
                }
                continue;
            }
            let (t1, t2) = ((min - o) / d, (max - o) / d);
            lo = lo.max(t1.min(t2));
            hi = hi.min(t1.max(t2));
            if lo > hi {
                return false;
            }
        }
        true
    }

    /// Test whether a ray (origin + positive-t direction) intersects this box.
    ///
    /// Uses the slab method. Returns `true` if the ray hits the box at any
    /// `t >= 0`.
    #[must_use]
    pub fn ray_intersects(self, origin: Point3, inv_dir: Vec3) -> bool {
        if !self.is_number() {
            return false;
        }
        let t1x = (self.min.x() - origin.x()) * inv_dir.x();
        let t2x = (self.max.x() - origin.x()) * inv_dir.x();
        let t1y = (self.min.y() - origin.y()) * inv_dir.y();
        let t2y = (self.max.y() - origin.y()) * inv_dir.y();
        let t1z = (self.min.z() - origin.z()) * inv_dir.z();
        let t2z = (self.max.z() - origin.z()) * inv_dir.z();

        let tmin = t1x.min(t2x).max(t1y.min(t2y)).max(t1z.min(t2z));
        let tmax = t1x.max(t2x).min(t1y.max(t2y)).min(t1z.max(t2z));

        tmax >= tmin.max(0.0)
    }

    /// Squared distance from a point to the closest point on the box: not
    /// a number if a coordinate of the box or the point is not.
    #[must_use]
    pub fn distance_squared_to_point(self, p: Point3) -> f64 {
        let gap = |min: f64, max: f64, c: f64| upper(upper(min - c, 0.0), c - max);
        let dx = gap(self.min.x(), self.max.x(), p.x());
        let dy = gap(self.min.y(), self.max.y(), p.y());
        let dz = gap(self.min.z(), self.max.z(), p.z());
        dx.madd(dx, dy.madd(dy, dz * dz))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb3_from_points() {
        let bb = Aabb3::from_points([
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(-1.0, 5.0, 0.0),
            Point3::new(3.0, 0.0, 1.0),
        ]);
        assert_eq!(bb.min, Point3::new(-1.0, 0.0, 0.0));
        assert_eq!(bb.max, Point3::new(3.0, 5.0, 3.0));
    }

    #[test]
    fn aabb3_empty_returns_none() {
        let bb = Aabb3::try_from_points(std::iter::empty());
        assert!(bb.is_none());
    }

    #[test]
    fn aabb3_intersects() {
        let a = Aabb3::from_points([Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 2.0, 2.0)]);
        let b = Aabb3::from_points([Point3::new(1.0, 1.0, 1.0), Point3::new(3.0, 3.0, 3.0)]);
        let c = Aabb3::from_points([Point3::new(5.0, 5.0, 5.0), Point3::new(6.0, 6.0, 6.0)]);
        assert!(a.intersects(b));
        assert!(!a.intersects(c));
    }

    #[test]
    fn aabb3_contains_point() {
        let bb = Aabb3::from_points([Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)]);
        assert!(bb.contains_point(Point3::new(0.5, 0.5, 0.5)));
        assert!(!bb.contains_point(Point3::new(2.0, 0.5, 0.5)));
    }

    #[test]
    fn aabb3_union() {
        let a = Aabb3::from_points([Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)]);
        let b = Aabb3::from_points([Point3::new(2.0, 2.0, 2.0), Point3::new(3.0, 3.0, 3.0)]);
        let u = a.union(b);
        assert_eq!(u.min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(u.max, Point3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn aabb3_expanded() {
        let bb = Aabb3::from_points([Point3::new(1.0, 1.0, 1.0), Point3::new(2.0, 2.0, 2.0)]);
        let ex = bb.expanded(0.5);
        assert!((ex.min.x() - 0.5).abs() < 1e-14);
        assert!((ex.max.x() - 2.5).abs() < 1e-14);
    }

    #[test]
    fn aabb3_surface_area() {
        let bb = Aabb3::from_points([Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 2.0, 3.0)]);
        // SA = 2*(1*2 + 2*3 + 3*1) = 2*11 = 22
        assert!((bb.surface_area() - 22.0).abs() < 1e-14);
    }

    #[test]
    fn aabb3_distance_to_point() {
        let bb = Aabb3::from_points([Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)]);
        // Point inside: distance is 0
        assert!(bb.distance_squared_to_point(Point3::new(0.5, 0.5, 0.5)) < 1e-14);
        // Point outside along x axis: distance = 1.0
        assert!((bb.distance_squared_to_point(Point3::new(2.0, 0.5, 0.5)) - 1.0).abs() < 1e-14);
    }

    /// A coordinate that is not a number spoils that coordinate of the box,
    /// first point or later, and the fallible form refuses it; it refuses an
    /// infinite coordinate and no point too.
    #[test]
    fn a_box_hides_no_point_that_is_not_finite() {
        let (a, b) = (Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 2.0, 3.0));
        let bad = Point3::new(0.5, f64::NAN, 0.5);
        for points in [vec![bad, a, b], vec![a, bad, b], vec![a, b, bad]] {
            let bb = Aabb3::from_points(points.iter().copied());
            assert!(
                bb.min.y().is_nan() && bb.max.y().is_nan(),
                "{points:?}: {bb:?}"
            );
            assert!(bb.min.x().to_bits() == 0.0_f64.to_bits() && bb.max.x() > 0.5);
            assert!(!bb.is_finite() && !bb.is_number());
            assert!(Aabb3::try_from_points(points.iter().copied()).is_none());
            assert!(!bb.contains_point(Point3::new(0.5, 1.0, 0.5)));
            assert!(!bb.intersects(Aabb3::from_points([a, b])));
            assert!(
                bb.distance_squared_to_point(Point3::new(0.5, 1.0, 0.5))
                    .is_nan()
            );
            assert!(!bb.ray_intersects(
                Point3::new(-1.0, 1.0, 1.0),
                Vec3::new(1.0, f64::INFINITY, f64::INFINITY)
            ));
            assert!(!bb.line_intersects(Point3::new(-1.0, 1.0, 1.0), Vec3::new(1.0, 0.0, 0.0)));
        }
        let far = Point3::new(f64::INFINITY, 0.0, 0.0);
        assert!(Aabb3::try_from_points([a, far]).is_none());
        assert!(Aabb3::from_points([a, far]).is_number());
        let none = Aabb3::from_points(std::iter::empty());
        assert!(!none.is_number());
        assert!(Aabb3::try_from_points(std::iter::empty()).is_none());
        let flat = Point2::new(f64::NAN, 1.0);
        for points in [[flat, Point2::new(0.0, 0.0)], [Point2::new(0.0, 0.0), flat]] {
            assert!(Aabb2::from_points(points).min.x().is_nan());
            assert!(Aabb2::try_from_points(points).is_none());
        }
        assert!(Aabb2::try_from_points([Point2::new(0.0, f64::NEG_INFINITY)]).is_none());
        assert!(Aabb2::try_from_points(std::iter::empty()).is_none());
    }

    /// A union with a box that hides where a point lies hides it too, from
    /// either side, as a union with an infinite box keeps its infinite side.
    #[test]
    fn a_union_keeps_a_coordinate_that_is_not_a_number() {
        let good = Aabb3::from_points([Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0)]);
        let bad = Aabb3::from_points([Point3::new(f64::NAN, 0.0, 0.0)]);
        for u in [good.union(bad), bad.union(good)] {
            assert!(u.min.x().is_nan() && u.max.x().is_nan(), "{u:?}");
            assert!(!u.is_number());
        }
        assert!(good.union(Aabb3::WHOLE).min.x().is_infinite());
        assert!(good.expanded(f64::NAN).min.x().is_nan());
        let flat = Aabb2::from_points([Point2::new(0.0, 0.0)]);
        assert!(
            flat.union(Aabb2::from_points([Point2::new(0.0, f64::NAN)]))
                .max
                .y()
                .is_nan()
        );
    }

    #[test]
    fn aabb2_basic() {
        let bb = Aabb2::from_points([Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)]);
        assert!(bb.contains_point(Point2::new(0.5, 0.5)));
        assert!(!bb.contains_point(Point2::new(2.0, 0.5)));
    }
}
