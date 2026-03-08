//! Axis-aligned bounding boxes for spatial queries.

use crate::vec::{Point2, Point3, Vec2, Vec3};

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
    /// Create an AABB from an iterator of points.
    ///
    /// # Panics
    ///
    /// Panics if the iterator is empty. Use [`Aabb2::try_from_points`] for
    /// a fallible version.
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn from_points(points: impl IntoIterator<Item = Point2>) -> Self {
        Self::try_from_points(points).expect("at least one point required")
    }

    /// Create an AABB from an iterator of points, returning `None` if empty.
    #[must_use]
    pub fn try_from_points(points: impl IntoIterator<Item = Point2>) -> Option<Self> {
        let mut iter = points.into_iter();
        let first = iter.next()?;
        let mut min = first;
        let mut max = first;
        for p in iter {
            if p.x() < min.x() {
                min.0[0] = p.x();
            }
            if p.y() < min.y() {
                min.0[1] = p.y();
            }
            if p.x() > max.x() {
                max.0[0] = p.x();
            }
            if p.y() > max.y() {
                max.0[1] = p.y();
            }
        }
        Some(Self { min, max })
    }

    /// Whether this box intersects another.
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
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            min: Point2::new(
                self.min.x().min(other.min.x()),
                self.min.y().min(other.min.y()),
            ),
            max: Point2::new(
                self.max.x().max(other.max.x()),
                self.max.y().max(other.max.y()),
            ),
        }
    }

    /// Return a new box expanded by `margin` on each side.
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
    /// Create an AABB from an iterator of points.
    ///
    /// # Panics
    ///
    /// Panics if the iterator is empty. Use [`Aabb3::try_from_points`] for
    /// a fallible version.
    #[must_use]
    #[allow(clippy::expect_used)]
    pub fn from_points(points: impl IntoIterator<Item = Point3>) -> Self {
        Self::try_from_points(points).expect("at least one point required")
    }

    /// Create an AABB from an iterator of points, returning `None` if empty.
    #[must_use]
    pub fn try_from_points(points: impl IntoIterator<Item = Point3>) -> Option<Self> {
        let mut iter = points.into_iter();
        let first = iter.next()?;
        let mut min = first;
        let mut max = first;
        for p in iter {
            if p.x() < min.x() {
                min.0[0] = p.x();
            }
            if p.y() < min.y() {
                min.0[1] = p.y();
            }
            if p.z() < min.z() {
                min.0[2] = p.z();
            }
            if p.x() > max.x() {
                max.0[0] = p.x();
            }
            if p.y() > max.y() {
                max.0[1] = p.y();
            }
            if p.z() > max.z() {
                max.0[2] = p.z();
            }
        }
        Some(Self { min, max })
    }

    /// Whether this box intersects another.
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
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            min: Point3::new(
                self.min.x().min(other.min.x()),
                self.min.y().min(other.min.y()),
                self.min.z().min(other.min.z()),
            ),
            max: Point3::new(
                self.max.x().max(other.max.x()),
                self.max.y().max(other.max.y()),
                self.max.z().max(other.max.z()),
            ),
        }
    }

    /// Return a new box expanded by `margin` on each side.
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
        2.0 * d.x().mul_add(d.y(), d.y().mul_add(d.z(), d.z() * d.x()))
    }

    /// Center point of the bounding box.
    #[must_use]
    pub fn center(self) -> Point3 {
        Point3::new(
            self.min.x().mul_add(0.5, self.max.x() * 0.5),
            self.min.y().mul_add(0.5, self.max.y() * 0.5),
            self.min.z().mul_add(0.5, self.max.z() * 0.5),
        )
    }

    /// Test whether a ray (origin + positive-t direction) intersects this box.
    ///
    /// Uses the slab method. Returns `true` if the ray hits the box at any
    /// `t >= 0`.
    #[must_use]
    pub fn ray_intersects(self, origin: Point3, inv_dir: Vec3) -> bool {
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

    /// Squared distance from a point to the closest point on the box.
    #[must_use]
    pub fn distance_squared_to_point(self, p: Point3) -> f64 {
        let dx = (self.min.x() - p.x()).max(0.0).max(p.x() - self.max.x());
        let dy = (self.min.y() - p.y()).max(0.0).max(p.y() - self.max.y());
        let dz = (self.min.z() - p.z()).max(0.0).max(p.z() - self.max.z());
        dx.mul_add(dx, dy.mul_add(dy, dz * dz))
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

    #[test]
    fn aabb2_basic() {
        let bb = Aabb2::from_points([Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)]);
        assert!(bb.contains_point(Point2::new(0.5, 0.5)));
        assert!(!bb.contains_point(Point2::new(2.0, 0.5)));
    }
}
