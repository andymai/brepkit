//! Ephemeral 2D parameterization for planar faces.
//!
//! `FaceSurface::Plane` stores only `(normal, d)` with no origin or UV axes.
//! `PlaneFrame` computes a local orthonormal frame so plane faces can
//! participate in the 2D parameter-space boolean pipeline.

#![allow(dead_code)] // Used by later pipeline stages.

use brepkit_math::vec::{Point2, Point3, Vec3};

/// A local 2D coordinate frame on a plane.
///
/// `project(p3) -> Point2` and `evaluate(u, v) -> Point3` convert between
/// 3D world space and the plane's 2D parameter space.
#[derive(Debug, Clone)]
pub struct PlaneFrame {
    origin: Point3,
    u_axis: Vec3,
    v_axis: Vec3,
}

impl PlaneFrame {
    /// Build a frame from a plane normal and a point on the plane.
    ///
    /// The u-axis is chosen perpendicular to the normal (stable for all
    /// orientations). The v-axis completes the right-handed frame:
    /// `v = normal x u`.
    #[must_use]
    pub fn from_normal_and_point(normal: Vec3, origin: Point3) -> Self {
        // Choose a seed vector not parallel to the normal.
        let seed = if normal.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        // u = normalize(normal x seed) -- perpendicular to normal
        let u_raw = normal.cross(seed);
        let u_axis = u_raw.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0));
        // v = normal x u -- completes the right-handed frame
        let v_axis = normal.cross(u_axis);
        Self {
            origin,
            u_axis,
            v_axis,
        }
    }

    /// Build a frame for a `FaceSurface::Plane` using the first wire vertex as origin.
    #[must_use]
    pub fn from_plane_face(normal: Vec3, wire_points: &[Point3]) -> Self {
        let origin = wire_points
            .first()
            .copied()
            .unwrap_or(Point3::new(0.0, 0.0, 0.0));
        Self::from_normal_and_point(normal, origin)
    }

    /// Project a 3D point onto the 2D parameter space.
    #[must_use]
    pub fn project(&self, p: Point3) -> Point2 {
        let d = p - self.origin;
        Point2::new(d.dot(self.u_axis), d.dot(self.v_axis))
    }

    /// Evaluate 2D parameters back to 3D.
    #[must_use]
    pub fn evaluate(&self, u: f64, v: f64) -> Point3 {
        self.origin + self.u_axis * u + self.v_axis * v
    }

    /// The u-axis direction.
    #[must_use]
    pub fn u_axis(&self) -> Vec3 {
        self.u_axis
    }

    /// The v-axis direction.
    #[must_use]
    pub fn v_axis(&self) -> Vec3 {
        self.v_axis
    }

    /// The frame origin.
    #[must_use]
    pub fn origin(&self) -> Point3 {
        self.origin
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn xy_plane_roundtrips() {
        let frame =
            PlaneFrame::from_normal_and_point(Vec3::new(0.0, 0.0, 1.0), Point3::new(0.0, 0.0, 0.0));
        let p3 = Point3::new(3.0, 4.0, 0.0);
        let p2 = frame.project(p3);
        let dist_2d = (p2.x() * p2.x() + p2.y() * p2.y()).sqrt();
        assert!((dist_2d - 5.0).abs() < 1e-10);
        let p3_back = frame.evaluate(p2.x(), p2.y());
        assert!((p3_back - p3).length() < 1e-10);
    }

    #[test]
    fn tilted_plane_roundtrips() {
        let normal = Vec3::new(1.0, 1.0, 1.0).normalize().unwrap();
        let origin = Point3::new(5.0, 5.0, 5.0);
        let frame = PlaneFrame::from_normal_and_point(normal, origin);
        let p3 = origin + frame.u_axis() * 2.0 + frame.v_axis() * 3.0;
        let p2 = frame.project(p3);
        assert!((p2.x() - 2.0).abs() < 1e-10);
        assert!((p2.y() - 3.0).abs() < 1e-10);
        let back = frame.evaluate(p2.x(), p2.y());
        assert!((back - p3).length() < 1e-10);
    }

    #[test]
    fn from_plane_face_uses_first_vertex() {
        let pts = vec![
            Point3::new(1.0, 2.0, 0.0),
            Point3::new(5.0, 2.0, 0.0),
            Point3::new(5.0, 6.0, 0.0),
        ];
        let frame = PlaneFrame::from_plane_face(Vec3::new(0.0, 0.0, 1.0), &pts);
        assert!((frame.origin() - pts[0]).length() < 1e-10);
        let p2 = frame.project(pts[0]);
        assert!(p2.x().abs() < 1e-10);
        assert!(p2.y().abs() < 1e-10);
    }

    #[test]
    fn axes_are_orthonormal() {
        let normal = Vec3::new(0.3, -0.7, 0.5).normalize().unwrap();
        let frame = PlaneFrame::from_normal_and_point(normal, Point3::new(0.0, 0.0, 0.0));
        assert!((frame.u_axis().length() - 1.0).abs() < 1e-10);
        assert!((frame.v_axis().length() - 1.0).abs() < 1e-10);
        assert!(frame.u_axis().dot(frame.v_axis()).abs() < 1e-10);
        assert!(frame.u_axis().dot(normal).abs() < 1e-10);
        assert!(frame.v_axis().dot(normal).abs() < 1e-10);
    }
}
