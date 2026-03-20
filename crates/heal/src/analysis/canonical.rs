//! Canonical form analysis — recognize NURBS surfaces as elementary.
//!
//! Attempts to identify NURBS surfaces that represent known analytic
//! surfaces (planes, cylinders, cones, spheres) and returns the
//! corresponding [`RecognizedSurface`] variant.
//!
//! The recognition logic lives in `brepkit-geometry`; this module
//! provides the heal-crate's public type and a thin delegation wrapper
//! so existing callers are not affected.

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};

/// Result of surface recognition.
pub enum RecognizedSurface {
    /// Recognized as a plane.
    Plane {
        /// Normal vector.
        normal: Vec3,
        /// Signed distance from origin.
        d: f64,
    },
    /// Recognized as a cylinder.
    Cylinder {
        /// A point on the cylinder axis.
        origin: Point3,
        /// Axis direction (unit vector).
        axis: Vec3,
        /// Cylinder radius.
        radius: f64,
    },
    /// Recognized as a sphere.
    Sphere {
        /// Center of the sphere.
        center: Point3,
        /// Sphere radius.
        radius: f64,
    },
    /// Could not be recognized as an elementary surface.
    NotRecognized,
}

/// Attempt to recognize a NURBS surface as an elementary analytic surface.
///
/// Delegates to [`brepkit_geometry::convert::recognize_surface`] and
/// converts the result into the heal crate's own [`RecognizedSurface`].
///
/// Tries recognition in order: plane, cylinder, sphere. Returns the first
/// match that fits within tolerance.
#[must_use]
pub fn recognize_surface(surface: &NurbsSurface, tolerance: &Tolerance) -> RecognizedSurface {
    use brepkit_geometry::convert::RecognizedSurface as GeomRecognized;

    match brepkit_geometry::convert::recognize_surface(surface, tolerance.linear) {
        GeomRecognized::Plane { normal, d } => RecognizedSurface::Plane { normal, d },
        GeomRecognized::Cylinder {
            origin,
            axis,
            radius,
        } => RecognizedSurface::Cylinder {
            origin,
            axis,
            radius,
        },
        GeomRecognized::Sphere { center, radius } => RecognizedSurface::Sphere { center, radius },
        GeomRecognized::NotRecognized => RecognizedSurface::NotRecognized,
    }
}
