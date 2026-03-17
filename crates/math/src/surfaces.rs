//! Analytic surface types for exact geometric computations.
//!
//! These surfaces complement NURBS surfaces by providing exact parameterizations
//! for common shapes (cylinder, cone, sphere, torus). This enables exact
//! intersection algorithms (e.g., plane-cylinder = ellipse) without sampling.

use crate::MathError;
use crate::frame::Frame3;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

/// An infinite cylindrical surface.
///
/// Parameterized as `P(u, v) = origin + radius*(cos(u)*x_axis + sin(u)*y_axis) + v*axis`
/// where `u ∈ [0, 2π)` and `v ∈ (-∞, +∞)`.
#[derive(Debug, Clone)]
pub struct CylindricalSurface {
    origin: Point3,
    axis: Vec3,
    radius: f64,
    x_axis: Vec3,
    y_axis: Vec3,
}

impl CylindricalSurface {
    /// Creates a new cylindrical surface.
    ///
    /// # Errors
    /// Returns an error if radius is not positive or axis is zero.
    pub fn new(origin: Point3, axis: Vec3, radius: f64) -> Result<Self, MathError> {
        if radius <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: radius,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        let f = Frame3::from_normal(origin, axis)?;
        Ok(Self {
            origin,
            axis: f.z,
            radius,
            x_axis: f.x,
            y_axis: f.y,
        })
    }

    /// Evaluates the surface at parameters `(u, v)`.
    #[must_use]
    pub fn evaluate(&self, u: f64, v: f64) -> Point3 {
        let (sin_u, cos_u) = u.sin_cos();
        self.origin
            + self.x_axis * (self.radius * cos_u)
            + self.y_axis * (self.radius * sin_u)
            + self.axis * v
    }

    /// Returns the surface normal at parameters `(u, v)`.
    #[must_use]
    pub fn normal(&self, u: f64, _v: f64) -> Vec3 {
        let (sin_u, cos_u) = u.sin_cos();
        self.x_axis * cos_u + self.y_axis * sin_u
    }

    /// Returns the origin.
    #[must_use]
    pub const fn origin(&self) -> Point3 {
        self.origin
    }

    /// Returns the axis direction.
    #[must_use]
    pub const fn axis(&self) -> Vec3 {
        self.axis
    }

    /// Returns the radius.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// Returns the local X axis (first radial direction in the parametric frame).
    #[must_use]
    pub const fn x_axis(&self) -> Vec3 {
        self.x_axis
    }

    /// Returns the local Y axis (second radial direction in the parametric frame).
    #[must_use]
    pub const fn y_axis(&self) -> Vec3 {
        self.y_axis
    }

    /// Creates a cylindrical surface with a specified reference direction.
    ///
    /// `ref_dir` defines the x-axis of the parametric frame (projected
    /// perpendicular to `axis`). This preserves the parametric orientation
    /// from STEP `AXIS2_PLACEMENT_3D` or BREP round-trips.
    ///
    /// # Errors
    /// Returns an error if radius is not positive or axis is zero.
    pub fn with_ref_dir(
        origin: Point3,
        axis: Vec3,
        radius: f64,
        ref_dir: Vec3,
    ) -> Result<Self, MathError> {
        if radius <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: radius,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        let f = Frame3::from_normal_and_ref(origin, axis, ref_dir)?;
        Ok(Self {
            origin,
            axis: f.z,
            radius,
            x_axis: f.x,
            y_axis: f.y,
        })
    }

    /// Returns a copy of this cylinder with its origin translated by `offset`.
    #[must_use]
    pub fn translated(&self, offset: Vec3) -> Self {
        Self {
            origin: self.origin + offset,
            ..self.clone()
        }
    }

    /// Project a 3D point onto the cylinder surface, returning (u, v) parameters.
    ///
    /// `u` is the angular parameter [0, 2π), `v` is the axial parameter.
    #[must_use]
    pub fn project_point(&self, point: Point3) -> (f64, f64) {
        let to_pt = Vec3::new(
            point.x() - self.origin.x(),
            point.y() - self.origin.y(),
            point.z() - self.origin.z(),
        );
        let v = self.axis.dot(to_pt);
        let radial = to_pt - self.axis * v;
        let x = self.x_axis.dot(radial);
        let y = self.y_axis.dot(radial);
        let u = y.atan2(x).rem_euclid(std::f64::consts::TAU);
        (u, v)
    }

    /// Convert to an exact rational NURBS surface over the given v-range.
    ///
    /// Uses degree (2, 1) with 9 control points per ring (standard rational
    /// representation of a full circle). The result is geometrically exact.
    ///
    /// # Errors
    ///
    /// Returns an error if `NurbsSurface` construction fails.
    pub fn to_nurbs(&self, v_min: f64, v_max: f64) -> Result<NurbsSurface, MathError> {
        // 9 CPs for a full circle (degree 2, 4 arcs of 90°).
        let w1 = std::f64::consts::FRAC_1_SQRT_2;
        let circle_weights = [1.0, w1, 1.0, w1, 1.0, w1, 1.0, w1, 1.0];
        // Directions at 0°, 45°, 90°, ... 360° in the (x_axis, y_axis) plane.
        let dirs: [(f64, f64); 9] = [
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 0.0),
            (-1.0, -1.0),
            (0.0, -1.0),
            (1.0, -1.0),
            (1.0, 0.0),
        ];

        let mut cps = Vec::with_capacity(9);
        let mut ws = Vec::with_capacity(9);
        for (i, &(dx, dy)) in dirs.iter().enumerate() {
            let radial = self.x_axis * (self.radius * dx) + self.y_axis * (self.radius * dy);
            let p_bot = self.origin + radial + self.axis * v_min;
            let p_top = self.origin + radial + self.axis * v_max;
            cps.push(vec![p_bot, p_top]);
            ws.push(vec![circle_weights[i], circle_weights[i]]);
        }

        let knots_u = vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.0];
        let knots_v = vec![0.0, 0.0, 1.0, 1.0];
        NurbsSurface::new(2, 1, knots_u, knots_v, cps, ws)
    }
}

/// An infinite conical surface.
///
/// Parameterized as `P(u, v) = apex + v*(cos(half_angle)*(cos(u)*x_axis + sin(u)*y_axis) + sin(half_angle)*axis)`
/// where `u ∈ [0, 2π)` and `v ∈ [0, +∞)`.
#[derive(Debug, Clone)]
pub struct ConicalSurface {
    apex: Point3,
    axis: Vec3,
    half_angle: f64,
    x_axis: Vec3,
    y_axis: Vec3,
}

impl ConicalSurface {
    /// Creates a new conical surface.
    ///
    /// `half_angle` is the angle from the radial plane to the cone's surface
    /// generator (radians). Small angles produce wide/flat cones; angles near
    /// π/2 produce narrow/spike cones. In the evaluate formula
    /// `P(u,v) = apex + v*(cos(a)*radial + sin(a)*axis)`, `a` is this angle.
    ///
    /// # Errors
    /// Returns an error if half-angle is not in `(0, π/2)` or axis is zero.
    pub fn new(apex: Point3, axis: Vec3, half_angle: f64) -> Result<Self, MathError> {
        if half_angle <= 0.0 || half_angle >= std::f64::consts::FRAC_PI_2 {
            return Err(MathError::ParameterOutOfRange {
                value: half_angle,
                min: f64::EPSILON,
                max: std::f64::consts::FRAC_PI_2,
            });
        }
        let f = Frame3::from_normal(apex, axis)?;
        Ok(Self {
            apex,
            axis: f.z,
            half_angle,
            x_axis: f.x,
            y_axis: f.y,
        })
    }

    /// Evaluates the surface at parameters `(u, v)`.
    #[must_use]
    pub fn evaluate(&self, u: f64, v: f64) -> Point3 {
        let (sin_u, cos_u) = u.sin_cos();
        let (sin_a, cos_a) = self.half_angle.sin_cos();
        let radial = self.x_axis * cos_u + self.y_axis * sin_u;
        self.apex + (radial * cos_a + self.axis * sin_a) * v
    }

    /// Returns the surface normal at parameters `(u, v)`.
    #[must_use]
    pub fn normal(&self, u: f64, _v: f64) -> Vec3 {
        let (sin_u, cos_u) = u.sin_cos();
        let (sin_a, cos_a) = self.half_angle.sin_cos();
        let radial = self.x_axis * cos_u + self.y_axis * sin_u;
        // Normal points outward: radial * sin(a) - axis * cos(a)
        radial * sin_a + self.axis * (-cos_a)
    }

    /// Returns the apex point.
    #[must_use]
    pub const fn apex(&self) -> Point3 {
        self.apex
    }

    /// Returns the axis direction.
    #[must_use]
    pub const fn axis(&self) -> Vec3 {
        self.axis
    }

    /// Returns the half-angle in radians.
    #[must_use]
    pub const fn half_angle(&self) -> f64 {
        self.half_angle
    }

    /// Returns the local X axis (first radial direction in the parametric frame).
    #[must_use]
    pub const fn x_axis(&self) -> Vec3 {
        self.x_axis
    }

    /// Returns the local Y axis (second radial direction in the parametric frame).
    #[must_use]
    pub const fn y_axis(&self) -> Vec3 {
        self.y_axis
    }

    /// Creates a conical surface with a specified reference direction.
    ///
    /// `ref_dir` defines the x-axis of the parametric frame (projected
    /// perpendicular to `axis`). This preserves the parametric orientation
    /// from STEP `AXIS2_PLACEMENT_3D` or BREP round-trips.
    ///
    /// # Errors
    /// Returns an error if half-angle is not in `(0, π/2)` or axis is zero.
    pub fn with_ref_dir(
        apex: Point3,
        axis: Vec3,
        half_angle: f64,
        ref_dir: Vec3,
    ) -> Result<Self, MathError> {
        if half_angle <= 0.0 || half_angle >= std::f64::consts::FRAC_PI_2 {
            return Err(MathError::ParameterOutOfRange {
                value: half_angle,
                min: f64::EPSILON,
                max: std::f64::consts::FRAC_PI_2,
            });
        }
        let f = Frame3::from_normal_and_ref(apex, axis, ref_dir)?;
        Ok(Self {
            apex,
            axis: f.z,
            half_angle,
            x_axis: f.x,
            y_axis: f.y,
        })
    }

    /// Returns a copy of this cone with its apex translated by `offset`.
    #[must_use]
    pub fn translated(&self, offset: Vec3) -> Self {
        Self {
            apex: self.apex + offset,
            ..self.clone()
        }
    }

    /// Returns the radius at a given distance `v` along the axis from the apex.
    #[must_use]
    pub fn radius_at(&self, v: f64) -> f64 {
        v * self.half_angle.cos()
    }

    /// Project a 3D point onto the cone surface, returning `(u, v)` parameters.
    ///
    /// `u` is the angular parameter `[0, 2π)`, `v` is the distance from the
    /// apex along the cone surface generator line.
    #[must_use]
    pub fn project_point(&self, point: Point3) -> (f64, f64) {
        let to_pt = Vec3::new(
            point.x() - self.apex.x(),
            point.y() - self.apex.y(),
            point.z() - self.apex.z(),
        );

        let h = self.axis.dot(to_pt);
        let radial = to_pt - self.axis * h;
        let x = self.x_axis.dot(radial);
        let y = self.y_axis.dot(radial);

        let u = y.atan2(x).rem_euclid(std::f64::consts::TAU);

        let sin_a = self.half_angle.sin();
        let v = if sin_a.abs() > 1e-15 {
            h / sin_a
        } else {
            let cos_a = self.half_angle.cos();
            if cos_a.abs() > 1e-15 {
                radial.length() / cos_a
            } else {
                0.0
            }
        };

        (u, v)
    }

    /// Convert to an approximate NURBS surface over the given v-range.
    ///
    /// # Errors
    ///
    /// Returns an error if `NurbsSurface` construction fails.
    pub fn to_nurbs(&self, v_min: f64, v_max: f64) -> Result<NurbsSurface, MathError> {
        analytic_to_nurbs_sampled(
            |u, v| self.evaluate(u, v),
            (0.0, std::f64::consts::TAU),
            (v_min, v_max),
        )
    }
}

/// An infinite spherical surface (actually a sphere).
///
/// Parameterized as `P(u, v) = center + radius*(cos(v)*cos(u)*x + cos(v)*sin(u)*y + sin(v)*z)`
/// where `u ∈ [0, 2π)` (longitude) and `v ∈ [-π/2, π/2]` (latitude).
#[derive(Debug, Clone)]
pub struct SphericalSurface {
    center: Point3,
    radius: f64,
    x_axis: Vec3,
    y_axis: Vec3,
    z_axis: Vec3,
}

impl SphericalSurface {
    /// Creates a new spherical surface.
    ///
    /// # Errors
    /// Returns an error if radius is not positive.
    pub fn new(center: Point3, radius: f64) -> Result<Self, MathError> {
        if radius <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: radius,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        Ok(Self {
            center,
            radius,
            x_axis: Vec3::new(1.0, 0.0, 0.0),
            y_axis: Vec3::new(0.0, 1.0, 0.0),
            z_axis: Vec3::new(0.0, 0.0, 1.0),
        })
    }

    /// Creates a spherical surface with a custom orientation.
    ///
    /// # Errors
    /// Returns an error if radius is not positive or the z-axis is zero.
    pub fn with_axis(center: Point3, radius: f64, z_axis: Vec3) -> Result<Self, MathError> {
        if radius <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: radius,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        let f = Frame3::from_normal(center, z_axis)?;
        Ok(Self {
            center,
            radius,
            x_axis: f.x,
            y_axis: f.y,
            z_axis: f.z,
        })
    }

    /// Evaluates the surface at parameters `(u, v)`.
    #[must_use]
    pub fn evaluate(&self, u: f64, v: f64) -> Point3 {
        let (sin_u, cos_u) = u.sin_cos();
        let (sin_v, cos_v) = v.sin_cos();
        self.center
            + self.x_axis * (self.radius * cos_v * cos_u)
            + self.y_axis * (self.radius * cos_v * sin_u)
            + self.z_axis * (self.radius * sin_v)
    }

    /// Returns the outward normal at parameters `(u, v)`.
    #[must_use]
    pub fn normal(&self, u: f64, v: f64) -> Vec3 {
        let (sin_u, cos_u) = u.sin_cos();
        let (sin_v, cos_v) = v.sin_cos();
        self.x_axis * (cos_v * cos_u) + self.y_axis * (cos_v * sin_u) + self.z_axis * sin_v
    }

    /// Returns the center.
    #[must_use]
    pub const fn center(&self) -> Point3 {
        self.center
    }

    /// Returns the radius.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// Returns the local X axis.
    #[must_use]
    pub const fn x_axis(&self) -> Vec3 {
        self.x_axis
    }

    /// Returns the local Y axis.
    #[must_use]
    pub const fn y_axis(&self) -> Vec3 {
        self.y_axis
    }

    /// Returns the local Z axis (pole direction).
    #[must_use]
    pub const fn z_axis(&self) -> Vec3 {
        self.z_axis
    }

    /// Returns a copy of this sphere with its center translated by `offset`.
    #[must_use]
    pub fn translated(&self, offset: Vec3) -> Self {
        Self {
            center: self.center + offset,
            ..self.clone()
        }
    }

    /// Project a 3D point onto the sphere, returning (u, v) parameters.
    ///
    /// `u` is the longitudinal angle [0, 2π), `v` is the latitude [-π/2, π/2].
    #[must_use]
    pub fn project_point(&self, point: Point3) -> (f64, f64) {
        let to_pt = Vec3::new(
            point.x() - self.center.x(),
            point.y() - self.center.y(),
            point.z() - self.center.z(),
        );
        let r = to_pt.length();
        if r < 1e-15 {
            return (0.0, 0.0);
        }
        let x = self.x_axis.dot(to_pt);
        let y = self.y_axis.dot(to_pt);
        let z = self.z_axis.dot(to_pt);
        let u = y.atan2(x).rem_euclid(std::f64::consts::TAU);
        let v = (z / r).clamp(-1.0, 1.0).asin();
        (u, v)
    }

    /// Convert to an approximate NURBS surface.
    ///
    /// # Errors
    ///
    /// Returns an error if `NurbsSurface` construction fails.
    pub fn to_nurbs(&self) -> Result<NurbsSurface, MathError> {
        analytic_to_nurbs_sampled(
            |u, v| self.evaluate(u, v),
            (0.0, std::f64::consts::TAU),
            (-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
        )
    }
}

/// A toroidal surface.
///
/// Parameterized as `P(u, v) = center + (R + r*cos(v))*(cos(u)*x + sin(u)*y) + r*sin(v)*z`
/// where `R` is the major radius, `r` is the minor radius,
/// `u ∈ [0, 2π)` (around the tube) and `v ∈ [0, 2π)` (around the cross-section).
#[derive(Debug, Clone)]
pub struct ToroidalSurface {
    center: Point3,
    major_radius: f64,
    minor_radius: f64,
    x_axis: Vec3,
    y_axis: Vec3,
    z_axis: Vec3,
}

impl ToroidalSurface {
    /// Creates a new toroidal surface.
    ///
    /// # Errors
    /// Returns an error if either radius is not positive or `minor_radius > major_radius`.
    pub fn new(center: Point3, major_radius: f64, minor_radius: f64) -> Result<Self, MathError> {
        if major_radius <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: major_radius,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        if minor_radius <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: minor_radius,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        Ok(Self {
            center,
            major_radius,
            minor_radius,
            x_axis: Vec3::new(1.0, 0.0, 0.0),
            y_axis: Vec3::new(0.0, 1.0, 0.0),
            z_axis: Vec3::new(0.0, 0.0, 1.0),
        })
    }

    /// Creates a toroidal surface with a specified axis direction.
    ///
    /// The axis is the central symmetry axis of the torus. The local
    /// coordinate frame is derived from it.
    ///
    /// # Errors
    /// Returns an error if either radius is not positive or axis is zero.
    pub fn with_axis(
        center: Point3,
        major_radius: f64,
        minor_radius: f64,
        z_axis: Vec3,
    ) -> Result<Self, MathError> {
        if major_radius <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: major_radius,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        if minor_radius <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: minor_radius,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        let f = Frame3::from_normal(center, z_axis)?;
        Ok(Self {
            center,
            major_radius,
            minor_radius,
            x_axis: f.x,
            y_axis: f.y,
            z_axis: f.z,
        })
    }

    /// Create a torus with explicit axis and reference direction.
    ///
    /// `ref_dir` defines the x-axis of the local frame (projected
    /// perpendicular to `z_axis`). This preserves the parametric
    /// orientation from STEP `AXIS2_PLACEMENT_3D`.
    ///
    /// # Errors
    ///
    /// Returns [`MathError::ParameterOutOfRange`] if either radius is
    /// non-positive, or [`MathError::ZeroVector`] if `z_axis` is zero.
    ///
    /// # Panics
    ///
    /// Panics if the fallback perpendicular vector cannot be normalized
    /// (should not occur for any valid unit `z_axis`).
    pub fn with_axis_and_ref_dir(
        center: Point3,
        major_radius: f64,
        minor_radius: f64,
        z_axis: Vec3,
        ref_dir: Vec3,
    ) -> Result<Self, MathError> {
        if major_radius <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: major_radius,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        if minor_radius <= 0.0 {
            return Err(MathError::ParameterOutOfRange {
                value: minor_radius,
                min: f64::EPSILON,
                max: f64::MAX,
            });
        }
        let f = Frame3::from_normal_and_ref(center, z_axis, ref_dir)?;
        Ok(Self {
            center,
            major_radius,
            minor_radius,
            x_axis: f.x,
            y_axis: f.y,
            z_axis: f.z,
        })
    }

    /// Evaluates the surface at parameters `(u, v)`.
    #[must_use]
    pub fn evaluate(&self, u: f64, v: f64) -> Point3 {
        let (sin_u, cos_u) = u.sin_cos();
        let (sin_v, cos_v) = v.sin_cos();
        let tube_radius = self.minor_radius.mul_add(cos_v, self.major_radius);
        self.center
            + self.x_axis * (tube_radius * cos_u)
            + self.y_axis * (tube_radius * sin_u)
            + self.z_axis * (self.minor_radius * sin_v)
    }

    /// Returns the outward surface normal at parameters `(u, v)`.
    #[must_use]
    pub fn normal(&self, u: f64, v: f64) -> Vec3 {
        let (sin_u, cos_u) = u.sin_cos();
        let (sin_v, cos_v) = v.sin_cos();
        let radial = self.x_axis * cos_u + self.y_axis * sin_u;
        radial * cos_v + self.z_axis * sin_v
    }

    /// Returns the center.
    #[must_use]
    pub const fn center(&self) -> Point3 {
        self.center
    }

    /// Returns a copy of this torus with its center translated by `offset`.
    #[must_use]
    pub fn translated(&self, offset: Vec3) -> Self {
        Self {
            center: self.center + offset,
            ..self.clone()
        }
    }

    /// Returns the major radius (distance from center to tube center).
    #[must_use]
    pub const fn major_radius(&self) -> f64 {
        self.major_radius
    }

    /// Returns the minor radius (tube cross-section radius).
    #[must_use]
    pub const fn minor_radius(&self) -> f64 {
        self.minor_radius
    }

    /// Returns the local X axis.
    #[must_use]
    pub const fn x_axis(&self) -> Vec3 {
        self.x_axis
    }

    /// Returns the local Y axis.
    #[must_use]
    pub const fn y_axis(&self) -> Vec3 {
        self.y_axis
    }

    /// Returns the torus axis direction (perpendicular to the ring plane).
    #[must_use]
    pub const fn z_axis(&self) -> Vec3 {
        self.z_axis
    }

    /// Project a 3D point onto the torus surface, returning `(u, v)` parameters.
    ///
    /// `u ∈ [0, 2π)` is the angle around the major circle.
    /// `v ∈ [0, 2π)` is the angle around the tube cross-section.
    #[must_use]
    pub fn project_point(&self, point: Point3) -> (f64, f64) {
        let to_pt = Vec3::new(
            point.x() - self.center.x(),
            point.y() - self.center.y(),
            point.z() - self.center.z(),
        );

        let x_comp = self.x_axis.dot(to_pt);
        let y_comp = self.y_axis.dot(to_pt);
        let u = y_comp.atan2(x_comp).rem_euclid(std::f64::consts::TAU);

        let (sin_u, cos_u) = u.sin_cos();
        let tube_center = self.center
            + self.x_axis * (self.major_radius * cos_u)
            + self.y_axis * (self.major_radius * sin_u);

        let to_tube = Vec3::new(
            point.x() - tube_center.x(),
            point.y() - tube_center.y(),
            point.z() - tube_center.z(),
        );

        let radial_dir = self.x_axis * cos_u + self.y_axis * sin_u;
        let r_comp = radial_dir.dot(to_tube);
        let z_comp = self.z_axis.dot(to_tube);

        let v = z_comp.atan2(r_comp).rem_euclid(std::f64::consts::TAU);
        (u, v)
    }

    /// Convert to an approximate NURBS surface.
    ///
    /// # Errors
    ///
    /// Returns an error if `NurbsSurface` construction fails.
    pub fn to_nurbs(&self) -> Result<NurbsSurface, MathError> {
        analytic_to_nurbs_sampled(
            |u, v| self.evaluate(u, v),
            (0.0, std::f64::consts::TAU),
            (0.0, std::f64::consts::TAU),
        )
    }
}

/// A surface of revolution created by revolving a curve around an axis.
///
/// Parameterized as `P(u, v) = origin + (curve(v) ⊗ rotation(u, axis))`
/// where `u ∈ [0, 2π)` is the revolution angle and `v` parameterizes
/// the generatrix curve.
#[derive(Debug, Clone)]
pub struct RevolutionSurface {
    origin: Point3,
    axis: Vec3,
    x_axis: Vec3,
    y_axis: Vec3,
    /// The generatrix (meridian) curve in the `(distance_from_axis, height)` plane.
    generatrix_radii: Vec<f64>,
    generatrix_heights: Vec<f64>,
}

impl RevolutionSurface {
    /// Creates a surface of revolution from a set of meridian profile points.
    ///
    /// Each point `(radius, height)` defines the generatrix in the rotation plane.
    ///
    /// # Errors
    /// Returns an error if the profile is empty or the axis is zero.
    pub fn new(
        origin: Point3,
        axis: Vec3,
        radii: Vec<f64>,
        heights: Vec<f64>,
    ) -> Result<Self, MathError> {
        if radii.is_empty() || heights.is_empty() {
            return Err(MathError::EmptyInput);
        }
        if radii.len() != heights.len() {
            return Err(MathError::InvalidWeights {
                expected: radii.len(),
                got: heights.len(),
            });
        }
        let f = Frame3::from_normal(origin, axis)?;
        Ok(Self {
            origin,
            axis: f.z,
            x_axis: f.x,
            y_axis: f.y,
            generatrix_radii: radii,
            generatrix_heights: heights,
        })
    }

    /// Evaluates at `(u, v)` where `u` is the revolution angle and `v ∈ [0, 1]`
    /// parameterizes the generatrix via linear interpolation.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn evaluate(&self, u: f64, v: f64) -> Point3 {
        let num_pts = self.generatrix_radii.len();
        let param = v.clamp(0.0, 1.0) * (num_pts - 1) as f64;
        let idx = (param as usize).min(num_pts - 2);
        let frac = param - idx as f64;

        let r = frac.mul_add(
            self.generatrix_radii[idx + 1] - self.generatrix_radii[idx],
            self.generatrix_radii[idx],
        );
        let height = frac.mul_add(
            self.generatrix_heights[idx + 1] - self.generatrix_heights[idx],
            self.generatrix_heights[idx],
        );

        let (sin_u, cos_u) = u.sin_cos();
        self.origin + self.x_axis * (r * cos_u) + self.y_axis * (r * sin_u) + self.axis * height
    }

    /// Returns the origin.
    #[must_use]
    pub const fn origin(&self) -> Point3 {
        self.origin
    }

    /// Returns the axis.
    #[must_use]
    pub const fn axis(&self) -> Vec3 {
        self.axis
    }
}

// ---------------------------------------------------------------------------
// Analytic → NURBS conversion helper
// ---------------------------------------------------------------------------

/// Sample an analytic surface on a grid and build a degree (1,1) NURBS surface.
///
/// Evaluates the surface at a regular grid of (u, v) parameters and constructs
/// a `NurbsSurface` whose control points pass through the sampled positions.
fn analytic_to_nurbs_sampled(
    surface_fn: impl Fn(f64, f64) -> Point3,
    u_range: (f64, f64),
    v_range: (f64, f64),
) -> Result<NurbsSurface, MathError> {
    // Dense sampling reduces chord-height error. For angular coordinates
    // (u on cylinder/sphere), 32 spans → max error R*(1-cos(π/32)) ≈ 0.005*R.
    // For v (latitude/height), 8 spans keeps error under 0.02*R.
    let nu = 33;
    let nv = 9;

    let mut cps = Vec::with_capacity(nu);
    let mut weights = Vec::with_capacity(nu);

    #[allow(clippy::cast_precision_loss)]
    for iu in 0..nu {
        let u = u_range.0 + (u_range.1 - u_range.0) * (iu as f64 / (nu - 1) as f64);
        let mut row = Vec::with_capacity(nv);
        let mut w_row = Vec::with_capacity(nv);
        for iv in 0..nv {
            let v = v_range.0 + (v_range.1 - v_range.0) * (iv as f64 / (nv - 1) as f64);
            row.push(surface_fn(u, v));
            w_row.push(1.0);
        }
        cps.push(row);
        weights.push(w_row);
    }

    // Uniform clamped knot vectors for degree 1 (bilinear interpolation
    // through the sample grid — control points ARE surface points).
    let knots_u = uniform_clamped_knots(nu, 1);
    let knots_v = uniform_clamped_knots(nv, 1);

    NurbsSurface::new(1, 1, knots_u, knots_v, cps, weights)
}

/// Build a uniform clamped knot vector for `n` control points at the given degree.
///
/// Produces `degree+1` zeros, then evenly spaced interior knots, then `degree+1` ones.
/// For degree-1 NURBS this gives bilinear interpolation through all control points.
#[allow(clippy::cast_precision_loss)]
fn uniform_clamped_knots(n: usize, degree: usize) -> Vec<f64> {
    let mut k = vec![0.0; degree + 1];
    for i in 1..n - degree {
        k.push(i as f64 / (n - degree) as f64);
    }
    k.extend(vec![1.0; degree + 1]);
    k
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    fn point_approx_eq(a: Point3, b: Point3) -> bool {
        approx_eq(a.x(), b.x()) && approx_eq(a.y(), b.y()) && approx_eq(a.z(), b.z())
    }

    // ── Cylinder tests ────────────────────────────────

    #[test]
    fn cylinder_at_origin() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0)
                .unwrap();

        let p = cyl.evaluate(0.0, 0.0);
        assert!(approx_eq((p - Point3::new(0.0, 0.0, 0.0)).length(), 2.0));

        let p = cyl.evaluate(0.0, 5.0);
        assert!(approx_eq(p.z(), 5.0));
    }

    #[test]
    fn cylinder_normal_is_radial() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();

        let n = cyl.normal(0.0, 0.0);
        // Normal should be perpendicular to axis
        assert!(approx_eq(n.dot(Vec3::new(0.0, 0.0, 1.0)), 0.0));
        // And unit length
        assert!(approx_eq(n.length(), 1.0));
    }

    #[test]
    fn cylinder_zero_radius_error() {
        assert!(
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.0,)
                .is_err()
        );
    }

    // ── Cone tests ────────────────────────────────────

    #[test]
    fn cone_apex() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            PI / 4.0,
        )
        .unwrap();

        // At v=0, all points should be at the apex
        let p = cone.evaluate(0.0, 0.0);
        assert!(point_approx_eq(p, Point3::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn cone_radius_at() {
        let half_angle = PI / 6.0; // 30 degrees
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            half_angle,
        )
        .unwrap();

        let r = cone.radius_at(10.0);
        assert!(approx_eq(r, 10.0 * half_angle.cos()));
    }

    #[test]
    fn cone_invalid_half_angle() {
        assert!(
            ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.0,)
                .is_err()
        );

        assert!(
            ConicalSurface::new(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                FRAC_PI_2,
            )
            .is_err()
        );
    }

    // ── Sphere tests ──────────────────────────────────

    #[test]
    fn sphere_north_pole() {
        let s = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0).unwrap();
        let p = s.evaluate(0.0, FRAC_PI_2);
        assert!(point_approx_eq(p, Point3::new(0.0, 0.0, 3.0)));
    }

    #[test]
    fn sphere_equator() {
        let s = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 2.0).unwrap();

        // u=0, v=0 → along x-axis
        let p = s.evaluate(0.0, 0.0);
        assert!(approx_eq(p.x(), 2.0));
        assert!(approx_eq(p.y(), 0.0));
        assert!(approx_eq(p.z(), 0.0));
    }

    #[test]
    fn sphere_normal_is_radial() {
        let sphere = SphericalSurface::new(Point3::new(1.0, 2.0, 3.0), 5.0).unwrap();

        let param_u = 1.0;
        let param_v = 0.5;
        let point = sphere.evaluate(param_u, param_v);
        let normal = sphere.normal(param_u, param_v);

        // Normal should point from center to surface point
        let expected_dir = point - sphere.center();
        let expected_norm = expected_dir.normalize().expect("non-zero");
        assert!(approx_eq(normal.dot(expected_norm), 1.0));
    }

    #[test]
    fn sphere_zero_radius_error() {
        assert!(SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 0.0).is_err());
    }

    // ── Torus tests ───────────────────────────────────

    #[test]
    fn torus_outer_point() {
        let t = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 2.0).unwrap();

        // u=0, v=0 → outermost point at (R+r, 0, 0) = (7, 0, 0) along x
        let p = t.evaluate(0.0, 0.0);
        assert!(approx_eq((p - Point3::new(0.0, 0.0, 0.0)).length(), 7.0));
    }

    #[test]
    fn torus_inner_point() {
        let t = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 2.0).unwrap();

        // v=π → innermost point at distance R-r = 3
        let p = t.evaluate(0.0, PI);
        assert!(approx_eq((p - Point3::new(0.0, 0.0, 0.0)).length(), 3.0));
    }

    #[test]
    fn torus_zero_radius_error() {
        assert!(ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 0.0, 1.0).is_err());
        assert!(ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 0.0).is_err());
    }

    // ── Revolution surface tests ──────────────────────

    #[test]
    fn revolution_cylinder_like() {
        // A revolution surface with constant radius = cylinder
        let rev = RevolutionSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            vec![3.0, 3.0],
            vec![0.0, 10.0],
        )
        .unwrap();

        let p = rev.evaluate(0.0, 0.0);
        assert!(approx_eq((p - Point3::new(0.0, 0.0, 0.0)).length(), 3.0));

        let p = rev.evaluate(0.0, 1.0);
        assert!(approx_eq(p.z(), 10.0));
    }

    // ── Additional coverage tests ─────────────────────

    #[test]
    fn cylinder_with_x_aligned_axis() {
        // Tests the other branch of the axis-candidate selection (a.x().abs() >= 0.9)
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 2.0)
                .unwrap();
        assert!(approx_eq(cyl.radius(), 2.0));
        let p = cyl.evaluate(0.0, 3.0);
        assert!(approx_eq(p.x(), 3.0));
    }

    #[test]
    fn cylinder_accessors() {
        let cyl =
            CylindricalSurface::new(Point3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, 1.0), 5.0)
                .unwrap();
        assert!(approx_eq(cyl.origin().x(), 1.0));
        assert!(approx_eq(cyl.axis().z(), 1.0));
        assert!(approx_eq(cyl.radius(), 5.0));
    }

    #[test]
    fn cone_normal() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            PI / 4.0,
        )
        .unwrap();
        let n = cone.normal(0.0, 1.0);
        // Normal should be non-zero and perpendicular to surface at that point
        assert!(n.length() > 0.5);
    }

    #[test]
    fn cone_accessors() {
        let cone = ConicalSurface::new(
            Point3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 0.0, 1.0),
            PI / 6.0,
        )
        .unwrap();
        assert!(point_approx_eq(cone.apex(), Point3::new(1.0, 2.0, 3.0)));
        assert!(approx_eq(cone.axis().z(), 1.0));
        assert!(approx_eq(cone.half_angle(), PI / 6.0));
    }

    #[test]
    fn cone_with_x_axis() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            PI / 4.0,
        )
        .unwrap();
        let p = cone.evaluate(0.0, 1.0);
        assert!((p - Point3::new(0.0, 0.0, 0.0)).length() > 0.5);
    }

    #[test]
    fn sphere_with_custom_axis() {
        let s =
            SphericalSurface::with_axis(Point3::new(0.0, 0.0, 0.0), 2.0, Vec3::new(1.0, 0.0, 0.0))
                .unwrap();
        assert!(approx_eq(s.radius(), 2.0));
        // North pole should be along x
        let p = s.evaluate(0.0, FRAC_PI_2);
        assert!(approx_eq((p - Point3::new(0.0, 0.0, 0.0)).length(), 2.0));
    }

    #[test]
    fn sphere_with_axis_zero_radius_error() {
        assert!(
            SphericalSurface::with_axis(Point3::new(0.0, 0.0, 0.0), 0.0, Vec3::new(0.0, 0.0, 1.0))
                .is_err()
        );
    }

    #[test]
    fn torus_normal() {
        let t = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 2.0).unwrap();
        let n = t.normal(0.0, 0.0);
        // At u=0, v=0 (outermost point), normal should point radially outward
        assert!(n.length() > 0.5);
    }

    #[test]
    fn torus_accessors() {
        let t = ToroidalSurface::new(Point3::new(1.0, 2.0, 3.0), 5.0, 2.0).unwrap();
        assert!(point_approx_eq(t.center(), Point3::new(1.0, 2.0, 3.0)));
        assert!(approx_eq(t.major_radius(), 5.0));
        assert!(approx_eq(t.minor_radius(), 2.0));
    }

    #[test]
    fn revolution_with_x_axis() {
        // Tests the other candidate branch for revolution surfaces
        let rev = RevolutionSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            vec![2.0, 2.0],
            vec![0.0, 5.0],
        )
        .unwrap();
        assert!(point_approx_eq(rev.origin(), Point3::new(0.0, 0.0, 0.0)));
        assert!(approx_eq(rev.axis().x(), 1.0));
    }

    #[test]
    fn revolution_empty_input_error() {
        assert!(
            RevolutionSurface::new(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                vec![],
                vec![],
            )
            .is_err()
        );
    }

    #[test]
    fn revolution_mismatched_lengths_error() {
        assert!(
            RevolutionSurface::new(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                vec![1.0, 2.0],
                vec![0.0],
            )
            .is_err()
        );
    }

    #[test]
    fn revolution_midpoint_evaluation() {
        let rev = RevolutionSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            vec![1.0, 3.0, 1.0],
            vec![0.0, 5.0, 10.0],
        )
        .unwrap();
        // At v=0.5 (midpoint of generatrix): radius should be 3.0, height 5.0
        let p = rev.evaluate(0.0, 0.5);
        assert!(approx_eq(p.z(), 5.0));
    }

    #[test]
    fn cylinder_evaluate_at_quarter_circle() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0)
                .unwrap();
        let p = cyl.evaluate(FRAC_PI_2, 0.0);
        // At u=π/2, should be at 90° around the cylinder
        assert!(approx_eq((p - Point3::new(0.0, 0.0, 0.0)).length(), 3.0));
    }

    #[test]
    fn cylinder_project_point_roundtrip() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0)
                .unwrap();

        // Evaluate at known parameters, then project back.
        let (u_orig, v_orig) = (1.0, 3.0);
        let pt = cyl.evaluate(u_orig, v_orig);
        let (u_proj, v_proj) = cyl.project_point(pt);

        assert!(
            approx_eq(u_proj, u_orig),
            "u should roundtrip: expected {u_orig}, got {u_proj}"
        );
        assert!(
            approx_eq(v_proj, v_orig),
            "v should roundtrip: expected {v_orig}, got {v_proj}"
        );
    }

    #[test]
    fn cylinder_project_external_point() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();

        // Project a point and verify its v-parameter (axial position).
        let (_, v) = cyl.project_point(Point3::new(5.0, 0.0, 2.0));
        assert!(approx_eq(v, 2.0), "v should be 2, got {v}");

        // Verify that evaluating at the projected parameters gives a point
        // on the cylinder surface closest to the original point.
        let (u, v) = cyl.project_point(Point3::new(3.0, 4.0, 7.0));
        let on_surface = cyl.evaluate(u, v);
        assert!(
            approx_eq(on_surface.z(), 7.0),
            "projected z should be 7.0, got {}",
            on_surface.z()
        );
        // The surface point should be at distance = radius from the axis.
        let radial_dist =
            (on_surface.x() * on_surface.x() + on_surface.y() * on_surface.y()).sqrt();
        assert!(
            approx_eq(radial_dist, 1.0),
            "surface point should be at radius 1.0, got {radial_dist}"
        );
    }

    #[test]
    fn sphere_project_point_roundtrip() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0).unwrap();

        let (u_orig, v_orig) = (1.5, 0.3);
        let pt = sphere.evaluate(u_orig, v_orig);
        let (u_proj, v_proj) = sphere.project_point(pt);

        assert!(
            approx_eq(u_proj, u_orig),
            "u should roundtrip: expected {u_orig}, got {u_proj}"
        );
        assert!(
            approx_eq(v_proj, v_orig),
            "v should roundtrip: expected {v_orig}, got {v_proj}"
        );
    }

    #[test]
    fn cone_project_point_roundtrip() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            PI / 6.0, // 30° half-angle
        )
        .unwrap();

        let (u_orig, v_orig) = (1.2, 2.5);
        let pt = cone.evaluate(u_orig, v_orig);
        let (u_proj, v_proj) = cone.project_point(pt);

        assert!(
            approx_eq(u_proj, u_orig),
            "cone u should roundtrip: expected {u_orig}, got {u_proj}"
        );
        assert!(
            approx_eq(v_proj, v_orig),
            "cone v should roundtrip: expected {v_orig}, got {v_proj}"
        );
    }

    #[test]
    fn cone_project_external_point() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            PI / 4.0, // 45° half-angle
        )
        .unwrap();

        // A point above the cone should project with correct v.
        let (_, v) = cone.project_point(Point3::new(5.0, 0.0, 5.0));
        assert!(v > 0.0, "v should be positive above cone apex, got {v}");
    }

    #[test]
    fn torus_project_point_roundtrip() {
        let torus = ToroidalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            3.0, // major radius
            1.0, // minor radius
        )
        .unwrap();

        let (u_orig, v_orig) = (0.8, 1.5);
        let pt = torus.evaluate(u_orig, v_orig);
        let (u_proj, v_proj) = torus.project_point(pt);

        assert!(
            approx_eq(u_proj, u_orig),
            "torus u should roundtrip: expected {u_orig}, got {u_proj}"
        );
        assert!(
            approx_eq(v_proj, v_orig),
            "torus v should roundtrip: expected {v_orig}, got {v_proj}"
        );
    }

    #[test]
    fn torus_project_external_point() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 1.0).unwrap();

        // Point on the outer equator (x=6, y=0, z=0) should project to u=0, v=0.
        let (u, v) = torus.project_point(Point3::new(6.0, 0.0, 0.0));
        assert!(approx_eq(u, 0.0), "outer equator u should be 0, got {u}");
        assert!(approx_eq(v, 0.0), "outer equator v should be 0, got {v}");

        // Point directly above the tube center at u=0 → v=π/2.
        let (_, v) = torus.project_point(Point3::new(5.0, 0.0, 1.5));
        assert!(
            approx_eq(v, FRAC_PI_2),
            "above tube center v should be π/2, got {v}"
        );
    }
}
