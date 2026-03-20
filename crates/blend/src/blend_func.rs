//! Blend constraint functions for fillet and chamfer.
//!
//! Each blend type encodes a system of 4 equations in 4 unknowns `(u1, v1, u2, v2)`
//! that the Newton-Raphson walker solves at each step to trace the blend surface.
//!
//! The constraint system ensures:
//! - **Planarity**: the midpoint of the two contact points lies on the section plane
//! - **Equidistance**: the rolling ball center is equidistant from both surfaces
//!
//! # Blend types
//!
//! - [`ConstRadBlend`] — constant-radius fillet (OCCT `BlendFunc_ConstRad`)
//! - [`EvolRadBlend`] — variable-radius fillet using a [`RadiusLaw`]
//! - [`ChamferBlend`] — two-distance chamfer
//! - [`ChamferAngleBlend`] — distance-angle chamfer

use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};

use crate::radius_law::RadiusLaw;
use crate::section::CircSection;

/// Surface parameters for a blend contact pair.
#[derive(Debug, Clone, Copy)]
pub struct BlendParams {
    /// Parameter u on surface 1.
    pub u1: f64,
    /// Parameter v on surface 1.
    pub v1: f64,
    /// Parameter u on surface 2.
    pub u2: f64,
    /// Parameter v on surface 2.
    pub v2: f64,
}

/// Geometric context for evaluating blend constraints at a spine station.
#[derive(Debug, Clone, Copy)]
pub struct BlendContext {
    /// Point on the guide (spine) curve at the current parameter.
    pub guide_point: Point3,
    /// Section plane normal (spine tangent direction, unit vector).
    pub nplan: Vec3,
    /// Spine parameter (arc-length fraction in \[0, 1\]).
    pub t: f64,
}

/// Trait for blend constraint function systems.
///
/// Each implementation provides the residual vector, analytic Jacobian,
/// and cross-section extraction for a particular blend type.
pub trait BlendFunction {
    /// Evaluate the 4-component constraint residual.
    ///
    /// Returns `[f1, f2, f3, f4]` where:
    /// - `f1` = planarity constraint (midpoint on section plane)
    /// - `f2, f3, f4` = equidistance constraint (ball center agreement)
    #[must_use]
    fn value(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [f64; 4];

    /// Compute the 4x4 Jacobian matrix `∂f/∂(u1, v1, u2, v2)`.
    ///
    /// Row `i` corresponds to constraint `f_i`, column `j` to variable `j`
    /// in order `(u1, v1, u2, v2)`.
    #[must_use]
    fn jacobian(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [[f64; 4]; 4];

    /// Extract the cross-section geometry at the current solution.
    #[must_use]
    fn section(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> CircSection;
}

/// Project a surface normal into the section plane perpendicular to `nplan`.
///
/// Computes the component of `normal` that is perpendicular to `nplan`,
/// then normalizes it. This is the "rolling ball" direction: the surface
/// normal projected into the section plane.
///
/// Formula: `cross = nplan × normal`, then `npn = cross × nplan / |cross|`.
/// The sign is chosen so that `npn` points away from the surface (into the
/// fillet region).
///
/// Returns `Vec3::new(0.0, 0.0, 0.0)` if the normal is parallel to `nplan`
/// (degenerate case).
#[must_use]
pub fn project_normal_to_section(normal: Vec3, nplan: Vec3) -> Vec3 {
    let cross = nplan.cross(normal);
    let len = cross.length();
    if len < 1e-15 {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    let npn = cross.cross(nplan);
    let npn_len = npn.length();
    if npn_len < 1e-15 {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    // Sign: npn should point in the same half-space as the surface normal
    // relative to the section plane.
    let sign = if normal.dot(npn) >= 0.0 { 1.0 } else { -1.0 };
    Vec3::new(
        sign * npn.x() / npn_len,
        sign * npn.y() / npn_len,
        sign * npn.z() / npn_len,
    )
}

// ---------------------------------------------------------------------------
// Constant-radius fillet
// ---------------------------------------------------------------------------

/// Constant-radius fillet blend function.
///
/// Implements the OCCT `BlendFunc_ConstRad` constraint system:
/// - `f1 = nplan · (midpoint - guide)` (planarity)
/// - `f2..f4 = (P1 + R·npn1) - (P2 + R·npn2)` (equidistance)
#[derive(Debug, Clone)]
pub struct ConstRadBlend {
    /// Fillet radius.
    pub radius: f64,
}

impl ConstRadBlend {
    /// Compute the ball center from surface 1 side: `P1 + R * npn1`.
    fn ball_center_from(p: Point3, npn: Vec3, r: f64) -> Point3 {
        Point3::new(
            p.x() + r * npn.x(),
            p.y() + r * npn.y(),
            p.z() + r * npn.z(),
        )
    }

    /// Evaluate the residual for a given radius (shared with `EvolRadBlend`).
    fn value_with_radius(
        r: f64,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [f64; 4] {
        let p1 = surf1.evaluate(params.u1, params.v1);
        let p2 = surf2.evaluate(params.u2, params.v2);
        let n1 = surf1.normal(params.u1, params.v1);
        let n2 = surf2.normal(params.u2, params.v2);

        let npn1 = project_normal_to_section(n1, ctx.nplan);
        let npn2 = project_normal_to_section(n2, ctx.nplan);

        // f1: planarity — midpoint lies on the section plane through guide_point
        let mid = Point3::new(
            (p1.x() + p2.x()) * 0.5,
            (p1.y() + p2.y()) * 0.5,
            (p1.z() + p2.z()) * 0.5,
        );
        let diff = mid - ctx.guide_point;
        let f1 = ctx.nplan.dot(Vec3::new(diff.x(), diff.y(), diff.z()));

        // f2..f4: equidistance — ball center from S1 == ball center from S2
        let c1 = Self::ball_center_from(p1, npn1, r);
        let c2 = Self::ball_center_from(p2, npn2, r);

        [f1, c1.x() - c2.x(), c1.y() - c2.y(), c1.z() - c2.z()]
    }

    /// Compute the Jacobian for a given radius (shared with `EvolRadBlend`).
    ///
    /// First-order Jacobian ignoring `∂npn/∂u` terms (standard OCCT approach).
    fn jacobian_with_radius(
        r: f64,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [[f64; 4]; 4] {
        let du1 = surf1.partial_u(params.u1, params.v1);
        let dv1 = surf1.partial_v(params.u1, params.v1);
        let du2 = surf2.partial_u(params.u2, params.v2);
        let dv2 = surf2.partial_v(params.u2, params.v2);

        // dnpn/du terms for improved convergence
        let h = 1e-7;
        let dnpn1_du1 = Self::finite_diff_npn(surf1, params.u1, params.v1, h, 0.0, ctx.nplan);
        let dnpn1_dv1 = Self::finite_diff_npn(surf1, params.u1, params.v1, 0.0, h, ctx.nplan);
        let dnpn2_du2 = Self::finite_diff_npn(surf2, params.u2, params.v2, h, 0.0, ctx.nplan);
        let dnpn2_dv2 = Self::finite_diff_npn(surf2, params.u2, params.v2, 0.0, h, ctx.nplan);

        let nplan = ctx.nplan;

        // Row 0: ∂f1/∂(u1,v1,u2,v2)
        // f1 = nplan · (midpoint - guide)
        // ∂f1/∂u1 = nplan · (∂P1/∂u1) / 2
        let row0 = [
            0.5 * nplan.dot(du1),
            0.5 * nplan.dot(dv1),
            0.5 * nplan.dot(du2),
            0.5 * nplan.dot(dv2),
        ];

        // Rows 1-3: ∂(c1 - c2)/∂(u1,v1,u2,v2)
        // c1 = P1 + R*npn1, c2 = P2 + R*npn2
        // ∂c1/∂u1 = ∂P1/∂u1 + R*∂npn1/∂u1
        // ∂c2/∂u2 = ∂P2/∂u2 + R*∂npn2/∂u2
        // Column 0 (u1): ∂P1/∂u1 + R*∂npn1/∂u1
        // Column 1 (v1): ∂P1/∂v1 + R*∂npn1/∂v1
        // Column 2 (u2): -(∂P2/∂u2 + R*∂npn2/∂u2)
        // Column 3 (v2): -(∂P2/∂v2 + R*∂npn2/∂v2)

        let col0 = Vec3::new(
            du1.x() + r * dnpn1_du1.x(),
            du1.y() + r * dnpn1_du1.y(),
            du1.z() + r * dnpn1_du1.z(),
        );
        let col1 = Vec3::new(
            dv1.x() + r * dnpn1_dv1.x(),
            dv1.y() + r * dnpn1_dv1.y(),
            dv1.z() + r * dnpn1_dv1.z(),
        );
        let col2 = Vec3::new(
            -(du2.x() + r * dnpn2_du2.x()),
            -(du2.y() + r * dnpn2_du2.y()),
            -(du2.z() + r * dnpn2_du2.z()),
        );
        let col3 = Vec3::new(
            -(dv2.x() + r * dnpn2_dv2.x()),
            -(dv2.y() + r * dnpn2_dv2.y()),
            -(dv2.z() + r * dnpn2_dv2.z()),
        );

        [
            row0,
            [col0.x(), col1.x(), col2.x(), col3.x()],
            [col0.y(), col1.y(), col2.y(), col3.y()],
            [col0.z(), col1.z(), col2.z(), col3.z()],
        ]
    }

    /// Finite-difference approximation of ∂npn/∂u or ∂npn/∂v.
    pub(crate) fn finite_diff_npn(
        surf: &dyn ParametricSurface,
        u: f64,
        v: f64,
        du: f64,
        dv: f64,
        nplan: Vec3,
    ) -> Vec3 {
        let n_plus = surf.normal(u + du, v + dv);
        let n_minus = surf.normal(u - du, v - dv);
        let npn_plus = project_normal_to_section(n_plus, nplan);
        let npn_minus = project_normal_to_section(n_minus, nplan);

        let h2 = 2.0 * (du * du + dv * dv).sqrt();
        if h2 < 1e-30 {
            return Vec3::new(0.0, 0.0, 0.0);
        }
        Vec3::new(
            (npn_plus.x() - npn_minus.x()) / h2,
            (npn_plus.y() - npn_minus.y()) / h2,
            (npn_plus.z() - npn_minus.z()) / h2,
        )
    }

    /// Build a `CircSection` from parameters (shared with `EvolRadBlend`).
    fn section_with_radius(
        r: f64,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> CircSection {
        let p1 = surf1.evaluate(params.u1, params.v1);
        let p2 = surf2.evaluate(params.u2, params.v2);
        let n1 = surf1.normal(params.u1, params.v1);
        let npn1 = project_normal_to_section(n1, ctx.nplan);
        let center = Self::ball_center_from(p1, npn1, r);

        CircSection {
            p1,
            p2,
            center,
            radius: r,
            uv1: (params.u1, params.v1),
            uv2: (params.u2, params.v2),
            t: ctx.t,
        }
    }
}

impl BlendFunction for ConstRadBlend {
    fn value(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [f64; 4] {
        Self::value_with_radius(self.radius, surf1, surf2, params, ctx)
    }

    fn jacobian(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [[f64; 4]; 4] {
        Self::jacobian_with_radius(self.radius, surf1, surf2, params, ctx)
    }

    fn section(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> CircSection {
        Self::section_with_radius(self.radius, surf1, surf2, params, ctx)
    }
}

// ---------------------------------------------------------------------------
// Evolving-radius fillet
// ---------------------------------------------------------------------------

/// Variable-radius fillet blend function.
///
/// Delegates to the [`ConstRadBlend`] math with the radius evaluated from
/// a [`RadiusLaw`] at the current spine parameter `ctx.t`.
#[derive(Debug)]
pub struct EvolRadBlend {
    /// The radius law governing how the fillet radius varies along the spine.
    pub law: RadiusLaw,
}

impl BlendFunction for EvolRadBlend {
    fn value(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [f64; 4] {
        let r = self.law.evaluate(ctx.t);
        ConstRadBlend::value_with_radius(r, surf1, surf2, params, ctx)
    }

    fn jacobian(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [[f64; 4]; 4] {
        let r = self.law.evaluate(ctx.t);
        ConstRadBlend::jacobian_with_radius(r, surf1, surf2, params, ctx)
    }

    fn section(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> CircSection {
        let r = self.law.evaluate(ctx.t);
        ConstRadBlend::section_with_radius(r, surf1, surf2, params, ctx)
    }
}

// ---------------------------------------------------------------------------
// Two-distance chamfer
// ---------------------------------------------------------------------------

/// Two-distance chamfer blend function.
///
/// The chamfer is defined by two distances `d1` and `d2` from the edge,
/// measured along each surface. The constraint system is:
/// - `f1 = nplan · (midpoint - guide)` (planarity)
/// - `f2..f4 = (P1 + d1·npn1) - (P2 + d2·npn2)` (offset matching)
#[derive(Debug, Clone)]
pub struct ChamferBlend {
    /// Distance from edge along surface 1.
    pub d1: f64,
    /// Distance from edge along surface 2.
    pub d2: f64,
}

impl BlendFunction for ChamferBlend {
    fn value(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [f64; 4] {
        let p1 = surf1.evaluate(params.u1, params.v1);
        let p2 = surf2.evaluate(params.u2, params.v2);
        let n1 = surf1.normal(params.u1, params.v1);
        let n2 = surf2.normal(params.u2, params.v2);

        let npn1 = project_normal_to_section(n1, ctx.nplan);
        let npn2 = project_normal_to_section(n2, ctx.nplan);

        let mid = Point3::new(
            (p1.x() + p2.x()) * 0.5,
            (p1.y() + p2.y()) * 0.5,
            (p1.z() + p2.z()) * 0.5,
        );
        let diff = mid - ctx.guide_point;
        let f1 = ctx.nplan.dot(Vec3::new(diff.x(), diff.y(), diff.z()));

        let c1 = Point3::new(
            p1.x() + self.d1 * npn1.x(),
            p1.y() + self.d1 * npn1.y(),
            p1.z() + self.d1 * npn1.z(),
        );
        let c2 = Point3::new(
            p2.x() + self.d2 * npn2.x(),
            p2.y() + self.d2 * npn2.y(),
            p2.z() + self.d2 * npn2.z(),
        );

        [f1, c1.x() - c2.x(), c1.y() - c2.y(), c1.z() - c2.z()]
    }

    fn jacobian(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [[f64; 4]; 4] {
        let du1 = surf1.partial_u(params.u1, params.v1);
        let dv1 = surf1.partial_v(params.u1, params.v1);
        let du2 = surf2.partial_u(params.u2, params.v2);
        let dv2 = surf2.partial_v(params.u2, params.v2);

        // Include finite-difference ∂npn/∂u terms for convergence on
        // curved surfaces (same approach as ConstRadBlend).
        let h = 1e-7;
        let dnpn1_du1 =
            ConstRadBlend::finite_diff_npn(surf1, params.u1, params.v1, h, 0.0, ctx.nplan);
        let dnpn1_dv1 =
            ConstRadBlend::finite_diff_npn(surf1, params.u1, params.v1, 0.0, h, ctx.nplan);
        let dnpn2_du2 =
            ConstRadBlend::finite_diff_npn(surf2, params.u2, params.v2, h, 0.0, ctx.nplan);
        let dnpn2_dv2 =
            ConstRadBlend::finite_diff_npn(surf2, params.u2, params.v2, 0.0, h, ctx.nplan);

        let nplan = ctx.nplan;

        let row0 = [
            0.5 * nplan.dot(du1),
            0.5 * nplan.dot(dv1),
            0.5 * nplan.dot(du2),
            0.5 * nplan.dot(dv2),
        ];

        // Rows 1-3: ∂(c1 - c2)/∂(u1,v1,u2,v2)
        // c1 = P1 + d1*npn1, c2 = P2 + d2*npn2
        let col0 = Vec3::new(
            du1.x() + self.d1 * dnpn1_du1.x(),
            du1.y() + self.d1 * dnpn1_du1.y(),
            du1.z() + self.d1 * dnpn1_du1.z(),
        );
        let col1 = Vec3::new(
            dv1.x() + self.d1 * dnpn1_dv1.x(),
            dv1.y() + self.d1 * dnpn1_dv1.y(),
            dv1.z() + self.d1 * dnpn1_dv1.z(),
        );
        let col2 = Vec3::new(
            -(du2.x() + self.d2 * dnpn2_du2.x()),
            -(du2.y() + self.d2 * dnpn2_du2.y()),
            -(du2.z() + self.d2 * dnpn2_du2.z()),
        );
        let col3 = Vec3::new(
            -(dv2.x() + self.d2 * dnpn2_dv2.x()),
            -(dv2.y() + self.d2 * dnpn2_dv2.y()),
            -(dv2.z() + self.d2 * dnpn2_dv2.z()),
        );

        [
            row0,
            [col0.x(), col1.x(), col2.x(), col3.x()],
            [col0.y(), col1.y(), col2.y(), col3.y()],
            [col0.z(), col1.z(), col2.z(), col3.z()],
        ]
    }

    fn section(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> CircSection {
        let p1 = surf1.evaluate(params.u1, params.v1);
        let p2 = surf2.evaluate(params.u2, params.v2);

        // Chamfer has no rolling ball — use midpoint as "center", radius = 0
        let center = Point3::new(
            (p1.x() + p2.x()) * 0.5,
            (p1.y() + p2.y()) * 0.5,
            (p1.z() + p2.z()) * 0.5,
        );

        CircSection {
            p1,
            p2,
            center,
            radius: 0.0,
            uv1: (params.u1, params.v1),
            uv2: (params.u2, params.v2),
            t: ctx.t,
        }
    }
}

// ---------------------------------------------------------------------------
// Distance-angle chamfer
// ---------------------------------------------------------------------------

/// Distance-angle chamfer blend function.
///
/// The chamfer is defined by a distance `d` from the edge along surface 1,
/// and an angle `a` from surface 1. The effective distance along surface 2
/// is `d * tan(a)`.
#[derive(Debug, Clone)]
pub struct ChamferAngleBlend {
    /// Distance from edge along surface 1.
    pub distance: f64,
    /// Angle from surface 1 (radians).
    pub angle: f64,
}

impl BlendFunction for ChamferAngleBlend {
    fn value(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [f64; 4] {
        let d2 = self.distance * self.angle.tan();
        let chamfer = ChamferBlend {
            d1: self.distance,
            d2,
        };
        chamfer.value(surf1, surf2, params, ctx)
    }

    fn jacobian(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> [[f64; 4]; 4] {
        let d2 = self.distance * self.angle.tan();
        let chamfer = ChamferBlend {
            d1: self.distance,
            d2,
        };
        chamfer.jacobian(surf1, surf2, params, ctx)
    }

    fn section(
        &self,
        surf1: &dyn ParametricSurface,
        surf2: &dyn ParametricSurface,
        params: &BlendParams,
        ctx: &BlendContext,
    ) -> CircSection {
        let d2 = self.distance * self.angle.tan();
        let chamfer = ChamferBlend {
            d1: self.distance,
            d2,
        };
        chamfer.section(surf1, surf2, params, ctx)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// A simple infinite plane for testing: `P(u,v) = origin + u*u_dir + v*v_dir`.
    struct TestPlane {
        origin: Point3,
        u_dir: Vec3,
        v_dir: Vec3,
        norm: Vec3,
    }

    impl ParametricSurface for TestPlane {
        fn evaluate(&self, u: f64, v: f64) -> Point3 {
            Point3::new(
                self.origin.x() + u * self.u_dir.x() + v * self.v_dir.x(),
                self.origin.y() + u * self.u_dir.y() + v * self.v_dir.y(),
                self.origin.z() + u * self.u_dir.z() + v * self.v_dir.z(),
            )
        }

        fn normal(&self, _u: f64, _v: f64) -> Vec3 {
            self.norm
        }

        fn project_point(&self, point: Point3) -> (f64, f64) {
            let d = point - self.origin;
            let dv = Vec3::new(d.x(), d.y(), d.z());
            (dv.dot(self.u_dir), dv.dot(self.v_dir))
        }

        fn partial_u(&self, _u: f64, _v: f64) -> Vec3 {
            self.u_dir
        }

        fn partial_v(&self, _u: f64, _v: f64) -> Vec3 {
            self.v_dir
        }
    }

    /// Two planes at 90 degrees (inside corner of a box):
    /// - Plane 1: z=0 plane (floor), normal = (0,0,1) pointing into fillet
    /// - Plane 2: x=0 plane (wall), normal = (1,0,0) pointing into fillet
    ///
    /// Edge along y-axis. Spine tangent = (0,1,0).
    ///
    /// Normal orientation encodes which side the rolling ball is on.
    /// Both normals point toward the ball center (into the concavity).
    ///
    /// For radius R, the known solution is:
    /// - Contact on plane 1: (R, y, 0) → u1=R, v1=y
    /// - Contact on plane 2: (0, y, R) → u2=y, v2=R
    /// - Ball center: (R, y, R)
    fn make_test_planes() -> (TestPlane, TestPlane) {
        let plane1 = TestPlane {
            origin: Point3::new(0.0, 0.0, 0.0),
            u_dir: Vec3::new(1.0, 0.0, 0.0),
            v_dir: Vec3::new(0.0, 1.0, 0.0),
            norm: Vec3::new(0.0, 0.0, 1.0),
        };
        let plane2 = TestPlane {
            origin: Point3::new(0.0, 0.0, 0.0),
            u_dir: Vec3::new(0.0, 1.0, 0.0),
            v_dir: Vec3::new(0.0, 0.0, 1.0),
            norm: Vec3::new(1.0, 0.0, 0.0),
        };
        (plane1, plane2)
    }

    fn make_test_context(y: f64) -> BlendContext {
        BlendContext {
            guide_point: Point3::new(0.0, y, 0.0),
            nplan: Vec3::new(0.0, 1.0, 0.0),
            t: 0.5,
        }
    }

    #[test]
    fn const_rad_residual_zero_at_known_solution() {
        let (p1, p2) = make_test_planes();
        let r = 3.0;
        let y = 5.0;

        let blend = ConstRadBlend { radius: r };
        let ctx = make_test_context(y);

        // Known solution: P1 = (R, y, 0), P2 = (0, y, R)
        // On plane1: u1=R, v1=y → P1 = (R, y, 0)
        // On plane2: u2=y, v2=R → P2 = (0, y, R)
        let params = BlendParams {
            u1: r,
            v1: y,
            u2: y,
            v2: r,
        };

        let res = blend.value(&p1, &p2, &params, &ctx);
        let norm = (res[0] * res[0] + res[1] * res[1] + res[2] * res[2] + res[3] * res[3]).sqrt();
        assert!(
            norm < 1e-10,
            "residual should be ~0 at known solution, got {res:?} (norm={norm})"
        );
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        let (p1, p2) = make_test_planes();
        let r = 2.0;
        let y = 3.0;

        let blend = ConstRadBlend { radius: r };
        let ctx = make_test_context(y);

        // Use a nearby (not exact) solution to test Jacobian at a non-zero residual
        let params = BlendParams {
            u1: r + 0.1,
            v1: y - 0.05,
            u2: y + 0.05,
            v2: r - 0.1,
        };

        let jac = blend.jacobian(&p1, &p2, &params, &ctx);
        let h = 1e-6;

        // Central differences for each column
        let perturbations: [(usize, &str); 4] = [(0, "u1"), (1, "v1"), (2, "u2"), (3, "v2")];

        for &(col, name) in &perturbations {
            let mut p_plus = params;
            let mut p_minus = params;
            match col {
                0 => {
                    p_plus.u1 += h;
                    p_minus.u1 -= h;
                }
                1 => {
                    p_plus.v1 += h;
                    p_minus.v1 -= h;
                }
                2 => {
                    p_plus.u2 += h;
                    p_minus.u2 -= h;
                }
                3 => {
                    p_plus.v2 += h;
                    p_minus.v2 -= h;
                }
                _ => unreachable!(),
            }

            let f_plus = blend.value(&p1, &p2, &p_plus, &ctx);
            let f_minus = blend.value(&p1, &p2, &p_minus, &ctx);

            for row in 0..4 {
                let fd = (f_plus[row] - f_minus[row]) / (2.0 * h);
                let analytic = jac[row][col];
                let err = (fd - analytic).abs();
                assert!(
                    err < 1e-4,
                    "Jacobian[{row}][{col}] ({name}): analytic={analytic:.8}, fd={fd:.8}, err={err:.8}"
                );
            }
        }
    }

    #[test]
    fn project_normal_perpendicular_to_nplan() {
        let nplan = Vec3::new(0.0, 1.0, 0.0);

        // Test several normals
        let normals = [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-0.5, 0.3, 0.8),
        ];

        for n in &normals {
            let npn = project_normal_to_section(*n, nplan);
            let dot = npn.dot(nplan).abs();
            assert!(
                dot < 1e-12,
                "projected normal should be perpendicular to nplan, dot={dot} for normal={n:?}"
            );

            // Should be unit length (unless degenerate)
            let len = npn.length();
            if len > 1e-10 {
                assert!(
                    (len - 1.0).abs() < 1e-12,
                    "projected normal should be unit length, got {len}"
                );
            }
        }
    }

    #[test]
    fn project_normal_degenerate_parallel() {
        let nplan = Vec3::new(0.0, 1.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0); // parallel to nplan

        let npn = project_normal_to_section(normal, nplan);
        assert!(
            npn.length() < 1e-10,
            "parallel normal should project to zero vector"
        );
    }

    #[test]
    fn section_has_correct_radius() {
        let (p1, p2) = make_test_planes();
        let r = 4.0;
        let y = 2.0;

        let blend = ConstRadBlend { radius: r };
        let ctx = make_test_context(y);

        let params = BlendParams {
            u1: r,
            v1: y,
            u2: y,
            v2: r,
        };

        let sec = blend.section(&p1, &p2, &params, &ctx);
        assert!(
            (sec.radius - r).abs() < f64::EPSILON,
            "section radius should match input, got {}",
            sec.radius
        );

        // Verify contact points
        assert!((sec.p1.x() - r).abs() < 1e-12);
        assert!((sec.p1.y() - y).abs() < 1e-12);
        assert!(sec.p1.z().abs() < 1e-12);

        assert!(sec.p2.x().abs() < 1e-12);
        assert!((sec.p2.y() - y).abs() < 1e-12);
        assert!((sec.p2.z() - r).abs() < 1e-12);

        // Verify center is at (R, y, R) — equidistant from both contact points
        // Ball center from plane1: P1 + R*npn1 = (R, y, 0) + R*(0,0,1) = (R, y, R)
        assert!((sec.center.x() - r).abs() < 1e-12);
        assert!((sec.center.y() - y).abs() < 1e-12);
        assert!((sec.center.z() - r).abs() < 1e-12);
    }

    #[test]
    fn evol_rad_matches_const_at_constant_law() {
        let (p1, p2) = make_test_planes();
        let r = 3.0;
        let y = 5.0;
        let ctx = make_test_context(y);
        let params = BlendParams {
            u1: r,
            v1: y,
            u2: y,
            v2: r,
        };

        let const_blend = ConstRadBlend { radius: r };
        let evol_blend = EvolRadBlend {
            law: RadiusLaw::Constant(r),
        };

        let res_const = const_blend.value(&p1, &p2, &params, &ctx);
        let res_evol = evol_blend.value(&p1, &p2, &params, &ctx);

        for i in 0..4 {
            assert!(
                (res_const[i] - res_evol[i]).abs() < 1e-14,
                "EvolRad with constant law should match ConstRad"
            );
        }
    }

    #[test]
    fn chamfer_residual_at_known_solution() {
        let (p1, p2) = make_test_planes();
        let d1 = 3.0;
        let d2 = 3.0;
        let y = 5.0;

        let blend = ChamferBlend { d1, d2 };
        let ctx = make_test_context(y);

        // Same as fillet: contact at (d1, y, 0) and (0, y, d2)
        // npn1 for plane1 (normal=(0,0,1), nplan=(0,1,0)):
        //   cross = (0,1,0)×(0,0,1) = (1,0,0)
        //   npn = (1,0,0)×(0,1,0) = (0,0,1) → same as normal, sign positive
        //   Wait: cross×nplan = (1,0,0)×(0,1,0) = (0,0,1)
        //   npn1 = (0,0,1), length=1, dot(normal, npn)=1 → keep sign → (0,0,1)
        // npn2 for plane2 (normal=(-1,0,0)):
        //   cross = (0,1,0)×(-1,0,0) = (0,0,1)
        //   npn = (0,0,1)×(0,1,0) = ... wait let me recalc
        //   cross = nplan × normal = (0,1,0)×(-1,0,0) = (0·0 - 0·0, 0·(-1) - 1·0, 1·0 - 0·(-1)) = (0,0,1)
        //   npn = cross × nplan = (0,0,1)×(0,1,0) = (0·0-1·1, 1·0-0·0, 0·1-0·0) = (-1,0,0)
        //   dot(normal, npn) = (-1)·(-1)+0+0 = 1 ≥ 0 → keep sign → (-1,0,0)
        // c1 = P1 + d1*npn1 = (d1, y, 0) + d1*(0,0,1) = (d1, y, d1)
        // c2 = P2 + d2*npn2 = (0, y, d2) + d2*(-1,0,0) = (-d2, y, d2)
        // c1 - c2 = (d1+d2, 0, d1-d2)
        // For symmetric chamfer (d1=d2): (2d, 0, 0) ≠ 0
        // So for chamfer, the constraint is different from fillet!
        // The chamfer solution would be where c1 = c2.
        // (d1, y, d1) = (-d2, y, d2) → d1 = -d2 (impossible for positive distances)
        // This means the symmetric chamfer on 90° planes doesn't have a solution with this
        // formulation. The chamfer formulation works differently from fillet.
        // For chamfer, let me just verify the Jacobian matches FD instead.

        // With npn1=(0,0,1), npn2=(1,0,0):
        // c1 = (u1, v1, d1), c2 = (d2, u2, v2)
        // c1=c2: u1=d2, v1=u2, d1=v2
        // Planarity: mid_y = y → v1 = y (since v1=u2)
        // So: u1=d2, v1=y, u2=y, v2=d1

        let params = BlendParams {
            u1: d2,
            v1: y,
            u2: y,
            v2: d1,
        };

        let res = blend.value(&p1, &p2, &params, &ctx);
        let norm = (res[0] * res[0] + res[1] * res[1] + res[2] * res[2] + res[3] * res[3]).sqrt();
        assert!(
            norm < 1e-10,
            "chamfer residual should be ~0 at known solution, got {res:?} (norm={norm})"
        );
    }

    #[test]
    fn chamfer_angle_delegates_correctly() {
        let (p1, p2) = make_test_planes();
        let d = 3.0;
        let angle = std::f64::consts::FRAC_PI_4; // 45° → tan = 1 → d2 = d

        let angle_blend = ChamferAngleBlend { distance: d, angle };
        let equiv_blend = ChamferBlend {
            d1: d,
            d2: d * angle.tan(),
        };
        let ctx = make_test_context(5.0);
        let params = BlendParams {
            u1: 1.0,
            v1: 2.0,
            u2: 3.0,
            v2: 4.0,
        };

        let res_angle = angle_blend.value(&p1, &p2, &params, &ctx);
        let res_chamfer = equiv_blend.value(&p1, &p2, &params, &ctx);

        for i in 0..4 {
            assert!(
                (res_angle[i] - res_chamfer[i]).abs() < 1e-14,
                "ChamferAngle should delegate to ChamferBlend with d2=d*tan(a)"
            );
        }
    }
}
